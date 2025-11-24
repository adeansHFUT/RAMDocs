# -*- coding: utf-8 -*-
"""
MADAM-RAG with DyPRAG-style LoRA-as-message (delta inject/remove)
+ agent context switch (aggregator / history / both / none)
+ message_transport options:
    'text',
    'dyprag', 'dyprag-combine',
    'full-dyprag', 'full-dyprag-combine'
+ per-round timing & GPU model logging
+ DEBUG 日志（标注 agents->aggregator / aggregator->agents 的通信路径）
+ 本次运行唯一前缀到输出文件名（避免相同参数的不同运行互相覆盖）
- 移除断点续跑逻辑（每次运行都会完整评测数据集并覆盖输出）
+ 可选“参数知识代理”（param agent）：用模型自身参数生成一篇伪文档作为额外 agent
+ history + DyPRAG 模式下的置信度抑制：低置信度 agent 的 LoRA delta 在下一轮被缩放

依赖：
  - utils.py      : get_model, delta_inject, delta_remove
  - projector.py  : ParameterTranslator
"""

import argparse
import os
import re
import gc
import json
import time
import torch
import string
from typing import List
from tqdm import tqdm

from transformers import set_seed
from utils import get_model, delta_inject, delta_remove
from projector import ParameterTranslator
from peft import LoraConfig, get_peft_model


# -----------------------------
# Simple timestamp logger
# -----------------------------
def _ts():
    return time.strftime("[%Y-%m-%d %H:%M:%S]")


def _log(debug_on: bool, *parts):
    if debug_on:
        print(_ts(), *parts, flush=True)


def _make_run_prefix() -> str:
    """
    为输出文件名生成一次性前缀：
    优先 SLURM_JOB_ID / JOB_ID，其次时间戳 + 进程号。
    """
    job = os.getenv("SLURM_JOB_ID") or os.getenv("JOB_ID")
    ts = time.strftime("%Y%m%d-%H%M%S")
    if job:
        return f"run-{ts}-{job}"
    return f"run-{ts}-{os.getpid()}"


def _peft_wrap(model, args):
    """Wrap model as PeftModel shell (no adapters loaded)."""
    peft_cfg = LoraConfig(
        task_type="CAUSAL_LM",
        target_modules=['down_proj', 'gate_proj', 'up_proj'],
        inference_mode=True,
        r=args.lora_rank,
        lora_alpha=32,
        lora_dropout=0.0,
    )
    wrapped = get_peft_model(model, peft_cfg)
    wrapped.eval()
    return wrapped


def _detect_gpu_name() -> str:
    if torch.cuda.is_available():
        try:
            return torch.cuda.get_device_name(0)
        except Exception:
            return "cuda"
    return "cpu"


# -----------------------------
# Text utilities & evaluation
# -----------------------------

def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        return ''.join(ch for ch in text if ch not in string.punctuation)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def parse_aggregator_answers(agg_text: str):
    """Prefer JSON; fallback to 'All Correct Answers: [...]' format."""
    try:
        m = re.search(r'\{.*\}', agg_text, flags=re.S)
        if m:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict) and "answers" in obj:
                return [a.strip() for a in obj["answers"] if isinstance(a, str)]
    except Exception:
        pass

    m = re.search(r'All\s*Correct\s*Answers\s*:\s*\[(.*?)\]', agg_text, flags=re.I | re.S)
    if not m:
        return []
    inside = m.group(1)
    try:
        return [str(x).strip() for x in json.loads("[" + inside + "]")]
    except Exception:
        return [p.strip(" '\"\n. ") for p in inside.split(",") if p.strip()]


def em_score(pred_answers, gold_answers):
    P = {normalize_answer(x) for x in (pred_answers or [])}
    G = {normalize_answer(x) for x in (gold_answers or [])}
    return int(P == G)


# -----------------------------
# LLM calling helpers
# -----------------------------

def _to_gen_cfg_dict(generation_config):
    if generation_config is None:
        return {}
    if isinstance(generation_config, dict):
        return dict(generation_config)
    if hasattr(generation_config, "to_dict"):
        return generation_config.to_dict()
    return {}


def call_llm_chat(
    messages,
    model,
    tokenizer,
    generation_config,
    max_new_tokens: int = 128,
    return_confidence: bool = False,
):
    """
    当 return_confidence=True 时，除了文本，还会返回一个 [0,1] 区间内的
    生成置信度（基于生成 token 的平均概率）。
    """
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = tokenizer(prompt, return_tensors="pt")
    dev = next(model.parameters()).device
    inputs = {k: v.to(dev) for k, v in inputs.items()}

    gen_cfg = _to_gen_cfg_dict(generation_config)
    gen_cfg.update(dict(
        do_sample=False, temperature=None, top_p=None, top_k=None,
        num_beams=1, max_new_tokens=max_new_tokens,
        return_dict_in_generate=True,
        pad_token_id=getattr(tokenizer, "pad_token_id", getattr(tokenizer, "eos_token_id", None)),
        eos_token_id=getattr(tokenizer, "eos_token_id", None),
    ))
    if return_confidence:
        gen_cfg["output_scores"] = True

    with torch.no_grad():
        out = model.generate(**inputs, **gen_cfg)

    seq = out.sequences[0]
    cut = inputs["input_ids"].shape[1]
    gen_ids = seq[cut:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    if not return_confidence:
        return text

    # 计算生成 token 的平均概率，作为置信度
    scores = getattr(out, "scores", None)
    if scores is None or len(scores) == 0 or len(gen_ids) == 0:
        confidence = 0.0
        return text, confidence

    token_logprobs = []
    for step_idx, step_scores in enumerate(scores):
        # step_scores: [batch_size, vocab_size]
        log_probs = step_scores.log_softmax(dim=-1)
        token_id = gen_ids[step_idx]
        token_logprobs.append(log_probs[0, token_id])

    if token_logprobs:
        mean_logprob = torch.stack(token_logprobs).mean().item()
        confidence = float(torch.exp(torch.tensor(mean_logprob)).item())
    else:
        confidence = 0.0

    return text, confidence


# -----------------------------
# Prompts
# -----------------------------

def agent_response(
    query: str,
    document: str,
    model,
    tokenizer,
    generation_config,
    history: str = "",
    agg_summary: str = "",
    injected: bool = False,
    return_confidence: bool = False,
):
    """
    injected=True 表示本轮 agent 在前向前已经通过 LoRA 注入收到了上一轮的信息
   （可能来自 aggregator summary 或 agents history）。
    return_confidence=True 时，返回 (response_text, confidence)。
    """
    inject_hint = ""
    if injected:
        inject_hint = (
            "\nNote: Some information from previous rounds has been integrated into your "
            "internal state for THIS question via an internal compressed parameter channel. "
            "You should use this internal signal together with the document and any visible "
            "context when answering, but do NOT assume specific details that are not supported "
            "by either the document, the visible context, or the question itself.\n"
        )

    if agg_summary and history:
        prompt = f"""You are an agent reading a document to answer a question.

Question: {query}
Document: {document}

Aggregator summary from the previous round:
{agg_summary}

Additionally, here are other agents' previous responses as context:
{history}
{inject_hint}
Please reconsider your answer accordingly. Provide your answer and a step-by-step reasoning explanation.
Please follow the format: 'Answer: {{}}. Explanation: {{}}.'"""
    elif agg_summary:
        prompt = f"""You are an agent reading a document to answer a question.

Question: {query}
Document: {document}

The following is the aggregator-generated summary from the previous round:
{agg_summary}
{inject_hint}
Please reconsider your answer accordingly. Provide your answer and a step-by-step reasoning explanation.
Please follow the format: 'Answer: {{}}. Explanation: {{}}.'"""
    elif history:
        prompt = f"""You are an agent reading a document to answer a question.

Question: {query}
Document: {document}

The following responses are from other agents as additional information.
{history}
{inject_hint}
Answer the question based on the document and other agents' response. Provide your answer and a step-by-step reasoning explanation.
Please follow the format: 'Answer: {{}}. Explanation: {{}}.'"""
    else:
        prompt = f"""You are an agent reading a document to answer a question.

Question: {query}
Document: {document}
{inject_hint}
Answer the question based only on this document (and any internal signal that may have been integrated into your parameters for this question). Provide your answer and a step-by-step reasoning explanation.
Please follow the format: 'Answer: {{}}. Explanation: {{}}.'"""

    messages = [{"role": "user", "content": prompt}]
    return call_llm_chat(messages, model, tokenizer, generation_config, return_confidence=return_confidence)


def aggregate_responses(
    query: str,
    responses: List[str],
    model,
    tokenizer,
    generation_config,
    joined_text: str | None = None,
    injected: bool = False,
):
    """
    joined_text=None  ：使用 responses 组装文本
    joined_text=""    ：不展示 agent 文本（仅靠注入的 LoRA 消息）
    joined_text="..." ：显式使用传入文本（combine 模式）

    injected=True     ：当前聚合器在 forward 前已经注入了 agents 的 LoRA 消息
    """
    if joined_text is None:
        # text-only，无额外注入提示
        joined = "\n".join([f"Agent {i+1}: {r}" for i, r in enumerate(responses)])
        body = f"Agent responses:\n{joined}\n"
    elif joined_text == "":
        # LoRA-only：不暴露原始文本，但告诉模型信息已经注入其内部状态
        if injected:
            body = (
                "The other agents' responses have been integrated into your internal state "
                "for THIS question via an internal compressed parameter channel. "
                "You cannot see their raw texts, but you should treat this internal signal as "
                "additional, question-specific knowledge distilled from their answers. "
                "Do NOT assume any specific content that is not supported by this internal signal "
                "or by the question itself. If the internal signal is insufficient, reply exactly "
                "'unknown'.\n"
            )
        else:
            # 理论上不会出现（joined_text=="" 且未注入），但做个兜底
            body = "Agent responses are not explicitly visible in text for this question.\n"
    else:
        # combine 模式：既有文本，又有可能的注入
        if injected:
            body = (
                f"Agent responses:\n{joined_text}\n\n"
                "In addition, a compressed representation of these responses has been integrated "
                "into your internal state via an internal parameter channel. "
                "Use both the explicit texts and this internal signal when deciding which answers "
                "are correct, but do NOT hallucinate details not supported by either.\n"
            )
        else:
            body = f"Agent responses:\n{joined_text}\n"

    prompt = f"""You are an aggregator reading answers from multiple agents.

If there are multiple answers, please provide all possible correct answers and also provide a step-by-step reasoning explanation. If there is no correct answer, please reply 'unknown'.
Please follow the format: 'All Correct Answers: []. Explanation: {{}}.'

The following are examples:
Question: In which year was Michael Jordan born?
Agent responses:
Agent 1: Answer: 1963. Explanation: The document clearly states that Michael Jeffrey Jordan was born on February 17, 1963. 
Agent 2: Answer: 1956. Explanation: The document states that Michael Irwin Jordan was born on February 25, 1956. However, it's important to note that this document seems to be about a different Michael Jordan, who is an American scientist, not the basketball player. The other agents' responses do not align with the information provided in the document.
Agent 3: Answer: 1998. Explanation: The According to the document provided, Michael Jeffrey Jordan was born on February 17, 1998.
Agent 4: Answer: Unknown. Explanation: The provided document focuses on Jordan's college and early professional career, mentioning his college championship in 1982 and his entry into the NBA in 1984, but it does not include information about his birth year.
All Correct Answers: ["1963", "1956"]. Explanation: Agent 1 is talking about the basketball player Michael Jeffrey Jordan, who was born on Februray 17, 1963, so 1963 is correct. Agent 2 is talking about another person named Michael Jordan, who is an American scientist, and he was born in 1956. Therefore, the answer 1956 from Agent 2 is also correct. Agent 3 provides an error stating Michael Jordan's birth year as 1998, which is incorrect. Based on the correct information from Agent 1, Michael Jeffrey Jordan was born on February 17, 1963. Agent 4 does not provide any useful information.

Question: {query}
{body}
"""
    messages = [{"role": "user", "content": prompt}]
    return call_llm_chat(messages, model, tokenizer, generation_config)


# -----------------------------
# Param-knowledge agent: synthesize a pseudo-document from model parameters
# -----------------------------

def build_param_knowledge_document(
    question: str,
    model,
    tokenizer,
    generation_config,
    max_new_tokens: int = 512,
) -> str:
    """
    用模型自身参数（无外部文档）生成一篇“知识文档”，作为一个额外 agent 的输入。
    Prompt:

    Generate a document that provides accurate and relevant information to answer the given question.
    If the information is unclear or uncertain, explicitly state 'I don't know' to avoid any hallucinations.
    Question: {question}
    Document:
    """
    prompt = (
        "Generate a document that provides accurate and relevant information to answer the given question. "
        "If the information is unclear or uncertain, explicitly state 'I don't know' to avoid any hallucinations.\n\n"
        f"Question: {question}\n"
        "Document:"
    )
    messages = [{"role": "user", "content": prompt}]
    return call_llm_chat(messages, model, tokenizer, generation_config, max_new_tokens=max_new_tokens)


# -----------------------------
# DyPRAG projector & delta messages
# -----------------------------

def build_projector(model, projector_path: str, inference_epoch: int, lora_rank: int, projector_p: int) -> ParameterTranslator:
    modules = ["down_proj", "up_proj", "gate_proj"]
    layers = list(range(model.config.num_hidden_layers))
    dev = next(model.parameters()).device
    projector = ParameterTranslator(
        modules, layers,
        model.config.hidden_size,
        model.config.intermediate_size,
        lora_rank,
        projector_p
    ).to(dev)

    ckpt = os.path.join(projector_path, f"epoch_{max(inference_epoch - 1, 0)}.pt")
    try:
        state = torch.load(ckpt, map_location=dev, weights_only=True)  # PyTorch>=2.4
    except TypeError:
        state = torch.load(ckpt, map_location=dev)
    projector.load_state_dict(state.get("model_state_dict", state))
    projector.eval()
    return projector


def text_to_delta(text: str, model, tokenizer, projector: ParameterTranslator, max_len: int = 1500):
    tokens = tokenizer(
        text, padding=True, truncation=True, return_tensors="pt", max_length=max_len
    )
    dev = next(model.parameters()).device
    tokens = {k: v.to(dev) for k, v in tokens.items()}
    with torch.no_grad():
        out = model(tokens["input_ids"], attention_mask=tokens.get("attention_mask", None), output_hidden_states=True)
        input_embed = out.hidden_states[-1][:, -1, :]  # [1, hidden]
        deltas = projector(input_embed)
    return deltas


def messages_to_avg_delta(
    msg_list,
    model,
    tokenizer,
    projector,
    max_len: int = 1500,
    scales: List[float] | None = None,
):
    """
    从若干条文本消息得到一个平均 LoRA delta。
    若 scales 不为 None，则长度应与 msg_list 相同，表示对每个消息的 delta 进行缩放
    （例如对低置信度 agent 乘以 <1 的缩放因子）。
    """
    dev = next(model.parameters()).device
    all_deltas = []

    if scales is not None and len(scales) != len(msg_list):
        raise ValueError(f"len(scales)={len(scales)} must equal len(msg_list)={len(msg_list)}")

    for idx, msg in enumerate(msg_list):
        tokens = tokenizer(msg, padding=True, truncation=True, return_tensors="pt", max_length=max_len)
        tokens = {k: v.to(dev) for k, v in tokens.items()}
        with torch.no_grad():
            out = model(tokens["input_ids"], attention_mask=tokens.get("attention_mask", None), output_hidden_states=True)
            input_embeds = out.hidden_states[-1][:, -1, :]
            outputs = projector(input_embeds)  # dict[str, Tensor]

        if scales is not None:
            s = float(scales[idx])
            if s != 1.0:
                for k in outputs.keys():
                    outputs[k] = outputs[k] * s

        all_deltas.append(outputs)

    common_keys = set(all_deltas[0].keys())
    for d in all_deltas[1:]:
        common_keys &= set(d.keys())
    merged = {k: torch.stack([d[k] for d in all_deltas], dim=0).mean(dim=0) for k in common_keys}
    return merged


# -----------------------------
# Low-confidence suppression helper
# -----------------------------

def _compute_low_conf_scales(
    confidences: List[float],
    low_conf_ratio: float,
    low_conf_scale: float,
) -> List[float]:
    """
    根据每个 agent 的置信度，返回对应的缩放系数列表：
    - 置信度排序后，处于后 low_conf_ratio 比例的 agent 其 scale=low_conf_scale；
    - 其它 agent 的 scale=1.0。
    """
    n = len(confidences)
    if n == 0:
        return []

    # 规范化超参
    r = max(0.0, min(float(low_conf_ratio), 1.0))
    s = float(low_conf_scale)

    # 如果比例为 0 或缩放因子 >=1，等价于不抑制
    if r <= 0.0 or s >= 1.0:
        return [1.0] * n

    k = int(n * r)
    if k <= 0:
        k = 1  # 至少抑制 1 个

    # 从小到大排序，取最靠后的 low_conf_ratio 部分为低置信度
    indices = list(range(n))
    indices.sort(key=lambda i: confidences[i])  # 置信度从低到高
    low_idx_set = set(indices[:k])

    scales = []
    for i in range(n):
        if i in low_idx_set:
            scales.append(s)
        else:
            scales.append(1.0)
    return scales


# -----------------------------
# Multi-agent debate
# -----------------------------

def multi_agent_debate(
    query: str,
    documents: List[str],
    model,
    tokenizer,
    generation_config,
    num_rounds: int = 3,
    message_transport: str = "dyprag",
    projector: ParameterTranslator = None,
    agent_context: str = "aggregator",
    debug: bool = False,
    use_param_agent: bool = False,   # 是否启用参数知识代理
    low_conf_ratio: float = 0.5,     # 低置信度比例（仅 history+DyPRAG 模式下使用）
    low_conf_scale: float = 0.5,     # 低置信度 agent 的 LoRA delta 缩放系数
):
    """
    agent_context: 'aggregator' | 'history' | 'both' | 'none'
    message_transport:
        'text' | 'dyprag' | 'dyprag-combine'
        'full-dyprag' | 'full-dyprag-combine'

    use_param_agent:
        若为 True，则先让模型基于自身参数生成一篇“知识文档”，
        作为额外一个 document，加到 documents 末尾，对应一个额外 agent，
        在所有轮次中与其他 agents 一起参与。

    history + DyPRAG 模式下：
        - 每轮 agent 生成时，基于 token 概率计算该 agent 输出的置信度；
        - 在下一轮 aggregator->agents 的 history 注入阶段，按置信度排序，
          对于置信度处于后 low_conf_ratio 比例的 agent，将其 LoRA delta
          按 low_conf_scale 进行缩放。
    """
    records = {}
    agent_outputs: List[str] = []
    agent_confidences: List[float] = []  # 与 agent_outputs 对齐
    round_times = {}
    comm_debug = {"rounds": []}

    # --- 构造本地 documents 列表，并可选地追加 param-agent 文档 ---
    documents = list(documents)  # 防止原列表被外部复用
    param_doc = None
    param_agent_index = None
    if use_param_agent:
        param_doc = build_param_knowledge_document(
            question=query,
            model=model,
            tokenizer=tokenizer,
            generation_config=generation_config,
            max_new_tokens=512,
        )
        param_agent_index = len(documents)
        documents.append(param_doc)
        records["param_agent"] = {
            "enabled": True,
            "index": param_agent_index,
            "document": param_doc,
        }
    else:
        records["param_agent"] = {
            "enabled": False,
            "index": None,
            "document": None,
        }

    is_dyprag_like = message_transport in ("dyprag", "dyprag-combine", "full-dyprag", "full-dyprag-combine")
    is_combine = message_transport in ("dyprag-combine", "full-dyprag-combine")
    is_full = message_transport in ("full-dyprag", "full-dyprag-combine")

    # 仅在 history + DyPRAG 模式下计算置信度并做抑制
    need_conf_scores = is_dyprag_like and (agent_context == "history")

    # ---------------- Round 1: agents produce answers ----------------
    t0 = time.perf_counter()
    records["round1"] = {"answers": [], "explanations": []}

    for doc in documents:
        if need_conf_scores:
            response, conf = agent_response(
                query, doc, model, tokenizer, generation_config,
                injected=False,
                return_confidence=True,
            )
            agent_confidences.append(float(conf))
        else:
            response = agent_response(
                query, doc, model, tokenizer, generation_config,
                injected=False,
                return_confidence=False,
            )
        answer = response[response.find("Answer: ") + len("Answer: "):response.find("Explanation")].strip()
        explanation = response[response.find("Explanation: ") + len("Explanation: "):]
        records["round1"]["answers"].append(answer)
        records["round1"]["explanations"].append(explanation)
        agent_outputs.append(response)

    if need_conf_scores:
        records["round1"]["confidences"] = agent_confidences

    # ---------- agents -> aggregator (first aggregation) ----------
    if is_full:
        if projector is None:
            raise ValueError("projector is required for full-dyprag/full-dyprag-combine")

        _log(debug, f"[agents->aggregator] start round=round1 transport={message_transport} combine_text={is_combine} n_msgs={len(agent_outputs)}")

        joined_text = "\n".join([f"Agent {i+1}: {r}" for i, r in enumerate(agent_outputs)]) if is_combine else ""
        comm_item = {
            "round": "round1",
            "direction": "agents->aggregator",
            "transport": message_transport,
            "combine_text": bool(is_combine),
            "n_msgs": len(agent_outputs),
            "inject": False,
            "inject_error": None,
            "joined_text_len": len(joined_text),
        }
        try:
            # 这里仍然使用均值聚合，不做置信度加权（如需可进一步扩展）
            aggr_deltas = messages_to_avg_delta(agent_outputs, model, tokenizer, projector)
            delta_inject(model, aggr_deltas)
            comm_item["inject"] = True
            _log(debug, f"[agents->aggregator] injected=True")
        except Exception as e:
            comm_item["inject_error"] = str(e)
            _log(debug, f"[agents->aggregator] injected=False error={e}")

        try:
            records["round1"]["aggregation"] = aggregate_responses(
                query, agent_outputs, model, tokenizer, generation_config,
                joined_text=joined_text,
                injected=comm_item["inject"],
            )
        finally:
            if comm_item["inject"]:
                try:
                    delta_remove(model, aggr_deltas)
                except Exception as e:
                    _log(debug, f"[agents->aggregator] delta removal failed: {e}")
                del aggr_deltas
                torch.cuda.empty_cache()
                gc.collect()

        comm_debug["rounds"].append(comm_item)
    else:
        records["round1"]["aggregation"] = aggregate_responses(
            query, agent_outputs, model, tokenizer, generation_config,
            joined_text=None,
            injected=False,
        )

    round_times["round1"] = float(f"{(time.perf_counter() - t0):.6f}")
    prev_agg = records["round1"]["aggregation"]

    # ---------------- Additional rounds ----------------
    final_aggregation = None
    for t in range(1, num_rounds):
        t_start = time.perf_counter()
        round_key = f"round{t+1}"
        prev_round_key = f"round{t}"
        records[round_key] = {"answers": [], "explanations": []}
        new_outputs: List[str] = []
        new_confidences: List[float] = []

        injected = False
        deltas = None

        use_summary = agent_context in ("aggregator", "both")
        use_history = agent_context in ("history", "both")

        # ----- aggregator -> agents injection -----
        agg2agent_mode = None  # 'summary' | 'history' | None
        try:
            if is_dyprag_like and use_summary:
                if projector is None:
                    raise ValueError("projector is None while message_transport requires DyPRAG")
                deltas = text_to_delta(prev_agg, model, tokenizer, projector)
                delta_inject(model, deltas)
                injected = True
                agg2agent_mode = "summary"
            elif is_dyprag_like and agent_context == "history":
                if projector is None:
                    raise ValueError("projector is None while message_transport requires DyPRAG")

                # 基于上一轮 agent_confidences 计算缩放系数
                scales = None
                if need_conf_scores and agent_confidences:
                    scales = _compute_low_conf_scales(agent_confidences, low_conf_ratio, low_conf_scale)
                    records.setdefault(prev_round_key, {})
                    records[prev_round_key]["low_conf_scales"] = scales

                deltas = messages_to_avg_delta(
                    agent_outputs,
                    model,
                    tokenizer,
                    projector,
                    scales=scales,
                )
                delta_inject(model, deltas)
                injected = True
                agg2agent_mode = "history"
        except Exception as e:
            _log(debug, f"[aggregator->agents] inject FAILED mode={agg2agent_mode} err={e}")
            deltas = None
            injected = False
        else:
            _log(debug, f"[aggregator->agents] inject={bool(injected)} mode={agg2agent_mode}")

        # ----- agents generate this round -----
        num_agents = len(documents)
        agg_text_count = 0
        hist_text_count = 0

        for i, doc in enumerate(documents):
            history_str = ""
            if use_history:
                prev_agent_outputs = agent_outputs
                pieces = [f"Agent {j+1}: {prev_agent_outputs[j]}" for j in range(num_agents) if j != i]
                history_str = "\n".join(pieces)

            agg_text_for_prompt = ""
            hist_text_for_prompt = ""

            if agent_context == "aggregator":
                if message_transport == "text":
                    agg_text_for_prompt = prev_agg
                elif message_transport in ("dyprag", "full-dyprag"):
                    agg_text_for_prompt = ""           # 注入但不传文本
                else:  # dyprag-combine / full-dyprag-combine
                    agg_text_for_prompt = prev_agg     # 注入 + 文本
            elif agent_context == "history":
                if message_transport == "text":
                    hist_text_for_prompt = history_str
                elif message_transport in ("dyprag", "full-dyprag"):
                    hist_text_for_prompt = ""          # 注入平均 delta，不传文本
                else:
                    hist_text_for_prompt = history_str # 注入 + 文本
            elif agent_context == "both":
                if message_transport == "text":
                    agg_text_for_prompt = prev_agg
                    hist_text_for_prompt = history_str
                elif message_transport in ("dyprag", "full-dyprag"):
                    agg_text_for_prompt = ""           # 摘要已注入
                    hist_text_for_prompt = history_str # 历史走文本
                else:
                    agg_text_for_prompt = prev_agg     # 注入 + 文本
                    hist_text_for_prompt = history_str

            # 计数：本轮是否把文本传给了 agent
            if agg_text_for_prompt:
                agg_text_count += 1
            if hist_text_for_prompt:
                hist_text_count += 1

            if need_conf_scores:
                resp, conf = agent_response(
                    query,
                    doc,
                    model,
                    tokenizer,
                    generation_config,
                    agg_summary=agg_text_for_prompt,
                    history=hist_text_for_prompt,
                    injected=injected,
                    return_confidence=True,
                )
                new_confidences.append(float(conf))
            else:
                resp = agent_response(
                    query,
                    doc,
                    model,
                    tokenizer,
                    generation_config,
                    agg_summary=agg_text_for_prompt,
                    history=hist_text_for_prompt,
                    injected=injected,
                    return_confidence=False,
                )

            answer = resp[resp.find("Answer: ") + len("Answer: "):resp.find("Explanation")].strip()
            explanation = resp[resp.find("Explanation: ") + len("Explanation: "):]
            records[round_key]["answers"].append(answer)
            records[round_key]["explanations"].append(explanation)
            new_outputs.append(resp)

        # remove deltas (aggregator -> agents)
        if injected:
            try:
                delta_remove(model, deltas)
                del deltas
                torch.cuda.empty_cache()
                gc.collect()
            except Exception as e:
                _log(debug, f"[aggregator->agents] delta removal failed: {e}")

        # 记录本轮 aggregator->agents 的提示文本情况
        comm_debug["rounds"].append({
            "round": round_key,
            "direction": "aggregator->agents",
            "mode": agg2agent_mode,               # None/summary/history
            "transport": message_transport,
            "agents_n": num_agents,
            "text_in_prompt": {
                "agg_summary_count": agg_text_count,
                "history_count": hist_text_count
            }
        })

        agent_outputs = new_outputs
        if need_conf_scores:
            agent_confidences = new_confidences
            records[round_key]["confidences"] = new_confidences

        # ----- convergence check -----
        pred_ans_list = [normalize_answer(a) for a in records[round_key]["answers"]]
        prev_pred_ans_list = [normalize_answer(a) for a in records[f"round{t}"]["answers"]]
        assert len(pred_ans_list) == len(prev_pred_ans_list)
        flag = True
        for k in range(len(pred_ans_list)):
            a, b = pred_ans_list[k], prev_pred_ans_list[k]
            if (a in b) or (b in a):
                continue
            flag = False
            break

        if flag:
            final_aggregation = prev_agg
            round_times[round_key] = float(f"{(time.perf_counter() - t_start):.6f}")
            _log(debug, f"[round={round_key}] converged | time={round_times[round_key]:.2f}s")
            break
        else:
            # ----- agents -> aggregator (aggregation of this round) -----
            if is_full:
                if projector is None:
                    raise ValueError("projector is required for full-dyprag/full-dyprag-combine")

                _log(debug, f"[agents->aggregator] start round={round_key} transport={message_transport} combine_text={is_combine} n_msgs={len(agent_outputs)}")

                joined_text = "\n".join([f"Agent {i+1}: {r}" for i, r in enumerate(agent_outputs)]) if is_combine else ""
                comm_item = {
                    "round": round_key,
                    "direction": "agents->aggregator",
                    "transport": message_transport,
                    "combine_text": bool(is_combine),
                    "n_msgs": len(agent_outputs),
                    "inject": False,
                    "inject_error": None,
                    "joined_text_len": len(joined_text),
                }
                try:
                    # 这里仍然不使用置信度加权（如需可另行扩展）
                    aggr_deltas = messages_to_avg_delta(agent_outputs, model, tokenizer, projector)
                    delta_inject(model, aggr_deltas)
                    comm_item["inject"] = True
                    _log(debug, f"[agents->aggregator] injected=True")
                except Exception as e:
                    comm_item["inject_error"] = str(e)
                    _log(debug, f"[agents->aggregator] injected=False error={e}")

                try:
                    records[round_key]["aggregation"] = aggregate_responses(
                        query, agent_outputs, model, tokenizer, generation_config,
                        joined_text=joined_text,
                        injected=comm_item["inject"],
                    )
                finally:
                    if comm_item["inject"]:
                        try:
                            delta_remove(model, aggr_deltas)
                        except Exception as e:
                            _log(debug, f"[agents->aggregator] delta removal failed: {e}")
                        del aggr_deltas
                        torch.cuda.empty_cache()
                        gc.collect()

                comm_debug["rounds"].append(comm_item)
            else:
                records[round_key]["aggregation"] = aggregate_responses(
                    query, agent_outputs, model, tokenizer, generation_config,
                    joined_text=None,
                    injected=False,
                )

            final_aggregation = records[round_key]["aggregation"]
            prev_agg = records[round_key]["aggregation"]
            round_times[round_key] = float(f"{(time.perf_counter() - t_start):.6f}")
            _log(debug, f"[round={round_key}] continue | time={round_times[round_key]:.2f}s")

    records["final_aggregation"] = final_aggregation
    records["final_answers"] = parse_aggregator_answers(final_aggregation or "")
    records["round_time_seconds"] = round_times
    records["comm_debug"] = comm_debug
    return records


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="Model name or local path (compatible with utils.get_model).")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--num_rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)

    # message transport / DyPRAG projector
    parser.add_argument("--message_transport", type=str, default="dyprag",
                        choices=["text", "dyprag", "dyprag-combine", "full-dyprag", "full-dyprag-combine"])
    parser.add_argument("--projector_path", type=str, help="Dir containing projector ckpts named epoch_{k}.pt")
    parser.add_argument("--inference_epoch", type=int, default=1, help="Use epoch_{inference_epoch-1}.pt")
    parser.add_argument("--lora_rank", type=int, default=2)
    parser.add_argument("--projector_p", type=int, default=32)

    # agent context switch
    parser.add_argument("--agent_context", type=str, default="aggregator",
                        choices=["aggregator", "history", "both", "none"],
                        help="What to pass to agents from previous round(s).")

    # generation
    parser.add_argument("--max_new_tokens", type=int, default=128)

    # debug logs
    parser.add_argument("--debug", action="store_true", help="print detailed comm-path logs")

    # 是否启用“参数知识代理”
    parser.add_argument("--use_param_agent", action="store_true",
                        help="Add an extra agent that uses the model's own parametric knowledge (no external documents).")

    # 低置信度抑制相关超参（仅 history + DyPRAG 模式下生效）
    parser.add_argument("--low_conf_ratio", type=float, default=0.5,
                        help="Fraction (0-1) of agents per round treated as low-confidence (only when agent_context='history' and DyPRAG-like inject).")
    parser.add_argument("--low_conf_scale", type=float, default=0.5,
                        help="Scale factor (0-1) applied to LoRA delta of low-confidence agents when computing history-based injection.")

    args = parser.parse_args()
    set_seed(args.seed)

    # ---- unique output path (prefix per run) ----
    run_prefix = _make_run_prefix()
    out_tag = f"msg-{args.message_transport}_ctx-{args.agent_context}"
    if args.use_param_agent:
        out_tag += "_withParamAgent"
    if args.agent_context == "history" and args.message_transport in ("dyprag", "dyprag-combine", "full-dyprag", "full-dyprag-combine"):
        out_tag += f"_LC{args.low_conf_ratio:.2f}_SC{args.low_conf_scale:.2f}"
    model_tag = args.model_name.split("/")[-1]
    base_name = os.path.basename(args.data_path)
    file_name = f"{run_prefix}_{base_name}_{out_tag}_{model_tag}_rounds{args.num_rounds}.jsonl"
    output_path = os.path.join("cache", file_name)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # dataset
    with open(args.data_path, "r", encoding="utf-8") as f:
        all_data = [json.loads(line) for line in f]

    # load model
    model, tokenizer, generation_config = get_model(
        args.model_name,
        max_new_tokens=args.max_new_tokens,
    )
    model = _peft_wrap(model, args)
    if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token

    projector = None
    if args.message_transport in ("dyprag", "dyprag-combine", "full-dyprag", "full-dyprag-combine"):
        if not args.projector_path:
            raise ValueError("--projector_path is required when message_transport uses DyPRAG")
        projector = build_projector(model, args.projector_path, args.inference_epoch, args.lora_rank, args.projector_p)

    gpu_name = _detect_gpu_name()

    # run end-to-end (no resume) and overwrite output
    total_em, counted = 0, 0
    with open(output_path, "w", encoding="utf-8") as out:
        for idx in tqdm(range(len(all_data)), desc="MADAM-RAG (DyPRAG)"):
            ex = all_data[idx]
            query = ex["question"]
            documents = ex["documents"]

            rec = multi_agent_debate(
                query, documents,
                model, tokenizer, generation_config,
                num_rounds=args.num_rounds,
                message_transport=args.message_transport,
                projector=projector,
                agent_context=args.agent_context,
                debug=args.debug,
                use_param_agent=args.use_param_agent,
                low_conf_ratio=args.low_conf_ratio,
                low_conf_scale=args.low_conf_scale,
            )

            pred = rec.get("final_answers") or parse_aggregator_answers(rec.get("final_aggregation", ""))
            gold = ex.get("gold_answers", [])
            em = em_score(pred, gold) if gold is not None else None
            if em is not None:
                total_em += em
                counted += 1

            rec["pred_answers"] = pred
            rec["gold_answers"] = gold
            rec["em"] = em
            rec["index"] = idx
            rec["gpu_name"] = gpu_name
            rec["run_prefix"] = run_prefix

            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()

            # 周期性内存清理，缓解显存碎片
            if (idx + 1) % 10 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass

    if counted:
        print(f"[EM] {total_em}/{counted} = {total_em / counted * 100:.2f}%")
    print(f"[Output] {output_path}")

    # cleanup
    try:
        del model
        del tokenizer
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        torch.cuda.empty_cache()


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
