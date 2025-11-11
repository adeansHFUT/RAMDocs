# -*- coding: utf-8 -*-
"""
MADAM-RAG with DyPRAG-style LoRA-as-message (delta inject/remove) + resume support
+ agent context switch (aggregator / history / both / none)
+ message_transport option 'dyprag-combine'
+ per-round timing & GPU model logging.

Dependencies from DyPRAG project:
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


def _peft_wrap(model, args):
    """
    Wrap model as PeftModel to match projector's trained hierarchy.
    This uses a lightweight "inference-only" config and does NOT load any adapter.
    """
    peft_cfg = LoraConfig(
        task_type="CAUSAL_LM",
        # 作为外壳，target_modules 用 projector 训练的 MLP 三件套或置空都可。
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
    """
    优先解析 JSON：{"answers":[...], "explanation":"..."}
    回退解析：All Correct Answers: [...]. Explanation: ...
    """
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
    """集合级 EM：标准化后两个集合完全相等记 1，否则 0。"""
    P = {normalize_answer(x) for x in (pred_answers or [])}
    G = {normalize_answer(x) for x in (gold_answers or [])}
    return int(P == G)


# -----------------------------
# LLM calling helpers
# -----------------------------

def _to_gen_cfg_dict(generation_config):
    """把 transformers.GenerationConfig 或 dict 统一转成 dict。"""
    if generation_config is None:
        return {}
    if isinstance(generation_config, dict):
        return dict(generation_config)
    if hasattr(generation_config, "to_dict"):
        return generation_config.to_dict()
    return {}


def call_llm_chat(messages, model, tokenizer, generation_config, max_new_tokens: int = 128) -> str:
    """
    用 chat template 渲染，再用 model.generate() 生成。
    """
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = tokenizer(prompt, return_tensors="pt")
    dev = next(model.parameters()).device
    inputs = {k: v.to(dev) for k, v in inputs.items()}

    gen_cfg = _to_gen_cfg_dict(generation_config)
    # 置空 sample 相关项，消除 warning
    gen_cfg.update(dict(
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
        num_beams=1,
        max_new_tokens=max_new_tokens,
        return_dict_in_generate=True,
        pad_token_id=getattr(tokenizer, "pad_token_id", getattr(tokenizer, "eos_token_id", None)),
        eos_token_id=getattr(tokenizer, "eos_token_id", None),
    ))

    with torch.no_grad():
        out = model.generate(**inputs, **gen_cfg)

    seq = out.sequences[0]
    cut = inputs["input_ids"].shape[1]
    gen_ids = seq[cut:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


# -----------------------------
# Prompts
# -----------------------------

def agent_response(query: str, document: str, model, tokenizer, generation_config,
                   history: str = "", agg_summary: str = ""):
    """
    Agent 提示：
    - 若提供 agg_summary：展示聚合器摘要
    - 若提供 history     ：展示上一轮其他 agents 的原话
    - 均无                ：只基于本 doc
    """
    if agg_summary and history:
        prompt = f"""You are an agent reading a document to answer a question.

Question: {query}
Document: {document}

Aggregator summary from the previous round:
{agg_summary}

Additionally, here are other agents' previous responses as context:
{history}

Please reconsider your answer accordingly. Provide your answer and a step-by-step reasoning explanation.
Please follow the format: 'Answer: {{}}. Explanation: {{}}.'"""
    elif agg_summary:
        prompt = f"""You are an agent reading a document to answer a question.

Question: {query}
Document: {document}

The following is the aggregator-generated summary from the previous round:
{agg_summary}

Please reconsider your answer accordingly. Provide your answer and a step-by-step reasoning explanation.
Please follow the format: 'Answer: {{}}. Explanation: {{}}.'"""
    elif history:
        prompt = f"""You are an agent reading a document to answer a question.

Question: {query}
Document: {document}

The following responses are from other agents as additional information.
{history}

Answer the question based on the document and other agents' response. Provide your answer and a step-by-step reasoning explanation.
Please follow the format: 'Answer: {{}}. Explanation: {{}}.'"""
    else:
        prompt = f"""You are an agent reading a document to answer a question.

Question: {query}
Document: {document}

Answer the question based only on this document. Provide your answer and a step-by-step reasoning explanation.
Please follow the format: 'Answer: {{}}. Explanation: {{}}.'"""

    messages = [{"role": "user", "content": prompt}]
    return call_llm_chat(messages, model, tokenizer, generation_config)


def aggregate_responses(query: str, responses: List[str], model, tokenizer, generation_config):
    """
    聚合器提示：列出所有可能正确答案，并给出解释。
    """
    joined = "\n".join([f"Agent {i+1}: {r}" for i, r in enumerate(responses)])
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
All Correct Answers: ["1963", "1956"]. Explanation: Agent 1 is talking about the basketball player Michael Jeffrey Jordan, who was born on February 17, 1963, so 1963 is correct. Agent 2 is talking about another person named Michael Jordan, who is an American scientist, and he was born in 1956. Therefore, the answer 1956 from Agent 2 is also correct. Agent 3 provides an error stating Michael Jordan's birth year as 1998, which is incorrect. Based on the correct information from Agent 1, Michael Jeffrey Jordan was born on February 17, 1963. Agent 4 does not provide any useful information.

Question: {query}
Agent responses:
{joined}
"""
    messages = [{"role": "user", "content": prompt}]
    return call_llm_chat(messages, model, tokenizer, generation_config)


# -----------------------------
# DyPRAG projector & delta messages
# -----------------------------

def build_projector(model, projector_path: str, inference_epoch: int, lora_rank: int, projector_p: int) -> ParameterTranslator:
    """
    构造 DyPRAG 的 ParameterTranslator 并加载 checkpoint。
    模块选择与 DyPRAG 默认一致：MLP 的 down/up/gate。
    """
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
        state = torch.load(ckpt, map_location=dev)  # 兼容旧版本
    projector.load_state_dict(state.get("model_state_dict", state))
    projector.eval()
    return projector


def text_to_delta(agg_text: str, model, tokenizer, projector: ParameterTranslator, max_len: int = 1500):
    """
    单段文本 -> delta：
      1) tokenizer 编码
      2) 前向拿最后一层的 hidden_states
      3) 取最后 token 的隐藏向量输入 projector
      4) projector 输出 {模块名: delta_tensor} 字典
    """
    tokens = tokenizer(
        agg_text,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=max_len
    )
    dev = next(model.parameters()).device
    tokens = {k: v.to(dev) for k, v in tokens.items()}

    with torch.no_grad():
        out = model(tokens["input_ids"], attention_mask=tokens.get("attention_mask", None), output_hidden_states=True)
        input_embed = out.hidden_states[-1][:, -1, :]  # [1, hidden_size]
        deltas = projector(input_embed)
    return deltas


def messages_to_avg_delta(msg_list, model, tokenizer, projector, max_len: int = 1500):
    """
    多段文本 -> 多个 delta -> 逐键平均（与 DyPRAG 示例保持同款：stack+mean）。
    """
    dev = next(model.parameters()).device
    all_deltas = []

    for msg in msg_list:
        tokens = tokenizer(
            msg,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=max_len
        )
        tokens = {k: v.to(dev) for k, v in tokens.items()}

        with torch.no_grad():
            out = model(tokens["input_ids"],
                        attention_mask=tokens.get("attention_mask", None),
                        output_hidden_states=True)
            # 取最后一层、最后一个 token 隐向量（与 DyPRAG 保持一致）
            input_embeds = out.hidden_states[-1][:, -1, :]      # [1, hidden_size]
            outputs = projector(input_embeds)                    # dict[str, Tensor]
            all_deltas.append(outputs)

    # 取所有 delta 字典的公共键，避免缺键时报错
    common_keys = set(all_deltas[0].keys())
    for d in all_deltas[1:]:
        common_keys &= set(d.keys())

    merged = {}
    for k in common_keys:
        merged[k] = torch.stack([d[k] for d in all_deltas], dim=0).mean(dim=0)

    return merged


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
):
    """
    agent_context: 'aggregator' | 'history' | 'both' | 'none'
    message_transport: 'text' | 'dyprag' | 'dyprag-combine'
    """
    records = {}
    agent_outputs: List[str] = []
    round_times = {}

    # Round 1
    t0 = time.perf_counter()
    records["round1"] = {"answers": [], "explanations": []}
    for doc in documents:
        response = agent_response(query, doc, model, tokenizer, generation_config)
        answer = response[response.find("Answer: ") + len("Answer: "):response.find("Explanation")].strip()
        explanation = response[response.find("Explanation: ") + len("Explanation: "):]
        records["round1"]["answers"].append(answer)
        records["round1"]["explanations"].append(explanation)
        agent_outputs.append(response)

    records["round1"]["aggregation"] = aggregate_responses(query, agent_outputs, model, tokenizer, generation_config)
    round_times["round1"] = float(f"{(time.perf_counter() - t0):.6f}")
    prev_agg = records["round1"]["aggregation"]

    # Additional rounds
    final_aggregation = None
    for t in range(1, num_rounds):
        t_start = time.perf_counter()
        round_key = f"round{t+1}"
        records[round_key] = {"answers": [], "explanations": []}
        new_outputs = []

        is_dyprag = message_transport in ("dyprag", "dyprag-combine")
        is_combine = (message_transport == "dyprag-combine")

        injected = False
        deltas = None

        use_summary = agent_context in ("aggregator", "both")
        use_history = agent_context in ("history", "both")

        # == 注入逻辑 ==
        try:
            if is_dyprag and use_summary:
                # 对摘要做 delta 注入（both 也走摘要注入）
                if projector is None:
                    raise ValueError("projector is None while message_transport is dyprag/dyprag-combine")
                deltas = text_to_delta(prev_agg, model, tokenizer, projector)
                delta_inject(model, deltas)
                injected = True
            elif is_dyprag and (agent_context == "history"):
                # history-only：把上一轮所有 agents 原话做平均 delta 注入
                if projector is None:
                    raise ValueError("projector is None while message_transport is dyprag/dyprag-combine")
                prev_agent_outputs = agent_outputs
                deltas = messages_to_avg_delta(prev_agent_outputs, model, tokenizer, projector)
                delta_inject(model, deltas)
                injected = True
        except Exception as e:
            print(f"[WARN] DyPRAG delta inject failed; fallback to text if applicable. Error: {e}")
            deltas = None
            injected = False

        # == 这一轮的各 agent 生成 ==
        num_agents = len(documents)
        for i, doc in enumerate(documents):
            # 历史文本（上一轮所有 agents 原话，排除自己）
            history_str = ""
            if use_history:
                prev_agent_outputs = agent_outputs
                pieces = [f"Agent {j+1}: {prev_agent_outputs[j]}" for j in range(num_agents) if j != i]
                history_str = "\n".join(pieces)

            # 选择传入的上下文文本
            # text 模式：照常传文本
            # dyprag 模式：默认不传（因为信息在权重里）
            # dyprag-combine：同时传文本
            agg_text_for_prompt = ""
            hist_text_for_prompt = ""

            if agent_context == "aggregator":
                if message_transport == "text":
                    agg_text_for_prompt = prev_agg
                elif message_transport == "dyprag":
                    agg_text_for_prompt = ""  # 注入但不传文本
                else:  # dyprag-combine
                    agg_text_for_prompt = prev_agg  # 注入+传文本

            elif agent_context == "history":
                if message_transport == "text":
                    hist_text_for_prompt = history_str
                elif message_transport == "dyprag":
                    hist_text_for_prompt = ""  # 注入平均 delta，不传文本
                else:  # dyprag-combine
                    hist_text_for_prompt = history_str  # 注入+传文本

            elif agent_context == "both":
                if message_transport == "text":
                    agg_text_for_prompt = prev_agg
                    hist_text_for_prompt = history_str
                elif message_transport == "dyprag":
                    # 只对摘要注入；历史以文本形式给（与之前一致）
                    agg_text_for_prompt = ""  # 已注入
                    hist_text_for_prompt = history_str
                else:  # dyprag-combine
                    # 注入摘要，并同时把摘要文本 + 历史文本都给
                    agg_text_for_prompt = prev_agg
                    hist_text_for_prompt = history_str

            # 生成
            resp = agent_response(
                query, doc, model, tokenizer, generation_config,
                agg_summary=agg_text_for_prompt, history=hist_text_for_prompt
            )

            answer = resp[resp.find("Answer: ") + len("Answer: "):resp.find("Explanation")].strip()
            explanation = resp[resp.find("Explanation: ") + len("Explanation: "):]
            records[round_key]["answers"].append(answer)
            records[round_key]["explanations"].append(explanation)
            new_outputs.append(resp)

        # 移除 delta
        if injected:
            try:
                delta_remove(model, deltas)
                del deltas
                torch.cuda.empty_cache()
                gc.collect()
            except Exception as e:
                print(f"[WARN] DyPRAG delta removal failed: {e}")

        agent_outputs = new_outputs

        # 稳定性判断：包含关系视为稳定
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
            break
        else:
            records[round_key]["aggregation"] = aggregate_responses(
                query, agent_outputs, model, tokenizer, generation_config
            )
            final_aggregation = records[round_key]["aggregation"]
            prev_agg = records[round_key]["aggregation"]
            round_times[round_key] = float(f"{(time.perf_counter() - t_start):.6f}")

    records["final_aggregation"] = final_aggregation
    records["final_answers"] = parse_aggregator_answers(final_aggregation or "")
    records["round_time_seconds"] = round_times
    return records


# -----------------------------
# Resume helpers
# -----------------------------

def _read_existing_records(path: str):
    recs = []
    if not os.path.exists(path):
        return recs
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except Exception:
                continue
    return recs


def _compute_em_over(recs: list, dataset: list):
    total_em, counted = 0, 0
    N = min(len(recs), len(dataset))
    for i in range(N):
        rec = recs[i]
        ex = dataset[i]
        pred = rec.get("final_answers") or parse_aggregator_answers(rec.get("final_aggregation", ""))
        gold = ex.get("gold_answers", [])
        em = em_score(pred, gold) if gold is not None else None
        if em is not None:
            total_em += em
            counted += 1
    return total_em, counted


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
                        choices=["text", "dyprag", "dyprag-combine"])
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

    args = parser.parse_args()
    set_seed(args.seed)

    out_tag = f"msg-{args.message_transport}_ctx-{args.agent_context}"
    model_tag = args.model_name.split("/")[-1]
    output_path = f"{args.data_path}_madam_rag_{out_tag}_{model_tag}_rounds{args.num_rounds}.jsonl"
    output_path = os.path.join("cache", output_path)

    # dataset
    with open(args.data_path, "r", encoding="utf-8") as f:
        all_data = [json.loads(line) for line in f]
    expected_n = len(all_data)

    # resume read
    existing_recs = _read_existing_records(output_path)
    existing_n = len(existing_recs)

    if existing_n >= expected_n:
        total_em, counted = _compute_em_over(existing_recs, all_data)
        if counted:
            print(f"[EM] {total_em}/{counted} = {total_em / counted * 100:.2f}%")
        print(f"[Output] {output_path}")
        return

    # load model
    model, tokenizer, generation_config = get_model(
        args.model_name,
        max_new_tokens=args.max_new_tokens,
    )
    # wrap to PeftModel (path compatibility with projector training)
    model = _peft_wrap(model, args)
    if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token

    projector = None
    if args.message_transport in ("dyprag", "dyprag-combine"):
        if not args.projector_path:
            raise ValueError("--projector_path is required when message_transport is 'dyprag' or 'dyprag-combine'")
        projector = build_projector(model, args.projector_path, args.inference_epoch, args.lora_rank, args.projector_p)

    gpu_name = _detect_gpu_name()

    # resume append
    start_idx = existing_n
    with open(output_path, "a", encoding="utf-8") as out:
        for idx in tqdm(range(start_idx, expected_n), desc="MADAM-RAG (DyPRAG) resume"):
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
            )

            pred = rec.get("final_answers") or parse_aggregator_answers(rec.get("final_aggregation", ""))
            gold = ex.get("gold_answers", [])
            em = em_score(pred, gold) if gold is not None else None

            rec["pred_answers"] = pred
            rec["gold_answers"] = gold
            rec["em"] = em
            rec["index"] = idx
            rec["gpu_name"] = gpu_name  # 记录 GPU 型号

            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()

    # final EM
    final_recs = _read_existing_records(output_path)
    total_em, counted = _compute_em_over(final_recs, all_data)
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
