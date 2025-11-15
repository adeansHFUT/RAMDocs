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


def call_llm_chat(messages, model, tokenizer, generation_config, max_new_tokens: int = 128) -> str:
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


def aggregate_responses(query: str, responses: List[str], model, tokenizer, generation_config, joined_text: str | None = None):
    """
    joined_text=None  ：使用 responses 组装文本
    joined_text=""    ：不展示 agent 文本（仅靠注入的 LoRA 消息）
    joined_text="..." ：显式使用传入文本（combine 模式）
    """
    if joined_text is None:
        joined = "\n".join([f"Agent {i+1}: {r}" for i, r in enumerate(responses)])
        body = f"Agent responses:\n{joined}\n"
    elif joined_text == "":
        body = "Agent responses are provided via an internal compressed channel.\n"
    else:
        body = f"Agent responses:\n{joined_text}\n"

    prompt = f"""You are an aggregator reading answers from multiple agents.

If there are multiple answers, please provide all possible correct answers and also provide a step-by-step reasoning explanation. If there is no correct answer, please reply 'unknown'.
Please follow the format: 'All Correct Answers: []. Explanation: {{}}.'

{body}
Question: {query}
"""
    messages = [{"role": "user", "content": prompt}]
    return call_llm_chat(messages, model, tokenizer, generation_config)


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


def messages_to_avg_delta(msg_list, model, tokenizer, projector, max_len: int = 1500):
    dev = next(model.parameters()).device
    all_deltas = []
    for msg in msg_list:
        tokens = tokenizer(msg, padding=True, truncation=True, return_tensors="pt", max_length=max_len)
        tokens = {k: v.to(dev) for k, v in tokens.items()}
        with torch.no_grad():
            out = model(tokens["input_ids"], attention_mask=tokens.get("attention_mask", None), output_hidden_states=True)
            input_embeds = out.hidden_states[-1][:, -1, :]
            outputs = projector(input_embeds)  # dict[str, Tensor]
            all_deltas.append(outputs)
    common_keys = set(all_deltas[0].keys())
    for d in all_deltas[1:]:
        common_keys &= set(d.keys())
    merged = {k: torch.stack([d[k] for d in all_deltas], dim=0).mean(dim=0) for k in common_keys}
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
    debug: bool = False,
):
    """
    agent_context: 'aggregator' | 'history' | 'both' | 'none'
    message_transport:
        'text' | 'dyprag' | 'dyprag-combine'
        'full-dyprag' | 'full-dyprag-combine'
    """
    records = {}
    agent_outputs: List[str] = []
    round_times = {}
    comm_debug = {"rounds": []}

    is_dyprag_like = message_transport in ("dyprag", "dyprag-combine", "full-dyprag", "full-dyprag-combine")
    is_combine = message_transport in ("dyprag-combine", "full-dyprag-combine")
    is_full = message_transport in ("full-dyprag", "full-dyprag-combine")

    # ---------------- Round 1: agents produce answers ----------------
    t0 = time.perf_counter()
    records["round1"] = {"answers": [], "explanations": []}
    for doc in documents:
        response = agent_response(query, doc, model, tokenizer, generation_config)
        answer = response[response.find("Answer: ") + len("Answer: "):response.find("Explanation")].strip()
        explanation = response[response.find("Explanation: ") + len("Explanation: "):]
        records["round1"]["answers"].append(answer)
        records["round1"]["explanations"].append(explanation)
        agent_outputs.append(response)

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
                joined_text=joined_text
            )
        finally:
            if comm_item["inject"]:
                try:
                    delta_remove(model, aggr_deltas)
                except Exception as e:
                    _log(debug, f"[agents->aggregator] delta removal failed: {e}")
                del aggr_deltas
                torch.cuda.empty_cache(); gc.collect()

        comm_debug["rounds"].append(comm_item)
    else:
        records["round1"]["aggregation"] = aggregate_responses(
            query, agent_outputs, model, tokenizer, generation_config
        )

    round_times["round1"] = float(f"{(time.perf_counter() - t0):.6f}")
    prev_agg = records["round1"]["aggregation"]

    # ---------------- Additional rounds ----------------
    final_aggregation = None
    for t in range(1, num_rounds):
        t_start = time.perf_counter()
        round_key = f"round{t+1}"
        records[round_key] = {"answers": [], "explanations": []}
        new_outputs = []

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
                deltas = messages_to_avg_delta(agent_outputs, model, tokenizer, projector)
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

            resp = agent_response(
                query, doc, model, tokenizer, generation_config,
                agg_summary=agg_text_for_prompt, history=hist_text_for_prompt
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
                        joined_text=joined_text
                    )
                finally:
                    if comm_item["inject"]:
                        try:
                            delta_remove(model, aggr_deltas)
                        except Exception as e:
                            _log(debug, f"[agents->aggregator] delta removal failed: {e}")
                        del aggr_deltas
                        torch.cuda.empty_cache(); gc.collect()

                comm_debug["rounds"].append(comm_item)
            else:
                records[round_key]["aggregation"] = aggregate_responses(
                    query, agent_outputs, model, tokenizer, generation_config
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

    args = parser.parse_args()
    set_seed(args.seed)

    # ---- unique output path (prefix per run) ----
    run_prefix = _make_run_prefix()
    out_tag = f"msg-{args.message_transport}_ctx-{args.agent_context}"
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
