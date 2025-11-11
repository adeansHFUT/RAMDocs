import argparse
import os
import re
import json
import torch
import string
from tqdm import tqdm
from typing import List
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed

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
    尝试两种解析：
    1) 首选解析 JSON：{"answers":[...], "explanation":"..."}
    2) 回退解析老格式：All Correct Answers: [...]. Explanation: ...
    """
    # JSON 优先
    try:
        m = re.search(r'\{.*\}', agg_text, flags=re.S)
        if m:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict) and "answers" in obj:
                return [a.strip() for a in obj["answers"] if isinstance(a, str)]
    except Exception:
        pass

    # 回退：从方括号里抓答案
    m = re.search(r'All\s*Correct\s*Answers\s*:\s*\[(.*?)\]', agg_text, flags=re.I|re.S)
    if not m:
        return []
    inside = m.group(1)
    # 尝试把中间内容按 JSON 解析
    try:
        return [str(x).strip() for x in json.loads("[" + inside + "]")]
    except Exception:
        # 最后兜底：逗号切分
        return [p.strip(" '\"\n. ") for p in inside.split(",") if p.strip()]

def em_score(pred_answers, gold_answers):
    """集合级 EM：标准化后两个集合完全相等记 1，否则 0。"""
    P = {normalize_answer(x) for x in (pred_answers or [])}
    G = {normalize_answer(x) for x in (gold_answers or [])}
    return int(P == G)

def call_llm(prompt: str, generator, max_new_tokens: int = 128) -> str:
    messages = [{"role": "user", "content": prompt}]
    output = generator(
                messages,
                max_new_tokens=max_new_tokens,
                top_p=None,
                do_sample=False)
    return output[0]["generated_text"][-1]['content'].strip()


def agent_response(query: str, document: str, generator, history: str = "", agg_summary: str = ""):
    if agg_summary:
        prompt = f"""You are an agent reading a document to answer a question.

        Question: {query}
        Document: {document}

        The following is the aggregator-generated summary from the previous round:
        {agg_summary}
        Please reconsider your answer accordingly. Provide your answer and a step-by-step reasoning explanation.
        Please follow the format: 'Answer: {{}}. Explanation: {{}}.'"""
    if history:
        prompt = f"""You are an agent reading a document to answer a question.

        Question: {query}
        Document: {document}

        The following responses are from other agents as additional information.
        {history}
        Answer the question based on the document and other agents' response. Provide your answer and a step-by-step reasoning explanation.  
        Please follow the format: 'Answer: {{}}. Explanation: {{}}.''"""
    else:
        prompt = f"""You are an agent reading a document to answer a question.

        Question: {query}
        Document: {document}

        Answer the question based only on this document. Provide your answer and a step-by-step reasoning explanation.
        Please follow the format: 'Answer: {{}}. Explanation: {{}}.''"""

    output = call_llm(prompt, generator)
    return output


def aggregate_responses(query: str, responses: List[str], generator):
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
All Correct Answers: ["1963", "1956"]. Explanation: Agent 1 is talking about the basketball player Michael Jeffrey Jordan, who was born on Februray 17, 1963, so 1963 is correct. Agent 2 is talking about another person named Michael Jordan, who is an American scientist, and he was born in 1956. Therefore, the answer 1956 from Agent 2 is also correct. Agent 3 provides an error stating Michael Jordan's birth year as 1998, which is incorrect. Based on the correct information from Agent 1, Michael Jeffrey Jordan was born on February 17, 1963. Agent 4 does not provide any useful information.

Question: {query}
Agent responses:
{joined}
"""
    return call_llm(prompt, generator)


def multi_agent_debate(query: str, documents: List[str], generator, num_rounds: int = 3):
    records = {}
    num_agents = len(documents)
    agent_outputs = []

    # Round 1
    records["round1"] = {"answers": [], "explanations": []}
    for doc in documents:
        response = agent_response(query, doc, generator)
        answer = response[response.find("Answer: ") + len("Answer: "):response.find("Explanation")].strip()
        explanation = response[response.find("Explanation: ") + len("Explanation: "):]
        records["round1"]["answers"].append(answer)
        records["round1"]["explanations"].append(explanation)
        agent_outputs.append(response)
    records["round1"]["aggregation"] = aggregate_responses(query, agent_outputs, generator)
    prev_agg = records["round1"]["aggregation"]

    # Additional rounds
    final_aggregation = None
    for t in range(1, num_rounds):
        round_key = f"round{t+1}"
        records[round_key] = {"answers": [], "explanations": []}
        new_outputs = []
        for i, doc in enumerate(documents):
            #history = "\n".join([f"Agent {j+1}: {agent_outputs[j]}" for j in range(num_agents) if j != i])
            #response = agent_response(query, doc, generator, history)
            # Pass ONLY the aggregator summary from previous round, per paper
            response = agent_response(query, doc, generator, agg_summary=prev_agg)
            answer = response[response.find("Answer: ") + len("Answer: "):response.find("Explanation")].strip()
            explanation = response[response.find("Explanation: ") + len("Explanation: "):]
            records[round_key]["answers"].append(answer)
            records[round_key]["explanations"].append(explanation)
            new_outputs.append(response)
        agent_outputs = new_outputs
        pred_ans_list = []
        for ans in records[round_key]["answers"]:
            pred_ans_list.append(normalize_answer(ans))
        prev_pred_ans_list = []
        for ans in records[f"round{t}"]["answers"]:
            prev_pred_ans_list.append(normalize_answer(ans))
        assert len(pred_ans_list) == len(prev_pred_ans_list)
        flag = True
        for k in range(len(pred_ans_list)):
            if pred_ans_list[k] in prev_pred_ans_list[k] or prev_pred_ans_list[k] in pred_ans_list[k]:
                continue
            else:
                flag = False
        if flag:
            final_aggregation = prev_agg   # debate收敛，用上一轮聚合  #使用上一轮的聚合作为输出
            break
        else:
            records[round_key]["aggregation"] = aggregate_responses(query, agent_outputs, generator)
            final_aggregation = records[round_key]["aggregation"]  #保证n轮之后返回final aggregation
            prev_agg = records[round_key]["aggregation"]  # 更新，供下一轮使用

    records["final_aggregation"] = final_aggregation
    # 解析最终聚合答案，便于主程序做 EM
    records["final_answers"] = parse_aggregator_answers(final_aggregation or "")
    return records

def _read_existing_records(path: str):
    """读取已存在的输出 jsonl，返回 list[dict]（忽略空行/坏行）"""
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
                # 忽略坏行，避免卡住
                continue
    return recs

def _compute_em_over(recs: list, dataset: list):
    """
    用输出文件中的记录 + 原始数据集的 gold，一次性计算 EM。
    要求顺序一致：第 i 条 rec 对应 dataset[i]。
    若 rec 里没 final_answers，就从 final_aggregation 兜底解析。
    """
    total_em, counted = 0, 0
    N = min(len(recs), len(dataset))
    for i in range(N):
        rec = recs[i]
        ex  = dataset[i]
        pred = rec.get("final_answers") or parse_aggregator_answers(rec.get("final_aggregation", ""))
        gold = ex.get("gold_answers", [])
        em = em_score(pred, gold) if gold is not None else None
        if em is not None:
            total_em += em
            counted += 1
    return total_em, counted


def main():
    import argparse, os, json, gc, sys
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer, pipeline, set_seed
    )
    from tqdm import tqdm
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default="./cache")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--num_rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=int, default=0)  # 0=GPU0, -1=CPU
    args = parser.parse_args()

    hf_token = os.getenv("HF_TOKEN", None)
    args.output_path = f"{args.data_path}_madam_rag_{args.model_name.split('/')[-1]}_rounds{args.num_rounds}.jsonl"

    set_seed(args.seed)

    # 读入数据集
    with open(args.data_path, "r", encoding="utf-8") as f:
        all_data = [json.loads(line) for line in f]
    expected_n = len(all_data)  # 一般是 500；这里不写死，自动取数据集长度

    # 读取已有输出（如果有）
    existing_recs = _read_existing_records(args.output_path)
    existing_n = len(existing_recs)

    # 情况 A：已满（>= 数据集条数），直接算 EM 然后退出
    if existing_n >= expected_n:
        total_em, counted = _compute_em_over(existing_recs, all_data)
        if counted:
            print(f"[EM] {total_em}/{counted} = {total_em / counted * 100:.2f}%", flush=True)
        return

    # 情况 B：不满，先初始化模型/分词器，随后从缺失的条目开始续跑
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float32,             # 你的小模型设了 fp32；用 GPU 可考虑 bfloat16
        cache_dir=args.cache_dir,
        token=hf_token,
        local_files_only=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        cache_dir=args.cache_dir,
        token=hf_token,
        local_files_only=True,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        trust_remote_code=True,
        device=args.device,   # 推荐明确设 device，避免 device_map 带来的退出阻滞
    )

    # 续跑缺失样本并直接 append 到同一个 out 文件
    start_idx = existing_n
    with open(args.output_path, "a", encoding="utf-8") as out:
        for idx in tqdm(range(start_idx, expected_n), desc="MADAM-RAG resume"):
            ex = all_data[idx]
            query = ex["question"]
            documents = ex["documents"]

            rec = multi_agent_debate(query, documents, generator, num_rounds=args.num_rounds)

            pred = rec.get("final_answers") or parse_aggregator_answers(rec.get("final_aggregation", ""))
            gold = ex.get("gold_answers", [])
            em = em_score(pred, gold) if gold is not None else None

            rec["pred_answers"] = pred
            rec["gold_answers"] = gold
            rec["em"] = em
            rec["index"] = idx  # 方便排查/对齐

            out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # 续跑完成后，重新读取完整文件并计算最终 EM
    final_recs = _read_existing_records(args.output_path)
    total_em, counted = _compute_em_over(final_recs, all_data)
    if counted:
        print(f"[EM] {total_em}/{counted} = {total_em / counted * 100:.2f}%", flush=True)

    # 资源清理，避免进程卡在退出阶段
    try:
        del generator
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
    main()