#!/usr/bin/env bash
# 或者直接写：#!/bin/bash

set -Eeuo pipefail

########################
# 1. 基本环境初始化
########################

# 项目目录
cd /share/home/yangjj/RAMDocs

# 激活 conda 环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate faithful-rag

# 确保日志目录存在
mkdir -p logs cache

########################
# 2. 定义要执行的 python 命令
########################
CMD=(
  python -u madam-dyprag.py
  --model_name qwen2.5-1.5b-instruct
  --data_path /share/home/yangjj/RAMDocs/RAMDocs_test.jsonl
  --num_rounds 3
  --message_transport text
  --projector_path /share/home/yangjj/DyPRAG/projector/qwen2.5-1.5b-instruct_hidden32_sample1.0_lr1e-05_augllama3.2-1b-instruct
  --inference_epoch 1
  --lora_rank 2
  --projector_p 32
  --agent_context aggregator
  --debug
)

########################
# 3. 执行命令 + 时间戳日志
########################

LOG_FILE="logs/madam-dyprag-$(date +%Y%m%d-%H%M%S).log"

{
  # 打印 SLURM 环境信息（如果当前是在 srun/salloc 里，这些变量会存在）
  printf "SLURM: job_id=%s step_id=%s name=%s partition=%s ntasks=%s cpus_per_task=%s gres=%s nodelist=%s\n" \
    "${SLURM_JOB_ID:-}" "${SLURM_STEP_ID:-}" "${SLURM_JOB_NAME:-}" "${SLURM_JOB_PARTITION:-}" \
    "${SLURM_NTASKS:-}" "${SLURM_CPUS_PER_TASK:-}" "${SLURM_JOB_GRES:-}" "${SLURM_NODELIST:-}"

  # scontrol 信息（如果有 job id）
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    scontrol show job "${SLURM_JOB_ID}" -o | sed "s/^/SCONTROL: /"
  fi

  # 打印 CMD
  printf "CMD: "
  printf "%q " "${CMD[@]}"
  echo

  export PYTHONUNBUFFERED=1

  # 真正执行 madam-dyprag.py
  "${CMD[@]}"

} 2>&1 | awk '
  BEGIN { fflush() }
  {
    gsub(/\r/, "", $0);
    print strftime("[%Y-%m-%d %H:%M:%S]"), $0;
    fflush();
  }
' | tee "$LOG_FILE"
