#!/bin/bash
#SBATCH --partition=NV3090
#SBATCH --job-name=madam-dyprag
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH -o logs/%x-%j.out

# --------------------
# 基本设置：错误立刻退出
# --------------------
set -Eeuo pipefail

# 工作目录：你的 RAMDocs 项目路径
cd /share/home/yangjj/RAMDocs

# 激活 conda 环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dyprag

# 如果没建好日志和 cache 目录，可以在这里再保险建一次
 mkdir -p logs cache

# --------------------
# 要执行的 Python 命令
# 这里已经把你那串参数全部写进来了
# agent context switch (aggregator / history / both / none)
# message_transport options:
#    'text',
#    'dyprag', 'dyprag-combine',
#    'full-dyprag', 'full-dyprag-combine'
# per-round timing & GPU model logging
# --------------------
CMD=(
  python -u madam-dyprag.py
  --model_name qwen2.5-1.5b-instruct
  --data_path /share/home/yangjj/RAMDocs/RAMDocs_test.jsonl
  --num_rounds 5
  --message_transport dyprag-combine
  --projector_path /share/home/yangjj/DyPRAG/projector/qwen2.5-1.5b-instruct_hidden32_sample1.0_lr1e-05_augllama3.2-1b-instruct
  --inference_epoch 1
  --lora_rank 2
  --projector_p 32
  --agent_context history
  --debug
  #--use_param_agent
  --low_conf_ratio 0.2
  --low_conf_scale 0.5
)

# --------------------
# 把 shell 打印 + python 输出 一起通过 awk 加时间戳
# --------------------
{
  # 基本 SLURM 环境信息（相当于原 sh 里的 printf）
  printf "SLURM: job_id=%s step_id=%s name=%s partition=%s ntasks=%s cpus_per_task=%s gres=%s nodelist=%s\n" \
    "${SLURM_JOB_ID:-}" "${SLURM_STEP_ID:-}" "${SLURM_JOB_NAME:-}" "${SLURM_JOB_PARTITION:-}" \
    "${SLURM_NTASKS:-}" "${SLURM_CPUS_PER_TASK:-}" "${SLURM_JOB_GRES:-}" "${SLURM_NODELIST:-}"

  # scontrol 信息
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    scontrol show job "${SLURM_JOB_ID}" -o | sed "s/^/SCONTROL: /"
  fi

  # 要执行的命令
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
    # 在每一行前加时间戳
    print strftime("[%Y-%m-%d %H:%M:%S]"), $0;
    fflush();
  }
'
