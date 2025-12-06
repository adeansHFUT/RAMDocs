#!/bin/bash
#SBATCH --partition=NV3090
#SBATCH --job-name=madam-dyprag
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=7:00:00
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
  --num_rounds 3
  --message_transport dyprag
  --projector_path /share/home/yangjj/DyPRAG/projector/qwen2.5-1.5b-instruct_hidden32_sample1.0_lr1e-05_augllama3.2-1b-instruct
  --inference_epoch 1
  --lora_rank 2
  --projector_p 32
  --agent_context history
  --debug
  --first_inject 2
  #--use_param_agent

  --scale_mode step # step / continuous / no
  --low_conf_ratio 0.4
  --low_conf_scale 0.8
  --high_conf_ratio 0
  --high_conf_scale 1
  --continuous_s_min 0.1
  --continuous_s_max 1.0 
  --continuous_temperature 1.0 #控制的是“置信度 → 缩放系数”的曲线有多陡、多“硬”。
  --round_aware_scaling  #后面round缩放得更狠

  --dynamic_agents  # 是否动态剪枝agent
  --dynamic_agent_frac 0.75 #表示每轮只保留 top n 置信度最高的 agents；
  --dynamic_agent_min 2 #保证至少保留 n个。

  #--stop_mode agent_answers #保持原来 per-agent 答案一致性（无动态时完全兼容旧逻辑，有动态时退化为答案集合一致）。
  --stop_mode agent_answers_set
  #--stop_mode aggregator #看聚合器两轮的 answers set 是否稳定。
  #--stop_mode cluster #看 agent 答案簇结构是否稳定（key 一致且相对频率变化小于 --cluster_stable_tol）
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
