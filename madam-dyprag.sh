#!/usr/bin/env bash
set -euo pipefail

printf "[%(%F %T)T] SLURM: job_id=%s step_id=%s name=%s partition=%s ntasks=%s cpus_per_task=%s gres=%s nodelist=%s\n" \
  -1 "${SLURM_JOB_ID:-}" "${SLURM_STEP_ID:-}" "${SLURM_JOB_NAME:-}" "${SLURM_JOB_PARTITION:-}" \
  "${SLURM_NTASKS:-}" "${SLURM_CPUS_PER_TASK:-}" "${SLURM_JOB_GRES:-}" "${SLURM_NODELIST:-}"

scontrol show job "${SLURM_JOB_ID:-}" -o | sed "s/^/[${SLURM_JOB_ID:-} $(date +'%F %T')] SCONTROL: /"

CMD=(python -u madam-dyprag.py "$@")
printf "[%(%F %T)T] CMD: " -1; printf "%q " "${CMD[@]}"; echo

export PYTHONUNBUFFERED=1
exec "${CMD[@]}"
