#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ISAACLAB_ROOT="${ISAACLAB_ROOT:-/home/hul/workspace/IsaacLab}"

if [[ $# -eq 0 ]]; then
    echo "Usage: $0 --motion /path/to/g1_motion.npz [training options]" >&2
    exit 2
fi
if [[ ! -x "${ISAACLAB_ROOT}/isaaclab.sh" ]]; then
    echo "ISAACLAB_ROOT must point to an Isaac Lab checkout: ${ISAACLAB_ROOT}" >&2
    exit 2
fi

export PYTHONPATH="${REPO_ROOT}/source/unitracker_lab:${REPO_ROOT}/rsl_rl${PYTHONPATH:+:${PYTHONPATH}}"
for argument in "$@"; do
    if [[ "${argument}" == "--distributed" ]]; then
        : "${NUM_GPUS:?Set NUM_GPUS to the number of local GPUs when using --distributed.}"
        exec "${ISAACLAB_ROOT}/isaaclab.sh" -p -m torch.distributed.run \
            --nnodes=1 --nproc_per_node="${NUM_GPUS}" \
            "${REPO_ROOT}/scripts/rsl_rl/teacher/train_teacher.py" "$@"
    fi
done
exec "${ISAACLAB_ROOT}/isaaclab.sh" -p "${REPO_ROOT}/scripts/rsl_rl/teacher/train_teacher.py" "$@"
