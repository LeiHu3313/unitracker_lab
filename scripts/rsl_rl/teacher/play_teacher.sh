#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ISAACLAB_ROOT="${ISAACLAB_ROOT:-/home/hul/workspace/IsaacLab}"

if [[ $# -eq 0 ]]; then
    echo "Usage: $0 --motion /path/to/g1_motion.npz --checkpoint_path /path/to/model.pt" >&2
    exit 2
fi
if [[ ! -x "${ISAACLAB_ROOT}/isaaclab.sh" ]]; then
    echo "ISAACLAB_ROOT must point to an Isaac Lab checkout: ${ISAACLAB_ROOT}" >&2
    exit 2
fi

export PYTHONPATH="${REPO_ROOT}/source/unitracker_lab:${REPO_ROOT}/rsl_rl${PYTHONPATH:+:${PYTHONPATH}}"
exec "${ISAACLAB_ROOT}/isaaclab.sh" -p "${REPO_ROOT}/scripts/rsl_rl/teacher/play_teacher.py" "$@"
