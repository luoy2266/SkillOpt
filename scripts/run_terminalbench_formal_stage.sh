#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 probe|preflight|training|freeze-skill|baseline-test|skill-test|aggregate" >&2
  exit 2
fi

STAGE="$1"
case "$STAGE" in
  probe|preflight|training|freeze-skill|baseline-test|skill-test|aggregate) ;;
  *)
    echo "unknown formal stage: $STAGE" >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
PYTHON="${SKILLOPT_PYTHON:-${PROJECT_ROOT}/.venv/bin/python}"
EXPERIMENT_ID_INPUT="${SKILLOPT_FORMAL_EXPERIMENT_ID:-}"
RUNTIME_ROOT_INPUT="${SKILLOPT_RUNTIME_ROOT:-}"
CONCURRENCY_INPUT="${SKILLOPT_TBENCH_CONCURRENCY:-}"
if [[ -z "$EXPERIMENT_ID_INPUT" || "$EXPERIMENT_ID_INPUT" == \<*\> ]]; then
  echo "OPERATOR INPUT REQUIRED: SKILLOPT_FORMAL_EXPERIMENT_ID" >&2
  exit 2
fi
if [[ -z "$RUNTIME_ROOT_INPUT" || "$RUNTIME_ROOT_INPUT" == \<*\> ]]; then
  echo "OPERATOR INPUT REQUIRED: SKILLOPT_RUNTIME_ROOT" >&2
  exit 2
fi
if [[ -z "$CONCURRENCY_INPUT" || "$CONCURRENCY_INPUT" == \<*\> ]]; then
  echo "OPERATOR INPUT REQUIRED: SKILLOPT_TBENCH_CONCURRENCY" >&2
  exit 2
fi
if [[ ! "$CONCURRENCY_INPUT" =~ ^[1-9][0-9]*$ ]]; then
  echo "SKILLOPT_TBENCH_CONCURRENCY must be a positive integer" >&2
  exit 2
fi
EXPERIMENT_ID="$EXPERIMENT_ID_INPUT"
if [[ ! -x "$PYTHON" ]]; then
  echo "SKILLOPT_PYTHON is not executable: $PYTHON" >&2
  exit 2
fi
RUNTIME_ROOT="$($PYTHON -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' \
  "$RUNTIME_ROOT_INPUT")"
FORMAL_ROOT="${SKILLOPT_FORMAL_ROOT:-${RUNTIME_ROOT}/outputs/formal/${EXPERIMENT_ID}}"
TBENCH_ROOT="${TERMINALBENCH_ROOT:-${RUNTIME_ROOT}/datasets/terminal-bench-2-1}"
SPLIT_DIR="${TERMINALBENCH_SPLIT_DIR:-${RUNTIME_ROOT}/splits/tbench-v2.1-s42}"
HARBOR_BASE_CONFIG="${TERMINALBENCH_HARBOR_BASE_CONFIG:-${RUNTIME_ROOT}/harbor-configs/tbench-v2.1-formal.yaml}"
FORMAL_CONFIG="${SKILLOPT_FORMAL_CONFIG:-${PROJECT_ROOT}/configs/terminalbench/formal.yaml}"
CACHE_ROOT="${TERMINALBENCH_FORMAL_CACHE_ROOT:-${RUNTIME_ROOT}/cache/terminal-bench-v2.1}"
LOCK_ROOT="${SKILLOPT_FORMAL_LOCK_ROOT:-${RUNTIME_ROOT}/locks}"
SKILLS_ROOT="${SKILLOPT_FORMAL_SKILLS_ROOT:-${RUNTIME_ROOT}/skills/${EXPERIMENT_ID}}"
INITIAL_SKILL="${PROJECT_ROOT}/skillopt/envs/terminalbench/skills/initial.md"
TRAINING_OUT="${FORMAL_ROOT}/training"
TRAINING_MANIFEST="${FORMAL_ROOT}/manifests/training.experiment_manifest.json"
FROZEN_SKILL="${SKILLS_ROOT}/best_skill.md"
SKILL_PROVENANCE="${SKILLS_ROOT}/skill_provenance.json"
OUTPUT_ROOT="${FORMAL_ROOT}/${STAGE}"
MANIFEST="${FORMAL_ROOT}/manifests/${STAGE}.experiment_manifest.json"
if [[ -n "${SKILLOPT_FORMAL_LOG_ROOT:-}" ]]; then
  LOG_ROOT="$SKILLOPT_FORMAL_LOG_ROOT"
elif [[ -n "${SKILLOPT_FORMAL_ROOT:-}" ]]; then
  LOG_ROOT="${FORMAL_ROOT}/logs"
else
  LOG_ROOT="${RUNTIME_ROOT}/logs/formal/${EXPERIMENT_ID}"
fi
LOG_PATH="${LOG_ROOT}/${STAGE}.console.log"

: "${SKILLOPT_FORMAL_HEAD:?Set SKILLOPT_FORMAL_HEAD to the reviewed committed HEAD}"
: "${SKILLOPT_FORMAL_DOCKER_MODE:?SKILLOPT_FORMAL_DOCKER_MODE must be set by the sg docker service path}"
: "${DEEPSEEK_API_KEY:?DEEPSEEK_API_KEY is required}"
if [[ "$SKILLOPT_FORMAL_DOCKER_MODE" != "sg" ]]; then
  echo "SKILLOPT_FORMAL_DOCKER_MODE=FAIL" >&2
  exit 2
fi
CONCURRENCY="$CONCURRENCY_INPUT"

export OPTIMIZER_OPENAI_COMPATIBLE_BASE_URL="https://api.deepseek.com"
export OPTIMIZER_OPENAI_COMPATIBLE_API_KEY="$DEEPSEEK_API_KEY"
export SKILLOPT_RUNTIME_ROOT="$RUNTIME_ROOT"
export SKILLOPT_TBENCH_CONCURRENCY="$CONCURRENCY"
export SKILLOPT_FORMAL_SKILLS_ROOT="$SKILLS_ROOT"
export TERMINALBENCH_FORMAL_CACHE_ROOT="$CACHE_ROOT"

if [[ -n "${HTTP_PROXY:-}" || -n "${HTTPS_PROXY:-}" || -n "${http_proxy:-}" || -n "${https_proxy:-}" ]]; then
  if [[ -z "${HTTP_PROXY:-}" || -z "${HTTPS_PROXY:-}" ]]; then
    echo "HTTP_PROXY and HTTPS_PROXY must both be set when proxying is enabled" >&2
    exit 2
  fi
  if [[ -n "${http_proxy:-}" && "$http_proxy" != "$HTTP_PROXY" ]]; then
    echo "HTTP_PROXY and http_proxy differ" >&2
    exit 2
  fi
  if [[ -n "${https_proxy:-}" && "$https_proxy" != "$HTTPS_PROXY" ]]; then
    echo "HTTPS_PROXY and https_proxy differ" >&2
    exit 2
  fi
  export http_proxy="$HTTP_PROXY"
  export https_proxy="$HTTPS_PROXY"
else
  unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
fi

if [[ -n "${NO_PROXY:-}" && -n "${no_proxy:-}" && "$NO_PROXY" != "$no_proxy" ]]; then
  echo "NO_PROXY and no_proxy differ" >&2
  exit 2
fi
NO_PROXY="${NO_PROXY:-${no_proxy:-}}"
DOCKER_NETWORK_IDS="$(docker network ls -q)"
if [[ -n "$DOCKER_NETWORK_IDS" ]]; then
  DOCKER_NETWORK_JSON="$(docker network inspect $DOCKER_NETWORK_IDS)"
else
  DOCKER_NETWORK_JSON='[]'
fi
DOCKER_LOCAL_ADDRESSES_TEXT="$({
  printf '%s' "$DOCKER_NETWORK_JSON" | "$PYTHON" -c '
import json, sys
for network in json.load(sys.stdin):
    for entry in (network.get("IPAM") or {}).get("Config") or []:
        for key in ("Subnet", "Gateway"):
            value = str(entry.get(key) or "").strip()
            if value:
                print(value)
'
})"
mapfile -t DOCKER_LOCAL_ADDRESSES <<< "$DOCKER_LOCAL_ADDRESSES_TEXT"
for local_entry in localhost 127.0.0.1 127.0.0.11 ::1 "${DOCKER_LOCAL_ADDRESSES[@]}"; do
  [[ -n "$local_entry" ]] || continue
  case ",${NO_PROXY}," in
    *",${local_entry},"*) ;;
    *)
      if [[ -n "$NO_PROXY" ]]; then
        NO_PROXY="${NO_PROXY},${local_entry}"
      else
        NO_PROXY="$local_entry"
      fi
      ;;
  esac
done
export NO_PROXY
export no_proxy="$NO_PROXY"

acquire_experiment_lock() {
  if ! command -v flock >/dev/null 2>&1; then
    echo "flock is required for formal execution stages" >&2
    exit 2
  fi
  mkdir -p "$LOCK_ROOT"
  local lock_name
  lock_name="$($PYTHON "$SCRIPT_DIR/terminalbench_formal_identity.py" \
    lock-name --experiment-id "$EXPERIMENT_ID")"
  exec 9>"${LOCK_ROOT}/${lock_name}"
  if ! flock -n 9; then
    echo "formal experiment is already running: $EXPERIMENT_ID" >&2
    exit 2
  fi
}

case "$STAGE" in
  training|freeze-skill|baseline-test|skill-test|aggregate) acquire_experiment_lock ;;
esac

cd "$PROJECT_ROOT"

if [[ "$STAGE" == "probe" ]]; then
  exec "$PYTHON" \
    "$PROJECT_ROOT/scripts/probe_terminalbench_formal_service.py"
fi

mkdir -p "${FORMAL_ROOT}/manifests" "$LOG_ROOT"
exec > >(tee -a "$LOG_PATH") 2>&1

PREFLIGHT_CONDITION="$STAGE"
if [[ "$STAGE" == "preflight" ]]; then
  PREFLIGHT_CONDITION="training"
fi

PREFLIGHT=(
  "$PYTHON" scripts/preflight_terminalbench.py
  --config "$FORMAL_CONFIG"
  --tbench-root "$TBENCH_ROOT"
  --split-dir "$SPLIT_DIR"
  --harbor-base-config "$HARBOR_BASE_CONFIG"
  --cache-root "$CACHE_ROOT"
  --output-root "$OUTPUT_ROOT"
  --manifest-out "$MANIFEST"
  --log-path "$LOG_PATH"
  --experiment-id "$EXPERIMENT_ID"
  --condition "$PREFLIGHT_CONDITION"
  --expected-skillopt-head "$SKILLOPT_FORMAL_HEAD"
  --concurrency "$CONCURRENCY"
  --require-persistent-runtime
)

COMMON_OVERRIDES=(
  "env.split_dir=${SPLIT_DIR}"
  "env.harbor_base_config=${HARBOR_BASE_CONFIG}"
  "env.out_root=${OUTPUT_ROOT}"
  "env.n_concurrent_trials=${CONCURRENCY}"
)

run_formal_preflight() {
  "${PREFLIGHT[@]}" "$@"
}

case "$STAGE" in
  freeze-skill)
    exec "$PYTHON" scripts/freeze_terminalbench_skill.py \
      --experiment-id "$EXPERIMENT_ID" \
      --training-output "$TRAINING_OUT" \
      --training-manifest "$TRAINING_MANIFEST" \
      --output-root "$SKILLS_ROOT" \
      --expected-skillopt-head "$SKILLOPT_FORMAL_HEAD"
    ;;
  aggregate)
    exec "$PYTHON" scripts/aggregate_terminalbench_results.py \
      --experiment-id "$EXPERIMENT_ID" \
      --baseline-output "${FORMAL_ROOT}/baseline-test" \
      --baseline-manifest "${FORMAL_ROOT}/manifests/baseline-test.experiment_manifest.json" \
      --skill-output "${FORMAL_ROOT}/skill-test" \
      --skill-manifest "${FORMAL_ROOT}/manifests/skill-test.experiment_manifest.json" \
      --skill-provenance "$SKILL_PROVENANCE" \
      --output-root "$OUTPUT_ROOT"
    ;;
  preflight)
    run_formal_preflight --skill "$INITIAL_SKILL"
    exit 0
    ;;
  training)
    run_formal_preflight --skill "$INITIAL_SKILL"
    exec "$PYTHON" scripts/train.py \
      --config "$FORMAL_CONFIG" \
      --cfg-options "${COMMON_OVERRIDES[@]}"
    ;;
  baseline-test)
    run_formal_preflight --skill "$INITIAL_SKILL"
    exec "$PYTHON" scripts/eval_only.py \
      --config "$FORMAL_CONFIG" \
      --skill "$INITIAL_SKILL" \
      --split valid_unseen \
      --cfg-options "${COMMON_OVERRIDES[@]}"
    ;;
  skill-test)
    run_formal_preflight \
      --skill "$FROZEN_SKILL" \
      --skill-provenance "$SKILL_PROVENANCE"
    exec "$PYTHON" scripts/eval_only.py \
      --config "$FORMAL_CONFIG" \
      --skill "$FROZEN_SKILL" \
      --split valid_unseen \
      --cfg-options "${COMMON_OVERRIDES[@]}"
    ;;
esac
