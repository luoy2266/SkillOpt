#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 probe|training|baseline-test|skill-test" >&2
  exit 2
fi

STAGE="$1"
case "$STAGE" in
  probe|training|baseline-test|skill-test) ;;
  *)
    echo "unknown formal stage: $STAGE" >&2
    exit 2
    ;;
esac

PROJECT_ROOT="/home/yunl/projects/SkillOpt"
EXPERIMENT_ID="${SKILLOPT_FORMAL_EXPERIMENT_ID:-tbench-v2.1-dsv4flash-s42-formal-001}"
FORMAL_ROOT="${SKILLOPT_FORMAL_ROOT:-/home/yunl/projects/skillopt-runtime/outputs/formal/${EXPERIMENT_ID}}"
TBENCH_ROOT="${TERMINALBENCH_ROOT:-/home/yunl/projects/skillopt-runtime/datasets/terminal-bench-2-1}"
SPLIT_DIR="${TERMINALBENCH_SPLIT_DIR:-/home/yunl/projects/skillopt-runtime/splits/tbench-v2.1-1-1-8}"
HARBOR_BASE_CONFIG="${TERMINALBENCH_HARBOR_BASE_CONFIG:-/home/yunl/projects/skillopt-runtime/harbor-configs/tbench-v2.1-dsv4flash-docker-formal.yaml}"
FORMAL_CONFIG="${SKILLOPT_FORMAL_CONFIG:-${PROJECT_ROOT}/configs/terminalbench/formal.yaml}"
CACHE_ROOT="${TERMINALBENCH_FORMAL_CACHE_ROOT:-/home/yunl/projects/skillopt-runtime/cache/terminal-bench-v2.1}"
INITIAL_SKILL="${PROJECT_ROOT}/skillopt/envs/terminalbench/skills/initial.md"
TRAINING_OUT="${FORMAL_ROOT}/training"
OUTPUT_ROOT="${FORMAL_ROOT}/${STAGE}"
MANIFEST="${FORMAL_ROOT}/manifests/${STAGE}.experiment_manifest.json"
LOG_PATH="${FORMAL_ROOT}/logs/${STAGE}.console.log"

: "${SKILLOPT_FORMAL_HEAD:?Set SKILLOPT_FORMAL_HEAD to the reviewed committed HEAD}"
: "${SKILLOPT_FORMAL_DOCKER_MODE:?SKILLOPT_FORMAL_DOCKER_MODE must be set by the sg docker service path}"
: "${DEEPSEEK_API_KEY:?DEEPSEEK_API_KEY is required}"
: "${HTTP_PROXY:?HTTP_PROXY is required}"
: "${HTTPS_PROXY:?HTTPS_PROXY is required}"
: "${NO_PROXY:?NO_PROXY is required}"
if [[ "$SKILLOPT_FORMAL_DOCKER_MODE" != "sg" ]]; then
  echo "SKILLOPT_FORMAL_DOCKER_MODE=FAIL" >&2
  exit 2
fi

export OPTIMIZER_OPENAI_COMPATIBLE_BASE_URL="https://api.deepseek.com"
export OPTIMIZER_OPENAI_COMPATIBLE_API_KEY="$DEEPSEEK_API_KEY"
export http_proxy="$HTTP_PROXY"
export https_proxy="$HTTPS_PROXY"
export TERMINALBENCH_FORMAL_CACHE_ROOT="$CACHE_ROOT"
DOCKER_NETWORK_IDS="$(docker network ls -q)"
DOCKER_NETWORK_JSON="$(docker network inspect $DOCKER_NETWORK_IDS)"
DOCKER_LOCAL_ADDRESSES_TEXT="$({
  printf '%s' "$DOCKER_NETWORK_JSON" | "$PROJECT_ROOT/.venv/bin/python" -c '
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
    *) NO_PROXY="${NO_PROXY},${local_entry}" ;;
  esac
done
export NO_PROXY
export no_proxy="$NO_PROXY"

cd "$PROJECT_ROOT"

if [[ "$STAGE" == "probe" ]]; then
  exec "$PROJECT_ROOT/.venv/bin/python" \
    "$PROJECT_ROOT/scripts/probe_terminalbench_formal_service.py"
fi

mkdir -p "${FORMAL_ROOT}/manifests" "${FORMAL_ROOT}/logs"
exec > >(tee -a "$LOG_PATH") 2>&1

PREFLIGHT=(
  .venv/bin/python scripts/preflight_terminalbench.py
  --config "$FORMAL_CONFIG"
  --tbench-root "$TBENCH_ROOT"
  --split-dir "$SPLIT_DIR"
  --harbor-base-config "$HARBOR_BASE_CONFIG"
  --cache-root "$CACHE_ROOT"
  --output-root "$OUTPUT_ROOT"
  --manifest-out "$MANIFEST"
  --log-path "$LOG_PATH"
  --experiment-id "$EXPERIMENT_ID"
  --condition "$STAGE"
  --expected-skillopt-head "$SKILLOPT_FORMAL_HEAD"
  --require-persistent-runtime
)

COMMON_OVERRIDES=(
  "env.split_dir=${SPLIT_DIR}"
  "env.harbor_base_config=${HARBOR_BASE_CONFIG}"
  "env.out_root=${OUTPUT_ROOT}"
)

case "$STAGE" in
  training)
    "${PREFLIGHT[@]}" --skill "$INITIAL_SKILL"
    exec .venv/bin/python scripts/train.py \
      --config "$FORMAL_CONFIG" \
      --cfg-options "${COMMON_OVERRIDES[@]}"
    ;;
  baseline-test)
    "${PREFLIGHT[@]}" --skill "$INITIAL_SKILL"
    exec .venv/bin/python scripts/eval_only.py \
      --config "$FORMAL_CONFIG" \
      --skill "$INITIAL_SKILL" \
      --split valid_unseen \
      --cfg-options "${COMMON_OVERRIDES[@]}"
    ;;
  skill-test)
    LEARNED_SKILL="${TRAINING_OUT}/best_skill.md"
    "${PREFLIGHT[@]}" \
      --skill "$LEARNED_SKILL" \
      --training-output "$TRAINING_OUT"
    exec .venv/bin/python scripts/eval_only.py \
      --config "$FORMAL_CONFIG" \
      --skill "$LEARNED_SKILL" \
      --split valid_unseen \
      --cfg-options "${COMMON_OVERRIDES[@]}"
    ;;
esac
