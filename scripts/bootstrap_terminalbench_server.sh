#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 inspect|init" >&2
  exit 2
fi

MODE="$1"
case "$MODE" in
  inspect|init) ;;
  *)
    echo "unknown bootstrap mode: $MODE" >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
PYTHON="${SKILLOPT_PYTHON:-${PROJECT_ROOT}/.venv/bin/python}"
RUNTIME_ROOT="${SKILLOPT_RUNTIME_ROOT:-${PROJECT_ROOT}/../skillopt-runtime}"
EXPERIMENT_ID="${SKILLOPT_FORMAL_EXPERIMENT_ID:-tbench-v2.1-server-formal-001}"
CONCURRENCY="${SKILLOPT_TBENCH_CONCURRENCY:-1}"
TBENCH_ROOT="${TERMINALBENCH_ROOT:-${RUNTIME_ROOT}/datasets/terminal-bench-2-1}"
CACHE_ROOT="${TERMINALBENCH_FORMAL_CACHE_ROOT:-${RUNTIME_ROOT}/cache/terminal-bench-v2.1}"
SPLIT_ROOT="${TERMINALBENCH_SPLIT_DIR:-${RUNTIME_ROOT}/splits/tbench-v2.1-s42}"
HARBOR_CONFIG="${TERMINALBENCH_HARBOR_BASE_CONFIG:-${RUNTIME_ROOT}/harbor-configs/tbench-v2.1-formal.yaml}"
ENV_TEMPLATE_OUT="${SKILLOPT_FORMAL_ENV_TEMPLATE_OUT:-${RUNTIME_ROOT}/terminalbench-formal.env.example}"
EXPECTED_TBENCH_HEAD="7131e4375048a0e408a8fb404b5f499d726b695b"
EXPECTED_SPLIT_SHA="bd36fe2f37a67cd2b46149263522d833166d3a4d036c8e9af082e742ad017500"

if [[ ! "$CONCURRENCY" =~ ^[1-9][0-9]*$ ]]; then
  echo "SKILLOPT_TBENCH_CONCURRENCY must be a positive integer" >&2
  exit 2
fi

proxy_mode() {
  if [[ -n "${HTTP_PROXY:-}" || -n "${HTTPS_PROXY:-}" || -n "${http_proxy:-}" || -n "${https_proxy:-}" ]]; then
    if [[ -z "${HTTP_PROXY:-}" || -z "${HTTPS_PROXY:-}" ]]; then
      echo "proxy configuration is partial: set both HTTP_PROXY and HTTPS_PROXY" >&2
      return 2
    fi
    if [[ -n "${http_proxy:-}" && "$http_proxy" != "$HTTP_PROXY" ]]; then
      echo "HTTP_PROXY and http_proxy differ" >&2
      return 2
    fi
    if [[ -n "${https_proxy:-}" && "$https_proxy" != "$HTTPS_PROXY" ]]; then
      echo "HTTPS_PROXY and https_proxy differ" >&2
      return 2
    fi
    echo environment
  else
    echo direct
  fi
}

PROXY_MODE="$(proxy_mode)"

inspect_host() {
  local blockers=0
  echo "PROJECT_ROOT=$PROJECT_ROOT"
  echo "SKILLOPT_RUNTIME_ROOT=$RUNTIME_ROOT"
  echo "SKILLOPT_FORMAL_EXPERIMENT_ID=$EXPERIMENT_ID"
  echo "SKILLOPT_TBENCH_CONCURRENCY=$CONCURRENCY"
  echo "PROXY_MODE=$PROXY_MODE"

  if command -v docker >/dev/null 2>&1; then
    echo "DOCKER_CLI=$(docker --version 2>/dev/null || echo UNAVAILABLE)"
    if docker info >/dev/null 2>&1; then
      echo "DOCKER_ACCESS=PASS"
      docker info --format 'DOCKER_ROOT_DIR={{.DockerRootDir}}' 2>/dev/null || true
      docker info --format 'DOCKER_DEFAULT_ADDRESS_POOLS={{json .DefaultAddressPools}}' 2>/dev/null || true
    else
      echo "DOCKER_ACCESS=BLOCK"
      blockers=$((blockers + 1))
    fi
  else
    echo "DOCKER_CLI=MISSING"
    blockers=$((blockers + 1))
  fi

  if command -v python3 >/dev/null 2>&1; then
    echo "PYTHON=$(python3 --version 2>&1)"
  else
    echo "PYTHON=MISSING"
    blockers=$((blockers + 1))
  fi
  if [[ -x "$PYTHON" ]]; then
    echo "REPO_VENV=PASS"
  else
    echo "REPO_VENV=BLOCK path=$PYTHON"
    blockers=$((blockers + 1))
  fi
  if command -v uv >/dev/null 2>&1; then
    echo "UV=$(uv --version 2>&1)"
  else
    echo "UV=OPTIONAL_MISSING"
  fi

  if command -v harbor >/dev/null 2>&1 && [[ -x "$PYTHON" ]]; then
    if "$PYTHON" -c '
from scripts.preflight_terminalbench import _harbor_version, _terminus_version
harbor = _harbor_version("harbor")
terminus = _terminus_version("harbor")
if harbor != "0.20.0" or terminus != "2.0.0":
    raise SystemExit(f"Harbor/Terminus mismatch: {harbor}/{terminus}")
print(f"HARBOR={harbor}")
print(f"TERMINUS_2={terminus}")
'; then
      echo "HARBOR_RUNTIME=PASS"
    else
      echo "HARBOR_RUNTIME=BLOCK"
      blockers=$((blockers + 1))
    fi
  else
    echo "HARBOR_RUNTIME=BLOCK"
    blockers=$((blockers + 1))
  fi

  if command -v systemctl >/dev/null 2>&1 && systemctl --user show-environment >/dev/null 2>&1; then
    echo "SYSTEMD_USER=PASS"
  else
    echo "SYSTEMD_USER=BLOCK"
    blockers=$((blockers + 1))
  fi

  echo "LOGICAL_CPUS=$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo UNRESOLVED)"
  awk '/MemTotal:/ {print "RAM_TOTAL_KIB=" $2} /MemAvailable:/ {print "RAM_AVAILABLE_KIB=" $2}' /proc/meminfo 2>/dev/null || true
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | sed 's/^/GPU=/' || echo "GPU=UNRESOLVED"
  else
    echo "GPU=NONE_OR_UNRESOLVED"
  fi
  for path in "$RUNTIME_ROOT" "$TBENCH_ROOT" "$CACHE_ROOT"; do
    local probe="$path"
    while [[ ! -e "$probe" && "$probe" != "/" ]]; do
      probe="$(dirname -- "$probe")"
    done
    df -Pk "$probe" 2>/dev/null | tail -n 1 | awk -v target="$path" '{print "DISK_FREE_KIB[" target "]=" $4}' || true
  done

  if [[ -d "$TBENCH_ROOT/.git" ]]; then
    local tbench_head tbench_status
    tbench_head="$(git -C "$TBENCH_ROOT" rev-parse HEAD 2>/dev/null || true)"
    tbench_status="$(git -C "$TBENCH_ROOT" status --short 2>/dev/null || true)"
    if [[ "$tbench_head" == "$EXPECTED_TBENCH_HEAD" && -z "$tbench_status" ]]; then
      echo "TBENCH_CHECKOUT=PASS head=$tbench_head"
    else
      echo "TBENCH_CHECKOUT=BLOCK head=${tbench_head:-MISSING}"
      blockers=$((blockers + 1))
    fi
  else
    echo "TBENCH_CHECKOUT=BLOCK path=$TBENCH_ROOT"
    blockers=$((blockers + 1))
  fi

  if [[ -x "$PYTHON" && -f "$CACHE_ROOT/MANIFEST.tsv" ]]; then
    if "$PYTHON" -c '
from pathlib import Path
import sys
from scripts.preflight_terminalbench import _validate_cache_contract
state = _validate_cache_contract(Path(sys.argv[1]))
print("CACHE_MANIFEST_SHA256=" + state["manifest_sha256"])
' "$CACHE_ROOT"; then
      echo "FORMAL_CACHE=PASS"
    else
      echo "FORMAL_CACHE=BLOCK"
      blockers=$((blockers + 1))
    fi
  else
    echo "FORMAL_CACHE=BLOCK path=$CACHE_ROOT"
    blockers=$((blockers + 1))
  fi

  if (( blockers > 0 )); then
    echo "BOOTSTRAP_INSPECT=BLOCK blockers=$blockers"
    return 2
  fi
  echo "BOOTSTRAP_INSPECT=PASS"
}

init_runtime() {
  if [[ ! -x "$PYTHON" ]]; then
    echo "repo Python is required for init: $PYTHON" >&2
    exit 2
  fi
  mkdir -p \
    "$RUNTIME_ROOT/datasets" \
    "$RUNTIME_ROOT/splits" \
    "$RUNTIME_ROOT/harbor-configs" \
    "$RUNTIME_ROOT/outputs" \
    "$RUNTIME_ROOT/skills" \
    "$RUNTIME_ROOT/cache" \
    "$RUNTIME_ROOT/logs" \
    "$RUNTIME_ROOT/locks"

  if [[ -e "$SPLIT_ROOT" ]]; then
    actual_split_sha="$($PYTHON -c '
import json, sys
from pathlib import Path
print(json.loads((Path(sys.argv[1]) / "split_manifest.json").read_text())["semantic_sha256"])
' "$SPLIT_ROOT")"
    if [[ "$actual_split_sha" != "$EXPECTED_SPLIT_SHA" ]]; then
      echo "existing runtime split has unexpected semantic SHA-256" >&2
      exit 2
    fi
  else
    mkdir -p "$(dirname -- "$SPLIT_ROOT")"
    cp -R "$PROJECT_ROOT/configs/terminalbench/splits/v2.1-s42" "$SPLIT_ROOT"
  fi

  if [[ -e "$HARBOR_CONFIG" ]]; then
    echo "Harbor config already exists; refusing to overwrite: $HARBOR_CONFIG" >&2
    exit 2
  fi
  "$PYTHON" "$SCRIPT_DIR/render_terminalbench_harbor_config.py" \
    --runtime-root "$RUNTIME_ROOT" \
    --tasks-path "$TBENCH_ROOT/tasks" \
    --cache-root "$CACHE_ROOT" \
    --concurrency "$CONCURRENCY" \
    --proxy-mode "$PROXY_MODE" \
    --output "$HARBOR_CONFIG" >/dev/null

  if [[ -e "$ENV_TEMPLATE_OUT" ]]; then
    echo "EnvironmentFile template already exists; refusing to overwrite: $ENV_TEMPLATE_OUT" >&2
    exit 2
  fi
  cp "$PROJECT_ROOT/configs/terminalbench/terminalbench-formal.env.example" "$ENV_TEMPLATE_OUT"

  echo "BOOTSTRAP_INIT=PASS"
  echo "RUNTIME_ROOT=$RUNTIME_ROOT"
  echo "PORTABLE_SPLIT=$SPLIT_ROOT"
  echo "PORTABLE_SPLIT_SHA256=$EXPECTED_SPLIT_SHA"
  echo "HARBOR_CONFIG=$HARBOR_CONFIG"
  echo "ENVIRONMENT_TEMPLATE=$ENV_TEMPLATE_OUT"
  echo "NEXT=copy the EnvironmentFile template outside the repo, set credentials and reviewed paths, then run inspect"
}

if [[ "$MODE" == "inspect" ]]; then
  inspect_host
else
  init_runtime
fi
