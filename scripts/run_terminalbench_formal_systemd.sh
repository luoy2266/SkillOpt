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
ENV_FILE="${SKILLOPT_FORMAL_ENV_FILE:-${XDG_CONFIG_HOME:-${HOME}/.config}/skillopt/terminalbench-formal.env}"
WRAPPER="${SCRIPT_DIR}/run_terminalbench_formal_stage.sh"

if [[ ! -r "$ENV_FILE" ]]; then
  echo "ENVIRONMENT_FILE=MISSING" >&2
  exit 2
fi
if [[ ! -x "$PYTHON" ]]; then
  echo "SKILLOPT_PYTHON is not executable: $PYTHON" >&2
  exit 2
fi

read_required_environment_value() {
  local name="$1"
  local value
  if ! value="$($PYTHON "$SCRIPT_DIR/terminalbench_formal_identity.py" \
    env-value --file "$ENV_FILE" --name "$name" 2>/dev/null)" \
    || [[ -z "$value" || "$value" == \<*\> ]]; then
    echo "OPERATOR INPUT REQUIRED: $name" >&2
    exit 2
  fi
  printf '%s' "$value"
}

EXPERIMENT_ID="$(read_required_environment_value SKILLOPT_FORMAL_EXPERIMENT_ID)"
read_required_environment_value SKILLOPT_RUNTIME_ROOT >/dev/null
CONCURRENCY="$(read_required_environment_value SKILLOPT_TBENCH_CONCURRENCY)"
if [[ ! "$CONCURRENCY" =~ ^[1-9][0-9]*$ ]]; then
  echo "SKILLOPT_TBENCH_CONCURRENCY must be a positive integer" >&2
  exit 2
fi
UNIT="$($PYTHON "$SCRIPT_DIR/terminalbench_formal_identity.py" \
  unit-name --experiment-id "$EXPERIMENT_ID" --stage "$STAGE")"
printf -v WRAPPER_COMMAND 'SKILLOPT_FORMAL_DOCKER_MODE=sg exec %q %q' "$WRAPPER" "$STAGE"

WAIT_ARGS=()
if [[ "$STAGE" == "probe" || "$STAGE" == "preflight" || "$STAGE" == "freeze-skill" || "$STAGE" == "aggregate" ]]; then
  WAIT_ARGS+=(--wait)
fi

exec systemd-run --user \
  --no-ask-password \
  "${WAIT_ARGS[@]}" \
  --collect \
  --unit="$UNIT" \
  --property=Type=exec \
  --property="EnvironmentFile=$ENV_FILE" \
  /usr/bin/sg docker -c "$WRAPPER_COMMAND"
