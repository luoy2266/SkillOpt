#!/usr/bin/env bash
set -euo pipefail

ENV_FILE="${SKILLOPT_FORMAL_ENV_FILE:-/home/yunl/.config/skillopt/terminalbench-formal.env}"
WRAPPER="/home/yunl/projects/SkillOpt/scripts/run_terminalbench_formal_stage.sh"
UNIT="skillopt-tbench-formal-preflight-probe"

if [[ ! -r "$ENV_FILE" ]]; then
  echo "ENVIRONMENT_FILE=MISSING" >&2
  exit 2
fi

exec systemd-run --user \
  --no-ask-password \
  --wait \
  --collect \
  --unit="$UNIT" \
  --property=Type=exec \
  --property="EnvironmentFile=$ENV_FILE" \
  /usr/bin/sg docker -c \
  "SKILLOPT_FORMAL_DOCKER_MODE=sg exec $WRAPPER probe"
