#!/usr/bin/env bash
set -euo pipefail

export AGENT_CONFIG=${AGENT_CONFIG:-"agent/config/agent_config.toml"}

python -m agent.main "$@"
