#!/usr/bin/env bash
set -euo pipefail

if [ ! -d "externals/llama.cpp" ]; then
  echo "llama.cpp submodule not found. Did you forget to init submodules?" >&2
  exit 1
fi

cd externals/llama.cpp

LLAMA_METAL=1 make -j$(sysctl -n hw.logicalcpu || nproc)
