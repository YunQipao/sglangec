#!/usr/bin/env bash
set -euo pipefail

NODE="${1:?usage: run_logical_node.sh A|B [--diagnostic-socket] [--no-cuda-graph] [--execute]}"
shift
[[ "$NODE" == A || "$NODE" == B ]] || { echo 'Node must be A or B' >&2; exit 2; }
DIAGNOSTIC=0 NO_GRAPH=0 EXECUTE=0
while (($#)); do
  case "$1" in
    --diagnostic-socket) DIAGNOSTIC=1 ;;
    --no-cuda-graph) NO_GRAPH=1 ;;
    --execute) EXECUTE=1 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
MODEL="${SGLANG_MODEL:-/media/sda1/iic_data/luotao/models/Qwen3-8B}"
RUN_ID="${SGLANG_EC_RUN_ID:-logical-tp4-$(date -u +%Y%m%dT%H%M%SZ)}"
PORT="${SGLANG_PORT:-30000}"
DIST_PORT="${SGLANG_DIST_PORT:-29577}"
MEM_FRACTION="${SGLANG_MEM_FRACTION_STATIC:-0.60}"
CONTEXT="${SGLANG_CONTEXT_LENGTH:-4096}"
PYTHON_BIN="${CONDA_PREFIX:+$CONDA_PREFIX/bin/python}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
if [[ "$NODE" == A ]]; then
  NS=kvectp-a DEVICES=0,1 NODE_RANK=0 IFACE=kvecta
else
  NS=kvectp-b DEVICES=2,3 NODE_RANK=1 IFACE=kvectb
fi
[[ -d "$MODEL" ]] || { echo "Model directory not found: $MODEL" >&2; exit 2; }
mkdir -p "$ROOT/experiments/kv_ec/results"
OUT="$ROOT/experiments/kv_ec/results/$RUN_ID-node${NODE_RANK}"
mkdir -m 700 "$OUT" 2>/dev/null || { echo "Refusing existing evidence path: $OUT" >&2; exit 2; }

export ROOT MODEL RUN_ID NODE_RANK DIAGNOSTIC
python - <<'PY' > "$OUT/rank-layout.json"
import json, os, sys
sys.path.insert(0, os.environ["ROOT"])
from experiments.kv_ec.logical_nodes import rank_manifest
print(json.dumps(rank_manifest(run_id=os.environ["RUN_ID"], model_path=os.environ["MODEL"],
    diagnostic=os.environ.get("DIAGNOSTIC", "0") == "1"), indent=2))
PY

CMD=(sudo -E ip netns exec "$NS" env
  "PATH=$PATH" "PYTHONPATH=$ROOT:$ROOT/python${PYTHONPATH:+:$PYTHONPATH}"
  "CUDA_VISIBLE_DEVICES=$DEVICES" "CUDA_DEVICE_ORDER=PCI_BUS_ID"
  "NCCL_SOCKET_IFNAME=${IFACE}" "GLOO_SOCKET_IFNAME=${IFACE}"
  "NCCL_DEBUG=INFO" "NCCL_DEBUG_SUBSYS=INIT,NET,ENV"
  "SGLANG_EC_RUN_ID=$RUN_ID" "SGLANG_EC_NODE_RANK=$NODE_RANK")
if (( DIAGNOSTIC )); then
  CMD+=("NCCL_P2P_DISABLE=1" "NCCL_SHM_DISABLE=1")
fi
CMD+=("$PYTHON_BIN" -m sglang.launch_server
  --model-path "$MODEL" --tp-size 4 --nnodes 2 --node-rank "$NODE_RANK"
  --dist-init-addr "10.233.77.1:$DIST_PORT" --host 0.0.0.0 --port "$PORT"
  --mem-fraction-static "$MEM_FRACTION" --context-length "$CONTEXT")
if (( DIAGNOSTIC )); then CMD+=(--disable-custom-all-reduce); fi
if (( NO_GRAPH )); then CMD+=(--disable-cuda-graph); fi
printf 'Node %s planned command (rank map file: %s):\n  ' "$NODE" "$OUT/rank-layout.json"
printf '%q ' "${CMD[@]}"
printf '\nLog: %s/server.log\n' "$OUT"
if (( ! EXECUTE )); then echo 'PLAN ONLY. Review command, then rerun with --execute.'; exit 0; fi
echo "This starts this experiment's SGLang worker in namespace $NS on GPUs $DEVICES; it will not stop existing tasks."
read -r -p "Type START-SGLANG-LOGICAL-NODE-$NODE to execute: " confirmation
[[ "$confirmation" == "START-SGLANG-LOGICAL-NODE-$NODE" ]] || { echo 'Cancelled.'; exit 1; }
printf '%q ' "${CMD[@]}" > "$OUT/command.txt"; printf '\n' >> "$OUT/command.txt"
"${CMD[@]}" 2>&1 | tee "$OUT/server.log"
