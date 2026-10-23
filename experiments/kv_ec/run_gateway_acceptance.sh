#!/usr/bin/env bash
set -euo pipefail

# Application-layer topology only. Packet capture/counters on Server G are a
# separate required observation; this script does not claim wire proof.
ROLE="${1:?usage: run_gateway_acceptance.sh receiver|gateway|send|cpu-a-c ...}"
shift
case "$ROLE" in
  receiver)
    exec python -m experiments.kv_ec.gateway_server --role receiver "$@" ;;
  gateway)
    exec python -m experiments.kv_ec.gateway_server --role gateway "$@" ;;
  send)
    exec python -m experiments.kv_ec.gateway_client "$@" ;;
  cpu-a-c)
    exec python -m experiments.kv_ec.gateway_pipeline "$@" ;;
  *) echo "unknown role: $ROLE" >&2; exit 2 ;;
esac
