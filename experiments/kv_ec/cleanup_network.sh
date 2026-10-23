#!/usr/bin/env bash
set -euo pipefail
NS_A=kvectp-a
NS_B=kvectp-b
VETH_A=kvecta
VETH_B=kvectb
APPLY=0
[[ "${1:-}" == "--apply" ]] && APPLY=1
exists_a=0 exists_b=0
sudo ip netns list | awk '{print $1}' | grep -Fxq "$NS_A" && exists_a=1 || true
sudo ip netns list | awk '{print $1}' | grep -Fxq "$NS_B" && exists_b=1 || true
printf 'Planned cleanup, limited to named experiment namespaces and their contained veths:\n'
(( exists_a )) && echo "  sudo ip netns del $NS_A"
(( exists_b )) && echo "  sudo ip netns del $NS_B"
if (( ! exists_a && ! exists_b )); then echo "Nothing to clean."; exit 0; fi
if (( exists_a )) && ! sudo ip -n "$NS_A" -d link show dev "$VETH_A" 2>/dev/null | grep -Fq 'alias sglangec-two-logical-nodes-v1'; then
  echo "REFUSED: $NS_A/$VETH_A lacks this experiment's ownership alias." >&2; exit 2
fi
if (( exists_b )) && ! sudo ip -n "$NS_B" -d link show dev "$VETH_B" 2>/dev/null | grep -Fq 'alias sglangec-two-logical-nodes-v1'; then
  echo "REFUSED: $NS_B/$VETH_B lacks this experiment's ownership alias." >&2; exit 2
fi
if (( ! APPLY )); then echo "PLAN ONLY. Review, then run: sudo bash $0 --apply"; exit 0; fi
echo "This will delete only $NS_A/$NS_B namespaces and resources inside them."
read -r -p "Type DELETE-KVECTP-NAMESPACES to execute: " confirmation
[[ "$confirmation" == "DELETE-KVECTP-NAMESPACES" ]] || { echo "Cancelled."; exit 1; }
if (( exists_a )); then sudo ip netns del "$NS_A"; fi
if (( exists_b )); then sudo ip netns del "$NS_B"; fi
echo "Removed only experiment namespaces."
