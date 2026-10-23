#!/usr/bin/env bash
set -euo pipefail

# Scoped topology only: two namespaces and a private veth pair. Default is
# PLAN-ONLY. --apply prints every sudo command first and requires typed consent.
NS_A=kvectp-a
NS_B=kvectp-b
VETH_A=kvecta
VETH_B=kvectb
IP_A=10.233.77.1/30
IP_B=10.233.77.2/30
APPLY=0
[[ "${1:-}" == "--apply" ]] && APPLY=1

if sudo ip netns list | awk '{print $1}' | grep -Fxq "$NS_A" ||
   sudo ip netns list | awk '{print $1}' | grep -Fxq "$NS_B"; then
  echo "REFUSED: one of the experiment namespace names already exists; nothing changed." >&2
  exit 2
fi
if sudo ip link show "$VETH_A" >/dev/null 2>&1 || sudo ip link show "$VETH_B" >/dev/null 2>&1; then
  echo "REFUSED: one of the dedicated veth names already exists; nothing changed." >&2
  exit 2
fi

commands=(
  "ip netns add $NS_A"
  "ip netns add $NS_B"
  "ip link add $VETH_A type veth peer name $VETH_B"
  "ip link set $VETH_A netns $NS_A"
  "ip link set $VETH_B netns $NS_B"
  "ip -n $NS_A link set lo up"
  "ip -n $NS_B link set lo up"
  "ip -n $NS_A link set $VETH_A alias sglangec-two-logical-nodes-v1"
  "ip -n $NS_B link set $VETH_B alias sglangec-two-logical-nodes-v1"
  "ip -n $NS_A addr add $IP_A dev $VETH_A"
  "ip -n $NS_B addr add $IP_B dev $VETH_B"
  "ip -n $NS_A link set $VETH_A up"
  "ip -n $NS_B link set $VETH_B up"
)
printf 'Planned scoped network commands (no physical NIC/default route/Docker/firewall changes):\n'
printf '  sudo %s\n' "${commands[@]}"
if (( ! APPLY )); then
  echo "PLAN ONLY. After review, run: sudo bash $0 --apply"
  exit 0
fi
echo "This creates only namespaces $NS_A/$NS_B and veth $VETH_A/$VETH_B."
read -r -p "Type CREATE-KVECTP-NAMESPACES to execute: " confirmation
[[ "$confirmation" == "CREATE-KVECTP-NAMESPACES" ]] || { echo "Cancelled."; exit 1; }
created_a=0 created_b=0 created_veth=0
rollback() {
  local rc=$?
  if (( rc != 0 )); then
    echo "Setup failed; rollback only resources confirmed created by this invocation."
    (( created_veth )) && sudo ip link del "$VETH_A" 2>/dev/null || true
    (( created_b )) && sudo ip netns del "$NS_B" 2>/dev/null || true
    (( created_a )) && sudo ip netns del "$NS_A" 2>/dev/null || true
  fi
  exit "$rc"
}
trap rollback EXIT
sudo ip netns add "$NS_A"; created_a=1
sudo ip netns add "$NS_B"; created_b=1
sudo ip link add "$VETH_A" type veth peer name "$VETH_B"; created_veth=1
sudo ip link set "$VETH_A" netns "$NS_A"
sudo ip link set "$VETH_B" netns "$NS_B"
sudo ip -n "$NS_A" link set lo up
sudo ip -n "$NS_B" link set lo up
sudo ip -n "$NS_A" link set "$VETH_A" alias sglangec-two-logical-nodes-v1
sudo ip -n "$NS_B" link set "$VETH_B" alias sglangec-two-logical-nodes-v1
sudo ip -n "$NS_A" addr add "$IP_A" dev "$VETH_A"
sudo ip -n "$NS_B" addr add "$IP_B" dev "$VETH_B"
sudo ip -n "$NS_A" link set "$VETH_A" up
sudo ip -n "$NS_B" link set "$VETH_B" up
trap - EXIT
echo "Created private logical-node veth topology. Verify before launching GPU workers."
