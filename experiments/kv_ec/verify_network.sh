#!/usr/bin/env bash
set -euo pipefail
NS_A=kvectp-a
NS_B=kvectp-b
APPLY=0
[[ "${1:-}" == "--run" ]] && APPLY=1
if ! ip netns list | awk '{print $1}' | grep -Fxq "$NS_A" ||
   ! ip netns list | awk '{print $1}' | grep -Fxq "$NS_B"; then
  echo "Missing experiment namespace(s); run setup_network.sh after reviewing its plan." >&2
  exit 2
fi
commands=("ip -n $NS_A -brief address show dev kvecta" "ip -n $NS_B -brief address show dev kvectb"
  "ip netns exec $NS_A ping -c 3 -W 2 10.233.77.2" "ip netns exec $NS_B ping -c 3 -W 2 10.233.77.1"
  "ip -n $NS_A route show" "ip -n $NS_B route show")
echo "Read-only verification commands:"; printf '  sudo %s\n' "${commands[@]}"
if (( ! APPLY )); then echo "PLAN ONLY. Review then run: sudo bash $0 --run"; exit 0; fi
read -r -p "Type VERIFY-KVECTP-NETWORK to run: " confirmation
[[ "$confirmation" == "VERIFY-KVECTP-NETWORK" ]] || { echo "Cancelled."; exit 1; }
sudo ip -n "$NS_A" -brief address show dev kvecta
sudo ip -n "$NS_B" -brief address show dev kvectb
sudo ip netns exec "$NS_A" ping -c 3 -W 2 10.233.77.2
sudo ip netns exec "$NS_B" ping -c 3 -W 2 10.233.77.1
echo "Routes scoped to experiment namespaces:"
sudo ip -n "$NS_A" route show
sudo ip -n "$NS_B" route show
