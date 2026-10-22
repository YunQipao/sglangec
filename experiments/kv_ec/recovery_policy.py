"""M7 recovery policy: pure decision function, no side effects.

Input is checkpoint-commit state plus identity evidence. Output is one of
EC_RECOVERY / PREFILL_RETRY / FAIL_CLOSED and a machine-readable reason.
The policy never inspects KV bytes and never trusts uncommitted parity.
"""

from dataclasses import dataclass, field


EC_RECOVERY, PREFILL_RETRY, FAIL_CLOSED = "EC_RECOVERY", "PREFILL_RETRY", "FAIL_CLOSED"


@dataclass(frozen=True)
class RecoveryDecision:
    action: str
    reason: str
    use_parity: bool = False
    evidence: dict = field(default_factory=dict)


def decide_recovery(checkpoint_state, *, request_cancelled, fault_ranks,
                    token_count, mapping_current, mapping_matches_checkpoint,
                    manager_epoch_matches, rank_preparation_unanimous,
                    all_ranks_decode_observed, committed_parity_available,
                    duplicate_fault=False):
    """Pure mapping from (commit state, identity evidence) to a decision."""
    if duplicate_fault:
        return RecoveryDecision(FAIL_CLOSED, "duplicate_recovery_transaction")
    if request_cancelled:
        return RecoveryDecision(FAIL_CLOSED, "request_cancelled")
    if not manager_epoch_matches:
        return RecoveryDecision(FAIL_CLOSED, "manager_epoch_changed")
    if not mapping_current or not mapping_matches_checkpoint:
        return RecoveryDecision(FAIL_CLOSED, "row_slot_generation_changed")
    if not isinstance(fault_ranks, (list, tuple)) or not fault_ranks:
        return RecoveryDecision(FAIL_CLOSED, "no_fault_rank_declared")
    if len(fault_ranks) != 1 or not all(isinstance(rank, int) and 0 <= rank < 4 for rank in fault_ranks):
        return RecoveryDecision(FAIL_CLOSED, "multi_or_invalid_fault_rank")
    if token_count != 256:
        return RecoveryDecision(FAIL_CLOSED, "unsupported_token_range")
    if not all_ranks_decode_observed:
        return RecoveryDecision(PREFILL_RETRY, "decode_not_yet_observed")
    if checkpoint_state == "COMMITTED":
        if not rank_preparation_unanimous:
            return RecoveryDecision(FAIL_CLOSED, "rank_preparation_vote_rejected")
        if not committed_parity_available:
            return RecoveryDecision(FAIL_CLOSED, "committed_parity_unavailable")
        return RecoveryDecision(EC_RECOVERY, "committed_parity_ready", use_parity=True)
    if checkpoint_state in ("UNPROTECTED", "SNAPSHOT_READY", "COMMITTING"):
        # Uncommitted parity must never be used. The snapshot bytes are also
        # not an independent oracle (they were captured before the fault).
        if not rank_preparation_unanimous:
            return RecoveryDecision(FAIL_CLOSED, "rank_preparation_vote_rejected")
        return RecoveryDecision(PREFILL_RETRY, f"parity_not_committed_{checkpoint_state.lower()}")
    if checkpoint_state == "FAILED":
        if not rank_preparation_unanimous:
            return RecoveryDecision(FAIL_CLOSED, "rank_preparation_vote_rejected")
        return RecoveryDecision(PREFILL_RETRY, "checkpoint_failed")
    return RecoveryDecision(FAIL_CLOSED, f"unknown_checkpoint_state_{checkpoint_state}")
