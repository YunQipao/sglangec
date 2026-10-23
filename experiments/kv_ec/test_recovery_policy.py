"""M7 recovery policy: pure-decision CPU tests, no CUDA, no collectives."""

import unittest

from experiments.kv_ec.recovery_policy import (
    EC_RECOVERY, FAIL_CLOSED, PREFILL_RETRY, decide_recovery,
)


def decide(state, **overrides):
    kwargs = dict(
        request_cancelled=False, fault_ranks=[0], token_count=256,
        mapping_current=True, mapping_matches_checkpoint=True,
        manager_epoch_matches=True, rank_preparation_unanimous=True,
        all_ranks_decode_observed=True, committed_parity_available=False)
    kwargs.update(overrides)
    return decide_recovery(state, **kwargs)


class RecoveryPolicyTests(unittest.TestCase):
    def test_four_protection_states(self):
        self.assertEqual(decide("COMMITTED",
                                committed_parity_available=True).action, EC_RECOVERY)
        for state in ("UNPROTECTED", "SNAPSHOT_READY", "COMMITTING", "FAILED"):
            decision = decide(state)
            self.assertEqual(decision.action, PREFILL_RETRY, state)
            self.assertFalse(decision.use_parity, state)

    def test_uncommitted_parity_never_used_even_if_available(self):
        # committed_parity_available only matters in the COMMITTED branch.
        decision = decide("COMMITTING", committed_parity_available=True)
        self.assertEqual(decision.action, PREFILL_RETRY)
        self.assertFalse(decision.use_parity)

    def test_epoch_change_fails_closed(self):
        decision = decide("COMMITTED", manager_epoch_matches=False,
                          committed_parity_available=True)
        self.assertEqual(decision.action, FAIL_CLOSED)
        self.assertIn("epoch", decision.reason)

    def test_row_slot_generation_change_fails_closed(self):
        for kwargs in (dict(mapping_current=False),
                       dict(mapping_matches_checkpoint=False)):
            decision = decide("COMMITTED", committed_parity_available=True, **kwargs)
            self.assertEqual(decision.action, FAIL_CLOSED)
            self.assertIn("generation", decision.reason)

    def test_cancelled_request_fails_closed(self):
        decision = decide("COMMITTED", request_cancelled=True,
                          committed_parity_available=True)
        self.assertEqual(decision.action, FAIL_CLOSED)

    def test_rank_preparation_divergence_fails_closed(self):
        decision = decide("COMMITTED", rank_preparation_unanimous=False,
                          committed_parity_available=True)
        self.assertEqual(decision.action, FAIL_CLOSED)

    def test_multi_or_missing_fault_rank_fails_closed(self):
        for ranks in ([], [0, 1], [4], [-1]):
            decision = decide("COMMITTED", fault_ranks=ranks,
                              committed_parity_available=True)
            self.assertEqual(decision.action, FAIL_CLOSED)

    def test_unsupported_token_range_fails_closed(self):
        self.assertEqual(decide("COMMITTED", token_count=512,
                                committed_parity_available=True).action, FAIL_CLOSED)

    def test_duplicate_fault_fails_closed(self):
        decision = decide("COMMITTED", duplicate_fault=True,
                          committed_parity_available=True)
        self.assertEqual(decision.action, FAIL_CLOSED)
        self.assertIn("duplicate", decision.reason)

    def test_unknown_state_fails_closed(self):
        self.assertEqual(decide("SOMETHING_ELSE").action, FAIL_CLOSED)

    def test_decode_not_observed_prefers_prefill_retry(self):
        decision = decide("COMMITTED", all_ranks_decode_observed=False,
                          committed_parity_available=True)
        self.assertEqual(decision.action, PREFILL_RETRY)


if __name__ == "__main__":
    unittest.main()
