"""CPU fail-closed coordination model for four local Scheduler leases.

This module deliberately performs no distributed collectives and no GPU
access.  It defines the protocol that a future IPC/Coordinator integration
must implement without allowing one rank to release while another rank is
still preparing, copying, or validating.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import time
from typing import Callable, Iterable

from .recovery_lease import LeaseRejected, RankLeaseObservation


class CrossRankLeaseState(str, Enum):
    PREPARE = "PREPARE"
    LOCAL_LEASED = "LOCAL_LEASED"
    ALL_RANKS_READY = "ALL_RANKS_READY"
    CUDA_COMPLETED = "CUDA_COMPLETED"
    ALL_RANKS_VALIDATED = "ALL_RANKS_VALIDATED"
    GLOBAL_RELEASED = "GLOBAL_RELEASED"
    FAILED = "FAILED"
    EXPIRED = "EXPIRED"


@dataclass(frozen=True)
class CrossRankStatus:
    transaction_id: str
    state: CrossRankLeaseState
    tp_size: int
    prepared_ranks: tuple[int, ...]
    leased_ranks: tuple[int, ...]
    cuda_completed_ranks: tuple[int, ...]
    validated_ranks: tuple[int, ...]
    request_id: str
    request_version: str
    checkpoint_version: str
    manager_epoch: str
    token_start: int
    token_count: int
    failure_reason: str | None = None


@dataclass
class _Transaction:
    transaction_id: str
    observations: dict[int, RankLeaseObservation]
    state: CrossRankLeaseState
    deadline: float
    leased: set[int]
    cuda_completed: set[int]
    validated: set[int]
    failure_reason: str | None = None
    abort_acks: set[int] | None = None


class CrossRankLeaseCoordinator:
    """A deterministic TP=4 state machine with no implicit release."""

    def __init__(
        self,
        *,
        timeout_s: float = 30.0,
        manager_epoch: str,
        clock: Callable[[], float] = time.monotonic,
    ):
        if timeout_s <= 0 or not manager_epoch:
            raise ValueError("invalid coordinator configuration")
        self.timeout_s = float(timeout_s)
        self.manager_epoch = manager_epoch
        self._clock = clock
        self._transactions: dict[str, _Transaction] = {}
        self._slot_owner: dict[tuple[int, int], str] = {}

    def prepare(
        self,
        transaction_id: str,
        observations: Iterable[RankLeaseObservation],
        *,
        checkpoint_version: str,
    ) -> CrossRankStatus:
        if transaction_id in self._transactions:
            raise LeaseRejected("duplicate_transaction")
        obs = sorted(list(observations), key=lambda item: item.tp_rank)
        self._validate_observations(obs, checkpoint_version)
        owners = {
            (item.tp_rank, slot)
            for item in obs
            for slot in item.physical_slots
        }
        if owners & self._slot_owner.keys():
            raise LeaseRejected("slot_already_coordinated")
        txn = _Transaction(
            transaction_id=transaction_id,
            observations={item.tp_rank: item for item in obs},
            state=CrossRankLeaseState.PREPARE,
            deadline=self._clock() + self.timeout_s,
            leased=set(),
            cuda_completed=set(),
            validated=set(),
            abort_acks=set(),
        )
        self._transactions[transaction_id] = txn
        for owner in owners:
            self._slot_owner[owner] = transaction_id
        return self.status(transaction_id)

    def mark_local_leased(self, transaction_id: str, tp_rank: int) -> CrossRankStatus:
        txn = self._get(transaction_id)
        if txn.state not in (
            CrossRankLeaseState.PREPARE,
            CrossRankLeaseState.LOCAL_LEASED,
        ):
            raise LeaseRejected(
                f"invalid_transition:{txn.state.value}->LOCAL_LEASED"
            )
        self._require_rank(txn, tp_rank)
        txn.leased.add(tp_rank)
        if len(txn.leased) == txn.observations[tp_rank].tp_size:
            txn.state = CrossRankLeaseState.ALL_RANKS_READY
        else:
            txn.state = CrossRankLeaseState.LOCAL_LEASED
        return self.status(transaction_id)

    def mark_cuda_completed(self, transaction_id: str, tp_rank: int) -> CrossRankStatus:
        txn = self._get(transaction_id)
        self._require_state(txn, CrossRankLeaseState.ALL_RANKS_READY)
        self._require_rank(txn, tp_rank)
        if tp_rank not in txn.leased:
            raise LeaseRejected("rank_not_locally_leased")
        txn.cuda_completed.add(tp_rank)
        if len(txn.cuda_completed) == txn.observations[tp_rank].tp_size:
            txn.state = CrossRankLeaseState.CUDA_COMPLETED
        return self.status(transaction_id)

    def mark_validated(self, transaction_id: str, tp_rank: int) -> CrossRankStatus:
        txn = self._get(transaction_id)
        self._require_state(txn, CrossRankLeaseState.CUDA_COMPLETED)
        self._require_rank(txn, tp_rank)
        if tp_rank not in txn.cuda_completed:
            raise LeaseRejected("rank_cuda_incomplete")
        txn.validated.add(tp_rank)
        if len(txn.validated) == txn.observations[tp_rank].tp_size:
            txn.state = CrossRankLeaseState.ALL_RANKS_VALIDATED
        return self.status(transaction_id)

    def release_global(self, transaction_id: str) -> CrossRankStatus:
        txn = self._get(transaction_id)
        self._require_state(txn, CrossRankLeaseState.ALL_RANKS_VALIDATED)
        txn.state = CrossRankLeaseState.GLOBAL_RELEASED
        self._release_owners(txn)
        return self.status(transaction_id)

    def fail(self, transaction_id: str, reason: str) -> CrossRankStatus:
        txn = self._get(transaction_id)
        if txn.state in (
            CrossRankLeaseState.GLOBAL_RELEASED,
            CrossRankLeaseState.FAILED,
            CrossRankLeaseState.EXPIRED,
        ):
            raise LeaseRejected("transaction_terminal")
        txn.state = CrossRankLeaseState.FAILED
        txn.failure_reason = reason or "rank_failure"
        return self.status(transaction_id)

    def acknowledge_abort(self, transaction_id: str, tp_rank: int) -> CrossRankStatus:
        txn = self._get(transaction_id)
        if txn.state not in (CrossRankLeaseState.FAILED, CrossRankLeaseState.EXPIRED):
            raise LeaseRejected("abort_ack_requires_failure")
        self._require_rank(txn, tp_rank)
        txn.abort_acks.add(tp_rank)
        if txn.abort_acks == set(txn.observations):
            self._release_owners(txn)
        return self.status(transaction_id)

    def status(self, transaction_id: str) -> CrossRankStatus:
        txn = self._get(transaction_id)
        first = next(iter(txn.observations.values()))
        return CrossRankStatus(
            transaction_id=transaction_id,
            state=txn.state,
            tp_size=first.tp_size,
            prepared_ranks=tuple(sorted(txn.observations)),
            leased_ranks=tuple(sorted(txn.leased)),
            cuda_completed_ranks=tuple(sorted(txn.cuda_completed)),
            validated_ranks=tuple(sorted(txn.validated)),
            request_id=first.request_id,
            request_version=first.request_version,
            checkpoint_version=first.checkpoint_version,
            manager_epoch=first.manager_epoch,
            token_start=first.token_start,
            token_count=first.token_count,
            failure_reason=txn.failure_reason,
        )

    def _get(self, transaction_id: str) -> _Transaction:
        txn = self._transactions.get(transaction_id)
        if txn is None:
            raise LeaseRejected("unknown_transaction")
        if txn.state in (
            CrossRankLeaseState.PREPARE,
            CrossRankLeaseState.LOCAL_LEASED,
            CrossRankLeaseState.ALL_RANKS_READY,
            CrossRankLeaseState.CUDA_COMPLETED,
            CrossRankLeaseState.ALL_RANKS_VALIDATED,
        ) and self._clock() >= txn.deadline:
            txn.state = CrossRankLeaseState.EXPIRED
            txn.failure_reason = "transaction_timeout"
        return txn

    @staticmethod
    def _require_state(txn: _Transaction, expected: CrossRankLeaseState) -> None:
        if txn.state != expected:
            raise LeaseRejected(
                f"invalid_transition:{txn.state.value}->{expected.value}"
            )

    @staticmethod
    def _require_rank(txn: _Transaction, tp_rank: int) -> None:
        if tp_rank not in txn.observations:
            raise LeaseRejected("invalid_tp_rank")

    def _validate_observations(
        self,
        observations: list[RankLeaseObservation],
        checkpoint_version: str,
    ) -> None:
        if len(observations) != 4 or {item.tp_rank for item in observations} != {0, 1, 2, 3}:
            raise LeaseRejected("incomplete_tp4_rank_set")
        first = observations[0]
        if first.tp_size != 4 or first.manager_epoch != self.manager_epoch:
            raise LeaseRejected("manager_epoch_mismatch")
        common = (
            first.request_id,
            first.request_version,
            first.checkpoint_version,
            first.manager_epoch,
            first.token_start,
            first.token_count,
        )
        if first.checkpoint_version != checkpoint_version:
            raise LeaseRejected("checkpoint_version_mismatch")
        for item in observations:
            if item.tp_size != 4 or (
                item.request_id,
                item.request_version,
                item.checkpoint_version,
                item.manager_epoch,
                item.token_start,
                item.token_count,
            ) != common:
                raise LeaseRejected("rank_metadata_mismatch")
            if len(item.physical_slots) != len(item.slot_generations):
                raise LeaseRejected("slot_generation_length_mismatch")
            if any(generation <= 0 for generation in item.slot_generations):
                raise LeaseRejected("invalid_slot_generation")
            if item.token_count <= 0:
                raise LeaseRejected("invalid_token_range")

    def _release_owners(self, txn: _Transaction) -> None:
        for item in txn.observations.values():
            for slot in item.physical_slots:
                self._slot_owner.pop((item.tp_rank, slot), None)
