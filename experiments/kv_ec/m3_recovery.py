"""Strict CPU-side evidence validation for M3 online recovery plans."""

from __future__ import annotations

from dataclasses import dataclass

from .recovery_lease import LeaseRejected


CHUNK_TOKENS = 256
SUPPORTED_RECOVERY_TOKENS = (256, 512, 768)
TP_SIZE = 4


def recovery_ranges(recover_tokens: int) -> tuple[tuple[int, int], ...]:
    if type(recover_tokens) is not int or recover_tokens not in SUPPORTED_RECOVERY_TOKENS:
        raise ValueError("recover_tokens_must_be_256_512_or_768")
    return tuple((start, CHUNK_TOKENS) for start in range(0, recover_tokens, CHUNK_TOKENS))


@dataclass(frozen=True)
class ChunkRecoveryProof:
    snapshot_id: str
    checkpoint_version: str
    manager_epoch: str
    token_start: int
    token_count: int
    lost_rank: int
    committed_ranks: frozenset[int]
    survivor_ranks_used: frozenset[int]
    parity_committed_before_fault: bool
    parity_version_valid: bool
    request_generation_unchanged: bool
    slot_generations_unchanged: bool
    layer_count: int
    kv_writeback_byte_exact: bool
    validation_byte_exact: bool


class M3RecoveryPlan:
    """Accept only every requested, independently versioned 256-token chunk."""

    def __init__(self, *, transaction_id: str, recover_tokens: int, lost_rank: int,
                 manager_epoch: str, layer_count: int = 36):
        self.ranges = recovery_ranges(recover_tokens)
        if not transaction_id or not manager_epoch:
            raise ValueError("transaction_and_manager_epoch_required")
        if type(lost_rank) is not int or lost_rank not in range(TP_SIZE):
            raise ValueError("lost_rank_must_be_0_1_2_or_3")
        if type(layer_count) is not int or layer_count <= 0:
            raise ValueError("invalid_layer_count")
        self.transaction_id = transaction_id
        self.recover_tokens = recover_tokens
        self.lost_rank = lost_rank
        self.manager_epoch = manager_epoch
        self.layer_count = layer_count
        self._proofs: dict[int, ChunkRecoveryProof] = {}
        self.failed = False

    def add(self, proof: ChunkRecoveryProof) -> str:
        if self.failed:
            raise LeaseRejected("recovery_plan_failed")
        try:
            self._validate(proof)
        except (TypeError, ValueError, LeaseRejected):
            self.failed = True
            raise
        if proof.token_start in self._proofs:
            self.failed = True
            raise LeaseRejected("duplicate_recovery_chunk")
        self._proofs[proof.token_start] = proof
        return "COMPLETE" if self.complete else "PENDING"

    def _validate(self, proof: ChunkRecoveryProof) -> None:
        expected = dict(
            snapshot_id=f"{self.transaction_id}-chunk{proof.token_start // CHUNK_TOKENS}",
            checkpoint_version=f"{self.transaction_id}-chunk{proof.token_start // CHUNK_TOKENS}",
            manager_epoch=self.manager_epoch,
            token_count=CHUNK_TOKENS,
            lost_rank=self.lost_rank,
            committed_ranks=frozenset(range(TP_SIZE)),
            survivor_ranks_used=frozenset(set(range(TP_SIZE)) - {self.lost_rank}),
            parity_committed_before_fault=True,
            parity_version_valid=True,
            request_generation_unchanged=True,
            slot_generations_unchanged=True,
            layer_count=self.layer_count,
            kv_writeback_byte_exact=True,
            validation_byte_exact=True,
        )
        if proof.token_start not in {start for start, _ in self.ranges}:
            raise LeaseRejected("recovery_chunk_out_of_requested_range")
        for key, value in expected.items():
            if getattr(proof, key) != value:
                raise LeaseRejected(f"recovery_chunk_{key}_mismatch")

    @property
    def complete(self) -> bool:
        return not self.failed and set(self._proofs) == {start for start, _ in self.ranges}

    def authorize_resume(self) -> dict:
        if not self.complete:
            raise LeaseRejected("resume_requires_all_chunks_byte_exact")
        ordered = [self._proofs[start] for start, _ in self.ranges]
        return {
            "state": "RECOVERY_COMPLETE",
            "lost_rank": self.lost_rank,
            "recover_tokens": self.recover_tokens,
            "chunk_tokens": CHUNK_TOKENS,
            "chunks": [
                {
                    "snapshot_id": proof.snapshot_id,
                    "checkpoint_version": proof.checkpoint_version,
                    "token_start": proof.token_start,
                    "token_count": proof.token_count,
                    "lost_rank": proof.lost_rank,
                    "survivor_ranks_used": sorted(proof.survivor_ranks_used),
                    "layer_count": proof.layer_count,
                    "kv_writeback_byte_exact": proof.kv_writeback_byte_exact,
                    "validation_byte_exact": proof.validation_byte_exact,
                }
                for proof in ordered
            ],
        }
