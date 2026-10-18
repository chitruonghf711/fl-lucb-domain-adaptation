"""Deterministic, non-replacement minibatch planning."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
from random import Random
from typing import Any, Dict, Hashable, Iterable, Mapping, Sequence, Tuple

from .models import Label, json_safe, label_key


def _identity_key(value: Any) -> str:
    """Return a type-aware identifier suitable for duplicate checks."""

    return f"{type(value).__name__}:{value!r}"


def _client_seed(seed: int, client_id: str) -> int:
    """Derive a stable client-specific seed without Python's randomized hash."""

    encoded = f"{seed}:{client_id}".encode("utf-8")
    return int.from_bytes(sha256(encoded).digest()[:8], byteorder="big")


@dataclass(frozen=True)
class SampleRef:
    """A sample identifier and its label, without input content."""

    sample_id: Any
    label: Label


@dataclass(frozen=True)
class Minibatch:
    """One deterministic non-replacement minibatch for one client."""

    client_id: str
    batch_index: int
    samples: Tuple[SampleRef, ...]

    @property
    def sample_ids(self) -> Tuple[Any, ...]:
        return tuple(sample.sample_id for sample in self.samples)

    @property
    def labels(self) -> Tuple[Label, ...]:
        return tuple(sample.label for sample in self.samples)

    @property
    def size(self) -> int:
        return len(self.samples)

    def to_dict(self) -> Dict[str, Any]:
        """Return non-sensitive batch metadata for debugging or audit coverage."""

        counts = Counter(label_key(label) for label in self.labels)
        return {
            "client_id": self.client_id,
            "batch_index": self.batch_index,
            "sample_ids": [json_safe(sample_id) for sample_id in self.sample_ids],
            "sample_count": self.size,
            "label_counts": dict(sorted(counts.items())),
        }


@dataclass(frozen=True)
class MinibatchPlan:
    """A complete finite schedule of client minibatches.

    The plan contains every provided sample exactly once. Its schedule is
    deterministic for a given client ID, labels, IDs, batch size, and seed.
    The simple round-robin ordering is best-effort stratified: it distributes
    currently available classes before consuming another item from a class, but
    it never duplicates data to force a rare class into every batch.
    """

    client_id: str
    batches: Tuple[Minibatch, ...]
    batch_size: int
    seed: int
    fingerprint: str

    @classmethod
    def build(
        cls,
        client_id: str,
        labels: Sequence[Label],
        batch_size: int = 64,
        seed: int = 42,
        sample_ids: Sequence[Any] | None = None,
    ) -> "MinibatchPlan":
        """Create a deterministic best-effort stratified plan.

        ``sample_ids`` must identify rows within a client and defaults to
        zero-based input positions. IDs and labels are the only data accepted;
        model features never enter a plan.
        """

        if not isinstance(client_id, str) or not client_id:
            raise ValueError("client_id must be a non-empty string")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if sample_ids is None:
            sample_ids = tuple(range(len(labels)))
        if len(labels) != len(sample_ids):
            raise ValueError("labels and sample_ids must have the same length")

        seen_ids: set[str] = set()
        pools: Dict[Label, list[SampleRef]] = defaultdict(list)
        for sample_id, label in zip(sample_ids, labels):
            try:
                hash(label)
            except TypeError as exc:
                raise ValueError("labels must be hashable") from exc
            identity = _identity_key(sample_id)
            if identity in seen_ids:
                raise ValueError("sample_ids must be unique within a client plan")
            seen_ids.add(identity)
            pools[label].append(SampleRef(sample_id=sample_id, label=label))

        rng = Random(_client_seed(seed, client_id))
        class_order = sorted(pools, key=label_key)
        for label in class_order:
            rng.shuffle(pools[label])
        rng.shuffle(class_order)

        schedule: list[SampleRef] = []
        active = list(class_order)
        while active:
            next_active: list[Label] = []
            for label in active:
                pool = pools[label]
                if pool:
                    schedule.append(pool.pop())
                if pool:
                    next_active.append(label)
            active = next_active

        batches = tuple(
            Minibatch(
                client_id=client_id,
                batch_index=batch_index,
                samples=tuple(schedule[start : start + batch_size]),
            )
            for batch_index, start in enumerate(range(0, len(schedule), batch_size))
        )
        fingerprint_source = {
            "client_id": client_id,
            "batch_size": batch_size,
            "seed": seed,
            "batches": [batch.to_dict() for batch in batches],
        }
        fingerprint = sha256(repr(fingerprint_source).encode("utf-8")).hexdigest()
        return cls(
            client_id=client_id,
            batches=batches,
            batch_size=batch_size,
            seed=seed,
            fingerprint=fingerprint,
        )

    @property
    def total_samples(self) -> int:
        return sum(batch.size for batch in self.batches)

    @property
    def sample_ids(self) -> Tuple[Any, ...]:
        return tuple(sample_id for batch in self.batches for sample_id in batch.sample_ids)

    def next_unscored(self, scored_batch_indices: Iterable[int]) -> Minibatch | None:
        """Return the first planned batch not already represented in evidence."""

        scored = set(scored_batch_indices)
        return next((batch for batch in self.batches if batch.batch_index not in scored), None)

    def coverage_summary(self) -> Dict[str, Any]:
        """Return class and batch coverage without input content."""

        total_counts = Counter()
        per_batch = []
        for batch in self.batches:
            counts = Counter(label_key(label) for label in batch.labels)
            total_counts.update(counts)
            per_batch.append(
                {
                    "batch_index": batch.batch_index,
                    "sample_count": batch.size,
                    "label_counts": dict(sorted(counts.items())),
                }
            )
        return {
            "total_samples": self.total_samples,
            "batch_count": len(self.batches),
            "unique_sample_count": len(self.sample_ids),
            "class_counts": dict(sorted(total_counts.items())),
            "batches": per_batch,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Return an audit-safe serialized plan."""

        return {
            "client_id": self.client_id,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "fingerprint": self.fingerprint,
            "coverage": self.coverage_summary(),
        }


def build_minibatch_plans(
    labels_by_client: Mapping[str, Sequence[Label]],
    batch_size: int = 64,
    seed: int = 42,
    sample_ids_by_client: Mapping[str, Sequence[Any]] | None = None,
) -> Dict[str, MinibatchPlan]:
    """Build one deterministic plan per client from aligned label collections."""

    plans: Dict[str, MinibatchPlan] = {}
    for client_id in sorted(labels_by_client):
        sample_ids = None if sample_ids_by_client is None else sample_ids_by_client.get(client_id)
        plans[client_id] = MinibatchPlan.build(
            client_id=client_id,
            labels=labels_by_client[client_id],
            sample_ids=sample_ids,
            batch_size=batch_size,
            seed=seed,
        )
    return plans
