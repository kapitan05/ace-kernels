"""Measure what autograd keeps alive for the backward pass ("saved tensors")."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import torch


@dataclass(frozen=True)
class SavedTensor:
    shape: tuple[int, ...]
    dtype: torch.dtype
    nbytes: int


@dataclass
class SavedTensorStats:
    """Tensors saved for backward during one forward call.

    ``unique_bytes`` counts each underlying storage once (what is really held in
    memory); ``excluded_bytes`` is the part shared with the tensors passed as
    ``exclude`` (typically the inputs, which the caller keeps alive anyway).
    """

    tensors: list[SavedTensor] = field(default_factory=list)
    unique_bytes: int = 0
    excluded_bytes: int = 0

    @property
    def extra_bytes(self) -> int:
        """Saved bytes beyond the excluded tensors: the loss's own memory cost."""
        return self.unique_bytes - self.excluded_bytes

    def largest(self, n: int = 5) -> list[SavedTensor]:
        return sorted(self.tensors, key=lambda t: t.nbytes, reverse=True)[:n]


def measure_saved_tensors(
    fn: Callable[[], torch.Tensor], exclude: Sequence[torch.Tensor] = ()
) -> tuple[torch.Tensor, SavedTensorStats]:
    """Run ``fn`` and record every tensor autograd saves for backward."""
    stats = SavedTensorStats()
    seen: set[int] = set()
    excluded = {t.untyped_storage().data_ptr() for t in exclude}

    def pack(t: torch.Tensor) -> torch.Tensor:
        storage = t.untyped_storage()
        nbytes = storage.nbytes()
        stats.tensors.append(SavedTensor(tuple(t.shape), t.dtype, nbytes))
        ptr = storage.data_ptr()
        if ptr not in seen:
            seen.add(ptr)
            stats.unique_bytes += nbytes
            if ptr in excluded:
                stats.excluded_bytes += nbytes
        return t

    def unpack(t: torch.Tensor) -> torch.Tensor:
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        out = fn()
    return out, stats
