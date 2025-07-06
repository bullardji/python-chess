"""GPU-accelerated helpers for :mod:`chess`.

This module provides :class:`GPUBoard`, a subclass of
:class:`~chess.Board` that can optionally perform certain computations on
a CUDA-capable GPU via :mod:`cupy`. If :mod:`cupy` is not available, the
methods transparently fall back to their CPU counterparts.
"""

from __future__ import annotations

import typing

import logging

import chess

try:
    import cupy as cp  # type: ignore
    try:
        GPU_AVAILABLE: bool = bool(cp.cuda.runtime.getDeviceCount())
    except Exception:
        logging.getLogger(__name__).debug("GPU check failed", exc_info=True)
        GPU_AVAILABLE = False
except Exception:  # pragma: no cover - cupy is optional
    cp = None  # type: ignore
    GPU_AVAILABLE = False

__all__ = [
    "GPUBoard",
    "GPU_AVAILABLE",
    "is_gpu_available",
    "GPU_BB_RANK_MASKS",
    "GPU_BB_FILE_MASKS",
    "GPU_BB_DIAG_MASKS",
    "GPU_BB_KING_ATTACKS",
    "GPU_BB_KNIGHT_ATTACKS",
    "GPU_BB_PAWN_ATTACKS",
]

if GPU_AVAILABLE:
    GPU_BB_RANK_MASKS = cp.asarray(chess.BB_RANK_MASKS, dtype=cp.uint64)
    GPU_BB_FILE_MASKS = cp.asarray(chess.BB_FILE_MASKS, dtype=cp.uint64)
    GPU_BB_DIAG_MASKS = cp.asarray(chess.BB_DIAG_MASKS, dtype=cp.uint64)
    GPU_BB_KING_ATTACKS = cp.asarray(chess.BB_KING_ATTACKS, dtype=cp.uint64)
    GPU_BB_KNIGHT_ATTACKS = cp.asarray(chess.BB_KNIGHT_ATTACKS, dtype=cp.uint64)
    GPU_BB_PAWN_ATTACKS = cp.asarray(chess.BB_PAWN_ATTACKS, dtype=cp.uint64)
else:  # pragma: no cover - GPU not available
    GPU_BB_RANK_MASKS = None  # type: ignore
    GPU_BB_FILE_MASKS = None  # type: ignore
    GPU_BB_DIAG_MASKS = None  # type: ignore
    GPU_BB_KING_ATTACKS = None  # type: ignore
    GPU_BB_KNIGHT_ATTACKS = None  # type: ignore
    GPU_BB_PAWN_ATTACKS = None  # type: ignore

def is_gpu_available() -> bool:
    """Returns ``True`` if :mod:`cupy` is installed and a GPU is detected."""
    return GPU_AVAILABLE

class GPUBoard(chess.Board):
    """A board with optional GPU-accelerated methods."""

    def attackers_mask(
        self,
        color: chess.Color,
        square: chess.Square,
        occupied: typing.Optional[chess.Bitboard] = None,
    ) -> chess.Bitboard:
        """Like :meth:`chess.Board.attackers_mask` but uses the GPU when possible."""
        if not GPU_AVAILABLE:
            return super().attackers_mask(color, square, occupied)

        occ = self.occupied if occupied is None else occupied
        rank_pieces = GPU_BB_RANK_MASKS[square] & cp.uint64(occ)
        file_pieces = GPU_BB_FILE_MASKS[square] & cp.uint64(occ)
        diag_pieces = GPU_BB_DIAG_MASKS[square] & cp.uint64(occ)

        qr = cp.uint64(self.queens | self.rooks)
        qb = cp.uint64(self.queens | self.bishops)

        attackers = (
            (GPU_BB_KING_ATTACKS[square] & cp.uint64(self.kings))
            | (GPU_BB_KNIGHT_ATTACKS[square] & cp.uint64(self.knights))
            | (
                cp.uint64(
                    chess.BB_RANK_ATTACKS[square][int(cp.asnumpy(rank_pieces))]
                )
                & qr
            )
            | (
                cp.uint64(
                    chess.BB_FILE_ATTACKS[square][int(cp.asnumpy(file_pieces))]
                )
                & qr
            )
            | (
                cp.uint64(
                    chess.BB_DIAG_ATTACKS[square][int(cp.asnumpy(diag_pieces))]
                )
                & qb
            )
            | (GPU_BB_PAWN_ATTACKS[not color][square] & cp.uint64(self.pawns))
        )

        result = attackers & cp.uint64(self.occupied_co[color])
        return int(result)

    def perft(self, depth: int) -> int:
        """Runs a performance test using GPU-accelerated move generation."""
        if depth <= 0:
            return 1
        if depth == 1:
            return self.legal_moves.count()

        count = 0
        for move in list(self.legal_moves):
            self.push(move)
            count += self.perft(depth - 1)
            self.pop()
        return count
