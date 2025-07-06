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

__all__ = ["GPUBoard", "GPU_AVAILABLE", "is_gpu_available"]

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
        rank_pieces = cp.uint64(chess.BB_RANK_MASKS[square]) & cp.uint64(occ)
        file_pieces = cp.uint64(chess.BB_FILE_MASKS[square]) & cp.uint64(occ)
        diag_pieces = cp.uint64(chess.BB_DIAG_MASKS[square]) & cp.uint64(occ)

        qr = cp.uint64(self.queens | self.rooks)
        qb = cp.uint64(self.queens | self.bishops)

        attackers = (
            (cp.uint64(chess.BB_KING_ATTACKS[square]) & cp.uint64(self.kings))
            | (cp.uint64(chess.BB_KNIGHT_ATTACKS[square]) & cp.uint64(self.knights))
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
            | (cp.uint64(chess.BB_PAWN_ATTACKS[not color][square]) & cp.uint64(self.pawns))
        )

        result = attackers & cp.uint64(self.occupied_co[color])
        return int(result)
