from __future__ import annotations

import typing

from . import (
    Board,
    Move,
    Color,
    Square,
    IntoSquareSet,
    SquareSet,
    BB_ALL,
    BB_RANK_MASKS,
    BB_FILE_MASKS,
    BB_DIAG_MASKS,
    BB_KING_ATTACKS,
    BB_KNIGHT_ATTACKS,
    BB_RANK_ATTACKS,
    BB_FILE_ATTACKS,
    BB_DIAG_ATTACKS,
    BB_PAWN_ATTACKS,
    BB_RANK_3,
    BB_RANK_4,
    BB_RANK_5,
    BB_RANK_6,
    WHITE,
    BLACK,
    scan_reversed,
    square_rank,
)

try:
    import cupy as cp  # type: ignore
    try:
        cp.cuda.runtime.getDeviceCount()
        GPU_AVAILABLE = True
    except Exception:
        GPU_AVAILABLE = False
except Exception:  # pragma: no cover - import failure
    cp = None  # type: ignore
    GPU_AVAILABLE = False

__all__ = ["GPU_AVAILABLE", "GPUBoard"]


class GPUBoard(Board):
    """A :class:`~chess.Board` variant that optionally uses cupy for heavy
    operations.

    If :mod:`cupy` is not available or no CUDA device is present, this class
    falls back to the regular :class:`~chess.Board` behaviour.
    """

    def __init__(self, *args: typing.Any, gpu: bool = True, **kwargs: typing.Any) -> None:
        super().__init__(*args, **kwargs)
        self._gpu_enabled = gpu and GPU_AVAILABLE
        if self._gpu_enabled:
            self._sync_gpu()

    # --- GPU helpers -----------------------------------------------------

    def _sync_gpu(self) -> None:
        """Synchronizes bitboards to the device."""
        assert cp is not None
        self._gp_pawns = cp.uint64(self.pawns)
        self._gp_knights = cp.uint64(self.knights)
        self._gp_bishops = cp.uint64(self.bishops)
        self._gp_rooks = cp.uint64(self.rooks)
        self._gp_queens = cp.uint64(self.queens)
        self._gp_kings = cp.uint64(self.kings)
        self._gp_promoted = cp.uint64(self.promoted)
        self._gp_occupied = cp.uint64(self.occupied)
        self._gp_occupied_co = [cp.uint64(self.occupied_co[0]), cp.uint64(self.occupied_co[1])]

    # Keep GPU state in sync when the board changes.
    def push(self, move: Move) -> None:  # type: ignore[override]
        super().push(move)
        if self._gpu_enabled:
            self._sync_gpu()

    def pop(self) -> Move:  # type: ignore[override]
        move = super().pop()
        if self._gpu_enabled:
            self._sync_gpu()
        return move

    def _attackers_mask_gpu(self, color: Color, square: Square, occupied: typing.Optional[int] = None) -> "cp.ndarray":
        assert cp is not None
        occ = self._gp_occupied if occupied is None else cp.uint64(occupied)
        rank_pieces = cp.uint64(BB_RANK_MASKS[square]) & occ
        file_pieces = cp.uint64(BB_FILE_MASKS[square]) & occ
        diag_pieces = cp.uint64(BB_DIAG_MASKS[square]) & occ

        qr = self._gp_queens | self._gp_rooks
        qb = self._gp_queens | self._gp_bishops

        attackers = (
            (cp.uint64(BB_KING_ATTACKS[square]) & self._gp_kings) |
            (cp.uint64(BB_KNIGHT_ATTACKS[square]) & self._gp_knights) |
            (cp.uint64(BB_RANK_ATTACKS[square][int(rank_pieces.item())]) & qr) |
            (cp.uint64(BB_FILE_ATTACKS[square][int(file_pieces.item())]) & qr) |
            (cp.uint64(BB_DIAG_ATTACKS[square][int(diag_pieces.item())]) & qb) |
            (cp.uint64(BB_PAWN_ATTACKS[not color][square]) & self._gp_pawns)
        )
        return attackers & self._gp_occupied_co[color]

    # --- Overrides -------------------------------------------------------

    def is_attacked_by(self, color: Color, square: Square, occupied: typing.Optional[IntoSquareSet] = None) -> bool:  # type: ignore[override]
        if self._gpu_enabled:
            mask = self._attackers_mask_gpu(color, square, None if occupied is None else SquareSet(occupied).mask)
            return bool(mask.item())
        return super().is_attacked_by(color, square, occupied)

    def generate_pseudo_legal_moves(self, from_mask: int = BB_ALL, to_mask: int = BB_ALL):  # type: ignore[override]
        if not self._gpu_enabled:
            yield from super().generate_pseudo_legal_moves(from_mask, to_mask)
            return

        assert cp is not None
        our_pieces = self._gp_occupied_co[self.turn]

        # Piece moves
        non_pawns = our_pieces & ~self._gp_pawns & from_mask
        for from_square in scan_reversed(int(non_pawns.item())):
            moves = self.attacks_mask(from_square) & ~int(our_pieces.item()) & to_mask
            for to_square in scan_reversed(moves):
                yield Move(from_square, to_square)

        # Castling moves
        if from_mask & self.kings:
            yield from self.generate_castling_moves(from_mask, to_mask)

        pawns = self._gp_pawns & our_pieces & from_mask
        if not pawns.item():
            return

        # Pawn captures
        capturers = pawns
        for from_square in scan_reversed(int(capturers.item())):
            targets = BB_PAWN_ATTACKS[self.turn][from_square] & int(self._gp_occupied_co[not self.turn].item()) & to_mask
            for to_square in scan_reversed(targets):
                if square_rank(to_square) in [0, 7]:
                    yield Move(from_square, to_square, QUEEN)
                    yield Move(from_square, to_square, ROOK)
                    yield Move(from_square, to_square, BISHOP)
                    yield Move(from_square, to_square, KNIGHT)
                else:
                    yield Move(from_square, to_square)

        # Pawn advances
        if self.turn == WHITE:
            single_moves = int(pawns.item()) << 8 & ~int(self._gp_occupied.item())
            double_moves = single_moves << 8 & ~int(self._gp_occupied.item()) & (BB_RANK_3 | BB_RANK_4)
        else:
            single_moves = int(pawns.item()) >> 8 & ~int(self._gp_occupied.item())
            double_moves = single_moves >> 8 & ~int(self._gp_occupied.item()) & (BB_RANK_6 | BB_RANK_5)

        single_moves &= to_mask
        double_moves &= to_mask

        for to_square in scan_reversed(single_moves):
            from_square = to_square + (8 if self.turn == BLACK else -8)
            if square_rank(to_square) in [0, 7]:
                yield Move(from_square, to_square, QUEEN)
                yield Move(from_square, to_square, ROOK)
                yield Move(from_square, to_square, BISHOP)
                yield Move(from_square, to_square, KNIGHT)
            else:
                yield Move(from_square, to_square)

        for to_square in scan_reversed(double_moves):
            from_square = to_square + (16 if self.turn == BLACK else -16)
            yield Move(from_square, to_square)

        if self.ep_square:
            yield from self.generate_pseudo_legal_ep(from_mask, to_mask)
