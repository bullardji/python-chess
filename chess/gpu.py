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
    "GPU_BB_SQUARES",
    "GPU_BB_DIAG_ATTACKS",
    "GPU_BB_FILE_ATTACKS",
    "GPU_BB_RANK_ATTACKS",
    "gpu_lsb",
    "gpu_msb",
    "gpu_popcount",
    "GPU_BB_RAYS",
    "gpu_ray",
    "gpu_between",
    "gpu_flip_vertical",
    "gpu_flip_horizontal",
    "gpu_flip_diagonal",
    "gpu_flip_anti_diagonal",
    "gpu_shift_down",
    "gpu_shift_2_down",
    "gpu_shift_up",
    "gpu_shift_2_up",
    "gpu_shift_right",
    "gpu_shift_2_right",
    "gpu_shift_left",
    "gpu_shift_2_left",
    "gpu_shift_up_left",
    "gpu_shift_up_right",
    "gpu_shift_down_left",
    "gpu_shift_down_right",
    "gpu_scan_reversed",
]

if GPU_AVAILABLE:
    GPU_BB_RANK_MASKS = cp.asarray(chess.BB_RANK_MASKS, dtype=cp.uint64)
    GPU_BB_FILE_MASKS = cp.asarray(chess.BB_FILE_MASKS, dtype=cp.uint64)
    GPU_BB_DIAG_MASKS = cp.asarray(chess.BB_DIAG_MASKS, dtype=cp.uint64)
    GPU_BB_KING_ATTACKS = cp.asarray(chess.BB_KING_ATTACKS, dtype=cp.uint64)
    GPU_BB_KNIGHT_ATTACKS = cp.asarray(chess.BB_KNIGHT_ATTACKS, dtype=cp.uint64)
    GPU_BB_PAWN_ATTACKS = cp.asarray(chess.BB_PAWN_ATTACKS, dtype=cp.uint64)
    GPU_BB_SQUARES = cp.asarray(chess.BB_SQUARES, dtype=cp.uint64)
    GPU_BB_RAYS = cp.asarray(chess.BB_RAYS, dtype=cp.uint64)
    GPU_BB_DIAG_ATTACKS = [
        {k: cp.uint64(v) for k, v in table.items()}
        for table in chess.BB_DIAG_ATTACKS
    ]
    GPU_BB_FILE_ATTACKS = [
        {k: cp.uint64(v) for k, v in table.items()}
        for table in chess.BB_FILE_ATTACKS
    ]
    GPU_BB_RANK_ATTACKS = [
        {k: cp.uint64(v) for k, v in table.items()}
        for table in chess.BB_RANK_ATTACKS
    ]
else:  # pragma: no cover - GPU not available
    GPU_BB_RANK_MASKS = None  # type: ignore
    GPU_BB_FILE_MASKS = None  # type: ignore
    GPU_BB_DIAG_MASKS = None  # type: ignore
    GPU_BB_KING_ATTACKS = None  # type: ignore
    GPU_BB_KNIGHT_ATTACKS = None  # type: ignore
    GPU_BB_PAWN_ATTACKS = None  # type: ignore
    GPU_BB_SQUARES = None  # type: ignore
    GPU_BB_RAYS = None  # type: ignore
    GPU_BB_DIAG_ATTACKS = None  # type: ignore
    GPU_BB_FILE_ATTACKS = None  # type: ignore
    GPU_BB_RANK_ATTACKS = None  # type: ignore

def is_gpu_available() -> bool:
    """Returns ``True`` if :mod:`cupy` is installed and a GPU is detected."""
    return GPU_AVAILABLE


def gpu_ray(a: chess.Square, b: chess.Square) -> chess.Bitboard:
    """GPU version of :func:`chess.ray`. Falls back to CPU if necessary."""
    if GPU_AVAILABLE:
        return int(cp.asnumpy(GPU_BB_RAYS[a, b]))
    return chess.ray(a, b)


def gpu_between(a: chess.Square, b: chess.Square) -> chess.Bitboard:
    """GPU version of :func:`chess.between`. Falls back to CPU if necessary."""
    if GPU_AVAILABLE:
        bb = GPU_BB_RAYS[a, b] & (
            (cp.uint64(chess.BB_ALL) << a) ^ (cp.uint64(chess.BB_ALL) << b)
        )
        bb = bb & (bb - cp.uint64(1))
        return int(cp.asnumpy(bb))
    return chess.between(a, b)


def gpu_lsb(bb: chess.Bitboard) -> int:
    """GPU version of :func:`chess.lsb`. Falls back to CPU if necessary."""
    if GPU_AVAILABLE:
        v = int(cp.asnumpy(cp.bitwise_and(cp.uint64(bb), -cp.uint64(bb))))
        return v.bit_length() - 1
    return chess.lsb(bb)


def gpu_msb(bb: chess.Bitboard) -> int:
    """GPU version of :func:`chess.msb`. Falls back to CPU if necessary."""
    if GPU_AVAILABLE:
        v = int(cp.asnumpy(cp.uint64(bb)))
        return v.bit_length() - 1
    return chess.msb(bb)


def gpu_popcount(bb: chess.Bitboard) -> int:
    """GPU version of :func:`chess.popcount`. Falls back to CPU if necessary."""
    if GPU_AVAILABLE:
        x = cp.uint64(bb)
        x = x - ((x >> cp.uint64(1)) & cp.uint64(0x5555_5555_5555_5555))
        x = (x & cp.uint64(0x3333_3333_3333_3333)) + ((x >> cp.uint64(2)) & cp.uint64(0x3333_3333_3333_3333))
        x = (x + (x >> cp.uint64(4))) & cp.uint64(0x0f0f_0f0f_0f0f_0f0f)
        x = (x * cp.uint64(0x0101_0101_0101_0101)) >> cp.uint64(56)
        return int(cp.asnumpy(x))
    return chess.popcount(bb)


def gpu_flip_vertical(bb: chess.Bitboard) -> chess.Bitboard:
    """GPU version of :func:`chess.flip_vertical`."""
    if GPU_AVAILABLE:
        x = cp.uint64(bb)
        x = ((x >> cp.uint64(8)) & cp.uint64(0x00ff_00ff_00ff_00ff)) | ((x & cp.uint64(0x00ff_00ff_00ff_00ff)) << cp.uint64(8))
        x = ((x >> cp.uint64(16)) & cp.uint64(0x0000_ffff_0000_ffff)) | ((x & cp.uint64(0x0000_ffff_0000_ffff)) << cp.uint64(16))
        x = (x >> cp.uint64(32)) | ((x & cp.uint64(0x0000_0000_ffff_ffff)) << cp.uint64(32))
        return int(cp.asnumpy(x))
    return chess.flip_vertical(bb)


def gpu_flip_horizontal(bb: chess.Bitboard) -> chess.Bitboard:
    """GPU version of :func:`chess.flip_horizontal`."""
    if GPU_AVAILABLE:
        x = cp.uint64(bb)
        x = ((x >> cp.uint64(1)) & cp.uint64(0x5555_5555_5555_5555)) | ((x & cp.uint64(0x5555_5555_5555_5555)) << cp.uint64(1))
        x = ((x >> cp.uint64(2)) & cp.uint64(0x3333_3333_3333_3333)) | ((x & cp.uint64(0x3333_3333_3333_3333)) << cp.uint64(2))
        x = ((x >> cp.uint64(4)) & cp.uint64(0x0f0f_0f0f_0f0f_0f0f)) | ((x & cp.uint64(0x0f0f_0f0f_0f0f_0f0f)) << cp.uint64(4))
        return int(cp.asnumpy(x))
    return chess.flip_horizontal(bb)


def gpu_flip_diagonal(bb: chess.Bitboard) -> chess.Bitboard:
    """GPU version of :func:`chess.flip_diagonal`."""
    if GPU_AVAILABLE:
        x = cp.uint64(bb)
        t = (x ^ (x << cp.uint64(28))) & cp.uint64(0x0f0f_0f0f_0000_0000)
        x = x ^ t ^ (t >> cp.uint64(28))
        t = (x ^ (x << cp.uint64(14))) & cp.uint64(0x3333_0000_3333_0000)
        x = x ^ t ^ (t >> cp.uint64(14))
        t = (x ^ (x << cp.uint64(7))) & cp.uint64(0x5500_5500_5500_5500)
        x = x ^ t ^ (t >> cp.uint64(7))
        return int(cp.asnumpy(x))
    return chess.flip_diagonal(bb)


def gpu_flip_anti_diagonal(bb: chess.Bitboard) -> chess.Bitboard:
    """GPU version of :func:`chess.flip_anti_diagonal`."""
    if GPU_AVAILABLE:
        x = cp.uint64(bb)
        t = x ^ (x << cp.uint64(36))
        x = x ^ ((t ^ (x >> cp.uint64(36))) & cp.uint64(0xf0f0_f0f0_0f0f_0f0f))
        t = (x ^ (x << cp.uint64(18))) & cp.uint64(0xcccc_0000_cccc_0000)
        x = x ^ t ^ (t >> cp.uint64(18))
        t = (x ^ (x << cp.uint64(9))) & cp.uint64(0xaa00_aa00_aa00_aa00)
        x = x ^ t ^ (t >> cp.uint64(9))
        return int(cp.asnumpy(x))
    return chess.flip_anti_diagonal(bb)


def gpu_shift_down(b: chess.Bitboard) -> chess.Bitboard:
    if GPU_AVAILABLE:
        return int(cp.asnumpy(cp.uint64(b) >> cp.uint64(8)))
    return chess.shift_down(b)


def gpu_shift_2_down(b: chess.Bitboard) -> chess.Bitboard:
    if GPU_AVAILABLE:
        return int(cp.asnumpy(cp.uint64(b) >> cp.uint64(16)))
    return chess.shift_2_down(b)


def gpu_shift_up(b: chess.Bitboard) -> chess.Bitboard:
    if GPU_AVAILABLE:
        return int(cp.asnumpy((cp.uint64(b) << cp.uint64(8)) & cp.uint64(chess.BB_ALL)))
    return chess.shift_up(b)


def gpu_shift_2_up(b: chess.Bitboard) -> chess.Bitboard:
    if GPU_AVAILABLE:
        return int(cp.asnumpy((cp.uint64(b) << cp.uint64(16)) & cp.uint64(chess.BB_ALL)))
    return chess.shift_2_up(b)


def gpu_shift_right(b: chess.Bitboard) -> chess.Bitboard:
    if GPU_AVAILABLE:
        return int(cp.asnumpy((cp.uint64(b) << cp.uint64(1)) & cp.uint64(~chess.BB_FILE_A) & cp.uint64(chess.BB_ALL)))
    return chess.shift_right(b)


def gpu_shift_2_right(b: chess.Bitboard) -> chess.Bitboard:
    if GPU_AVAILABLE:
        return int(cp.asnumpy((cp.uint64(b) << cp.uint64(2)) & cp.uint64(~chess.BB_FILE_A) & cp.uint64(~chess.BB_FILE_B) & cp.uint64(chess.BB_ALL)))
    return chess.shift_2_right(b)


def gpu_shift_left(b: chess.Bitboard) -> chess.Bitboard:
    if GPU_AVAILABLE:
        return int(cp.asnumpy((cp.uint64(b) >> cp.uint64(1)) & cp.uint64(~chess.BB_FILE_H)))
    return chess.shift_left(b)


def gpu_shift_2_left(b: chess.Bitboard) -> chess.Bitboard:
    if GPU_AVAILABLE:
        return int(cp.asnumpy((cp.uint64(b) >> cp.uint64(2)) & cp.uint64(~chess.BB_FILE_G) & cp.uint64(~chess.BB_FILE_H)))
    return chess.shift_2_left(b)


def gpu_shift_up_left(b: chess.Bitboard) -> chess.Bitboard:
    if GPU_AVAILABLE:
        return int(cp.asnumpy((cp.uint64(b) << cp.uint64(7)) & cp.uint64(~chess.BB_FILE_H) & cp.uint64(chess.BB_ALL)))
    return chess.shift_up_left(b)


def gpu_shift_up_right(b: chess.Bitboard) -> chess.Bitboard:
    if GPU_AVAILABLE:
        return int(cp.asnumpy((cp.uint64(b) << cp.uint64(9)) & cp.uint64(~chess.BB_FILE_A) & cp.uint64(chess.BB_ALL)))
    return chess.shift_up_right(b)


def gpu_shift_down_left(b: chess.Bitboard) -> chess.Bitboard:
    if GPU_AVAILABLE:
        return int(cp.asnumpy((cp.uint64(b) >> cp.uint64(9)) & cp.uint64(~chess.BB_FILE_H)))
    return chess.shift_down_left(b)


def gpu_shift_down_right(b: chess.Bitboard) -> chess.Bitboard:
    if GPU_AVAILABLE:
        return int(cp.asnumpy((cp.uint64(b) >> cp.uint64(7)) & cp.uint64(~chess.BB_FILE_A)))
    return chess.shift_down_right(b)


def gpu_scan_reversed(bb: chess.Bitboard) -> typing.Iterator[chess.Square]:
    """GPU-enabled version of :func:`chess.scan_reversed`."""
    while bb:
        sq = gpu_msb(bb)
        yield sq
        bb ^= chess.BB_SQUARES[sq]


class GPUBoard(chess.Board):
    """A board with optional GPU-accelerated methods."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if GPU_AVAILABLE:
            self._sync_gpu()

    def _sync_gpu(self) -> None:
        if not GPU_AVAILABLE:
            return
        self._gpu_pawns = cp.uint64(self.pawns)
        self._gpu_knights = cp.uint64(self.knights)
        self._gpu_bishops = cp.uint64(self.bishops)
        self._gpu_rooks = cp.uint64(self.rooks)
        self._gpu_queens = cp.uint64(self.queens)
        self._gpu_kings = cp.uint64(self.kings)
        self._gpu_promoted = cp.uint64(self.promoted)
        self._gpu_occupied = cp.uint64(self.occupied)
        self._gpu_occupied_co = [
            cp.uint64(self.occupied_co[chess.WHITE]),
            cp.uint64(self.occupied_co[chess.BLACK]),
        ]

    def push(self, move: chess.Move) -> None:
        super().push(move)
        if GPU_AVAILABLE:
            self._sync_gpu()

    def pop(self) -> chess.Move:
        move = super().pop()
        if GPU_AVAILABLE:
            self._sync_gpu()
        return move

    def attackers_mask(
        self,
        color: chess.Color,
        square: chess.Square,
        occupied: typing.Optional[chess.Bitboard] = None,
    ) -> chess.Bitboard:
        """Like :meth:`chess.Board.attackers_mask` but uses the GPU when possible."""
        if not GPU_AVAILABLE:
            return super().attackers_mask(color, square, occupied)

        occ = self._gpu_occupied if occupied is None else cp.uint64(occupied)
        rank_pieces = GPU_BB_RANK_MASKS[square] & occ
        file_pieces = GPU_BB_FILE_MASKS[square] & occ
        diag_pieces = GPU_BB_DIAG_MASKS[square] & occ

        qr = self._gpu_queens | self._gpu_rooks
        qb = self._gpu_queens | self._gpu_bishops

        attackers = (
            (GPU_BB_KING_ATTACKS[square] & self._gpu_kings)
            | (GPU_BB_KNIGHT_ATTACKS[square] & self._gpu_knights)
            | (GPU_BB_RANK_ATTACKS[square][int(cp.asnumpy(rank_pieces))] & qr)
            | (GPU_BB_FILE_ATTACKS[square][int(cp.asnumpy(file_pieces))] & qr)
            | (GPU_BB_DIAG_ATTACKS[square][int(cp.asnumpy(diag_pieces))] & qb)
            | (GPU_BB_PAWN_ATTACKS[not color][square] & self._gpu_pawns)
        )

        result = attackers & self._gpu_occupied_co[color]
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

    def attacks_mask(self, square: chess.Square) -> chess.Bitboard:
        """Like :meth:`chess.Board.attacks_mask` but uses GPU tables."""
        if not GPU_AVAILABLE:
            return super().attacks_mask(square)

        bb_square = chess.BB_SQUARES[square]

        if bb_square & self.pawns:
            color = bool(bb_square & self.occupied_co[chess.WHITE])
            return int(GPU_BB_PAWN_ATTACKS[color][square])
        elif bb_square & self.knights:
            return int(GPU_BB_KNIGHT_ATTACKS[square])
        elif bb_square & self.kings:
            return int(GPU_BB_KING_ATTACKS[square])
        else:
            attacks = 0
            if bb_square & self.bishops or bb_square & self.queens:
                occ = int(cp.asnumpy(GPU_BB_DIAG_MASKS[square] & self._gpu_occupied))
                attacks = int(GPU_BB_DIAG_ATTACKS[square][occ])
            if bb_square & self.rooks or bb_square & self.queens:
                occ_rank = int(cp.asnumpy(GPU_BB_RANK_MASKS[square] & self._gpu_occupied))
                occ_file = int(cp.asnumpy(GPU_BB_FILE_MASKS[square] & self._gpu_occupied))
                attacks |= int(GPU_BB_RANK_ATTACKS[square][occ_rank])
                attacks |= int(GPU_BB_FILE_ATTACKS[square][occ_file])
            return attacks

    def generate_pseudo_legal_ep(
        self,
        from_mask: chess.Bitboard = chess.BB_ALL,
        to_mask: chess.Bitboard = chess.BB_ALL,
    ) -> typing.Iterator[chess.Move]:
        if not GPU_AVAILABLE:
            yield from super().generate_pseudo_legal_ep(from_mask, to_mask)
            return

        if not self.ep_square or not chess.BB_SQUARES[self.ep_square] & to_mask:
            return
        if chess.BB_SQUARES[self.ep_square] & self.occupied:
            return

        capturers = (
            self.pawns
            & self.occupied_co[self.turn]
            & from_mask
            & chess.BB_PAWN_ATTACKS[not self.turn][self.ep_square]
            & chess.BB_RANKS[4 if self.turn else 3]
        )

        for capturer in gpu_scan_reversed(capturers):
            yield chess.Move(capturer, self.ep_square)

    def generate_pseudo_legal_moves(
        self,
        from_mask: chess.Bitboard = chess.BB_ALL,
        to_mask: chess.Bitboard = chess.BB_ALL,
    ) -> typing.Iterator[chess.Move]:
        if not GPU_AVAILABLE:
            yield from super().generate_pseudo_legal_moves(from_mask, to_mask)
            return

        our_pieces = self.occupied_co[self.turn]

        non_pawns = our_pieces & ~self.pawns & from_mask
        for from_square in gpu_scan_reversed(non_pawns):
            moves = self.attacks_mask(from_square) & ~our_pieces & to_mask
            for to_square in gpu_scan_reversed(moves):
                yield chess.Move(from_square, to_square)

        if from_mask & self.kings:
            yield from self.generate_castling_moves(from_mask, to_mask)

        pawns = self.pawns & our_pieces & from_mask
        if not pawns:
            return

        for from_square in gpu_scan_reversed(pawns):
            targets = (
                chess.gpu.GPU_BB_PAWN_ATTACKS[self.turn][from_square]
                & cp.uint64(self.occupied_co[not self.turn])
                & cp.uint64(to_mask)
            )
            targets = int(cp.asnumpy(targets))
            for to_square in gpu_scan_reversed(targets):
                if chess.square_rank(to_square) in [0, 7]:
                    yield chess.Move(from_square, to_square, chess.QUEEN)
                    yield chess.Move(from_square, to_square, chess.ROOK)
                    yield chess.Move(from_square, to_square, chess.BISHOP)
                    yield chess.Move(from_square, to_square, chess.KNIGHT)
                else:
                    yield chess.Move(from_square, to_square)

        if self.turn == chess.WHITE:
            single = gpu_shift_up(pawns) & ~self.occupied
            double = gpu_shift_up(single) & ~self.occupied & (chess.BB_RANK_3 | chess.BB_RANK_4)
        else:
            single = gpu_shift_down(pawns) & ~self.occupied
            double = gpu_shift_down(single) & ~self.occupied & (chess.BB_RANK_6 | chess.BB_RANK_5)

        single &= to_mask
        double &= to_mask

        for to_square in gpu_scan_reversed(single):
            from_square = to_square + (8 if self.turn == chess.BLACK else -8)
            if chess.square_rank(to_square) in [0, 7]:
                yield chess.Move(from_square, to_square, chess.QUEEN)
                yield chess.Move(from_square, to_square, chess.ROOK)
                yield chess.Move(from_square, to_square, chess.BISHOP)
                yield chess.Move(from_square, to_square, chess.KNIGHT)
            else:
                yield chess.Move(from_square, to_square)

        for to_square in gpu_scan_reversed(double):
            from_square = to_square + (16 if self.turn == chess.BLACK else -16)
            yield chess.Move(from_square, to_square)

        if self.ep_square:
            yield from self.generate_pseudo_legal_ep(from_mask, to_mask)

    def pin_mask(self, color: chess.Color, square: chess.Square) -> chess.Bitboard:
        if not GPU_AVAILABLE:
            return super().pin_mask(color, square)

        king = self.king(color)
        if king is None:
            return chess.BB_ALL

        square_mask = chess.BB_SQUARES[square]

        for attacks, sliders in [
            (chess.gpu.GPU_BB_FILE_ATTACKS, self.rooks | self.queens),
            (chess.gpu.GPU_BB_RANK_ATTACKS, self.rooks | self.queens),
            (chess.gpu.GPU_BB_DIAG_ATTACKS, self.bishops | self.queens),
        ]:
            rays = attacks[king][0]
            if int(cp.asnumpy(rays & cp.uint64(square_mask))):
                snipers = int(
                    cp.asnumpy(
                        rays & cp.uint64(sliders) & cp.uint64(self.occupied_co[not color])
                    )
                )
                for sniper in gpu_scan_reversed(snipers):
                    if (
                        gpu_between(sniper, king) & (self.occupied | square_mask)
                    ) == square_mask:
                        return gpu_ray(king, sniper)
                break

        return chess.BB_ALL

    def _ep_skewered(self, king: chess.Square, capturer: chess.Square) -> bool:
        if not GPU_AVAILABLE:
            return super()._ep_skewered(king, capturer)

        assert self.ep_square is not None

        last_double = self.ep_square + (-8 if self.turn == chess.WHITE else 8)

        occupancy = (
            cp.uint64(self.occupied)
            & ~chess.gpu.GPU_BB_SQUARES[last_double]
            & ~chess.gpu.GPU_BB_SQUARES[capturer]
            | chess.gpu.GPU_BB_SQUARES[self.ep_square]
        )

        horizontal = cp.uint64(self.occupied_co[not self.turn] & (self.rooks | self.queens))
        mask = chess.gpu.GPU_BB_RANK_MASKS[king] & occupancy
        if chess.gpu.GPU_BB_RANK_ATTACKS[king][int(cp.asnumpy(mask))] & horizontal:
            return True

        diagonal = cp.uint64(self.occupied_co[not self.turn] & (self.bishops | self.queens))
        mask = chess.gpu.GPU_BB_DIAG_MASKS[king] & occupancy
        if chess.gpu.GPU_BB_DIAG_ATTACKS[king][int(cp.asnumpy(mask))] & diagonal:
            return True

        return False

    def _slider_blockers(self, king: chess.Square) -> chess.Bitboard:
        if not GPU_AVAILABLE:
            return super()._slider_blockers(king)

        rooks_and_queens = self.rooks | self.queens
        bishops_and_queens = self.bishops | self.queens
        snipers = (
            (
                chess.gpu.GPU_BB_RANK_ATTACKS[king][0] & cp.uint64(rooks_and_queens)
            )
            | (
                chess.gpu.GPU_BB_FILE_ATTACKS[king][0] & cp.uint64(rooks_and_queens)
            )
            | (
                chess.gpu.GPU_BB_DIAG_ATTACKS[king][0] & cp.uint64(bishops_and_queens)
            )
        )

        blockers = 0

        for sniper in gpu_scan_reversed(int(cp.asnumpy(snipers & cp.uint64(self.occupied_co[not self.turn])))):
            b = gpu_between(king, sniper) & self.occupied
            if b and chess.BB_SQUARES[gpu_msb(b)] == b:
                blockers |= b

        return blockers & self.occupied_co[self.turn]

    def _is_safe(self, king: chess.Square, blockers: chess.Bitboard, move: chess.Move) -> bool:
        if not GPU_AVAILABLE:
            return super()._is_safe(king, blockers, move)

        if move.from_square == king:
            if self.is_castling(move):
                return True
            else:
                return not bool(self.attackers_mask(not self.turn, move.to_square))
        elif self.is_en_passant(move):
            return bool(
                self.pin_mask(self.turn, move.from_square)
                & chess.BB_SQUARES[move.to_square]
                and not self._ep_skewered(king, move.from_square)
            )
        else:
            return bool(
                not blockers & chess.BB_SQUARES[move.from_square]
                or gpu_ray(move.from_square, move.to_square) & chess.BB_SQUARES[king]
            )

    def _generate_evasions(
        self,
        king: chess.Square,
        checkers: chess.Bitboard,
        from_mask: chess.Bitboard = chess.BB_ALL,
        to_mask: chess.Bitboard = chess.BB_ALL,
    ) -> typing.Iterator[chess.Move]:
        if not GPU_AVAILABLE:
            yield from super()._generate_evasions(king, checkers, from_mask, to_mask)
            return

        sliders = checkers & (self.bishops | self.rooks | self.queens)

        attacked = 0
        for checker in gpu_scan_reversed(sliders):
            attacked |= gpu_ray(king, checker) & ~chess.BB_SQUARES[checker]

        if chess.BB_SQUARES[king] & from_mask:
            targets = (
                chess.gpu.GPU_BB_KING_ATTACKS[king]
                & cp.uint64(~self.occupied_co[self.turn])
                & cp.uint64(~attacked)
                & cp.uint64(to_mask)
            )
            for to_square in gpu_scan_reversed(int(cp.asnumpy(targets))):
                yield chess.Move(king, to_square)

        checker = gpu_msb(checkers)
        if chess.BB_SQUARES[checker] == checkers:
            target = gpu_between(king, checker) | checkers

            yield from self.generate_pseudo_legal_moves(~self.kings & from_mask, target & to_mask)

            if self.ep_square and not chess.BB_SQUARES[self.ep_square] & target:
                last_double = self.ep_square + (-8 if self.turn == chess.WHITE else 8)
                if last_double == checker:
                    yield from self.generate_pseudo_legal_ep(from_mask, to_mask)

    def _attacked_for_king(self, path: chess.Bitboard, occupied: chess.Bitboard) -> bool:
        if not GPU_AVAILABLE:
            return super()._attacked_for_king(path, occupied)
        return any(self.attackers_mask(not self.turn, sq, occupied) for sq in gpu_scan_reversed(path))

    def generate_castling_moves(
        self,
        from_mask: chess.Bitboard = chess.BB_ALL,
        to_mask: chess.Bitboard = chess.BB_ALL,
    ) -> typing.Iterator[chess.Move]:
        if not GPU_AVAILABLE:
            yield from super().generate_castling_moves(from_mask, to_mask)
            return

        if self.is_variant_end():
            return

        backrank = chess.BB_RANK_1 if self.turn == chess.WHITE else chess.BB_RANK_8
        king = (
            self.occupied_co[self.turn]
            & self.kings
            & ~self.promoted
            & backrank
            & from_mask
        )
        king &= -king
        if not king:
            return

        bb_c = chess.BB_FILE_C & backrank
        bb_d = chess.BB_FILE_D & backrank
        bb_f = chess.BB_FILE_F & backrank
        bb_g = chess.BB_FILE_G & backrank

        rights = self.clean_castling_rights() & backrank & to_mask
        for candidate in gpu_scan_reversed(rights):
            rook = chess.BB_SQUARES[candidate]
            a_side = rook < king
            king_to = bb_c if a_side else bb_g
            rook_to = bb_d if a_side else bb_f

            king_path = gpu_between(gpu_msb(king), gpu_msb(king_to))
            rook_path = gpu_between(candidate, gpu_msb(rook_to))

            if not (
                (self.occupied ^ king ^ rook) & (king_path | rook_path | king_to | rook_to)
                or self._attacked_for_king(king_path | king, self.occupied ^ king)
                or self._attacked_for_king(king_to, self.occupied ^ king ^ rook ^ rook_to)
            ):
                yield self._from_chess960(self.chess960, gpu_msb(king), candidate)

    def generate_legal_moves(
        self,
        from_mask: chess.Bitboard = chess.BB_ALL,
        to_mask: chess.Bitboard = chess.BB_ALL,
    ) -> typing.Iterator[chess.Move]:
        if not GPU_AVAILABLE:
            yield from super().generate_legal_moves(from_mask, to_mask)
            return

        if self.is_variant_end():
            return

        king_mask = self.kings & self.occupied_co[self.turn]
        if king_mask:
            king = gpu_msb(king_mask)
            blockers = self._slider_blockers(king)
            checkers = self.attackers_mask(not self.turn, king)
            if checkers:
                for move in self._generate_evasions(king, checkers, from_mask, to_mask):
                    if self._is_safe(king, blockers, move):
                        yield move
            else:
                for move in self.generate_pseudo_legal_moves(from_mask, to_mask):
                    if self._is_safe(king, blockers, move):
                        yield move
        else:
            yield from self.generate_pseudo_legal_moves(from_mask, to_mask)

