import chess
from typing import Tuple

# how much each piece is worth in the middlegame vs endgame
# bishops/knights go down in value as pieces come off, rooks go up
MG = {chess.PAWN:82, chess.KNIGHT:337, chess.BISHOP:365,
      chess.ROOK:477, chess.QUEEN:1025, chess.KING:0}
EG = {chess.PAWN:94, chess.KNIGHT:281, chess.BISHOP:297,
      chess.ROOK:512, chess.QUEEN: 936, chess.KING:0}

# these tables tell each piece where it wants to be on the board
# positive = good square, negative = bad square
# laid out from white's perspective so a1 is index 0, h8 is index 63
MG_PST = {
chess.PAWN: [
    0,  0,  0,  0,  0,  0,  0,  0,
   98,134, 61, 95, 68,126, 34,-11,
   -6,  7, 26, 31, 65, 56, 25,-20,
  -14, 13,  6, 21, 23, 12, 17,-23,
  -27, -2, -5, 12, 17,  6, 10,-25,
  -26, -4, -4,-10,  3,  3, 33,-12,
  -35, -1,-20,-23,-15, 24, 38,-22,
    0,  0,  0,  0,  0,  0,  0,  0,
],
chess.KNIGHT: [
 -167,-89,-34,-49, 61,-97,-15,-107,
  -73,-41, 72, 36, 23, 62,  7, -17,
  -47, 60, 37, 65, 84,129, 73,  44,
   -9, 17, 19, 53, 37, 69, 18,  22,
  -13,  4, 16, 13, 28, 19, 21,  -8,
  -23, -9, 12, 10, 19, 17, 25, -16,
  -29,-53,-12, -3, -1, 18,-14, -19,
 -105,-21,-58,-33,-17,-28,-19, -23,
],
chess.BISHOP: [
  -29,  4,-82,-37,-25,-42,  7, -8,
  -26, 16,-18,-13, 30, 59, 18,-47,
  -16, 37, 43, 40, 35, 50, 37, -2,
   -4,  5, 19, 50, 37, 37,  7, -2,
   -6, 13, 13, 26, 34, 12, 10,  4,
    0, 15, 15, 15, 14, 27, 18, 10,
    4, 15, 16,  0,  7, 21, 33,  1,
  -33, -3,-14,-21,-13,-12,-39,-21,
],
chess.ROOK: [
   32, 42, 32, 51, 63,  9, 31, 43,
   27, 32, 58, 62, 80, 67, 26, 44,
   -5, 19, 26, 36, 17, 45, 61, 16,
  -24,-11,  7, 26, 24, 35, -8,-20,
  -36,-26,-12, -1,  9, -7,  6,-23,
  -45,-25,-16,-17,  3,  0, -5,-33,
  -44,-16,-20, -9, -1, 11, -6,-71,
  -19,-13,  1, 17, 16,  7,-37,-26,
],
chess.QUEEN: [
  -28,  0, 29, 12, 59, 44, 43, 45,
  -24,-39, -5,  1,-16, 57, 28, 54,
  -13,-17,  7,  8, 29, 56, 47, 57,
  -27,-27,-16,-16, -1, 17, -2,  1,
   -9,-26, -9,-10, -2, -4,  3, -3,
  -14,  2,-11, -2, -5,  2, 14,  5,
  -35, -8, 11,  2,  8, 15, -3,  1,
   -1,-18, -9, 10,-15,-25,-31,-50,
],
chess.KING: [
  -65, 23, 16,-15,-56,-34,  2, 13,
   29, -1,-20, -7, -8, -4,-38,-29,
   -9, 24,  2,-16,-20,  6, 22,-22,
  -17,-20,-12,-27,-30,-25,-14,-36,
  -49, -1,-27,-39,-46,-44,-33,-51,
  -14,-14,-22,-46,-44,-30,-15,-27,
    1,  7, -8,-64,-43,-16,  9,  8,
  -15, 36, 12,-54,  8,-28, 24, 14,
],
}
EG_PST = {
chess.PAWN: [
    0,  0,  0,  0,  0,  0,  0,  0,
  178,173,158,134,147,132,165,187,
   94,100, 85, 67, 56, 53, 82, 84,
   32, 24, 13,  5, -2,  4, 17, 17,
   13,  9, -3, -7, -7, -8,  3, -1,
    4,  7, -6,  1,  0, -5, -1, -8,
   13,  8,  8, 10, 13,  0,  2, -7,
    0,  0,  0,  0,  0,  0,  0,  0,
],
chess.KNIGHT: [
  -58,-38,-13,-28,-31,-27,-63,-99,
  -25, -8,-25, -2, -9,-25,-24,-52,
  -24,-20, 10,  9, -1, -9,-19,-41,
  -17,  3, 22, 22, 22, 11,  8,-18,
  -18, -6, 16, 25, 16, 17,  4,-18,
  -23, -3, -1, 15, 10, -3,-20,-22,
  -42,-20,-10, -5, -2,-20,-23,-44,
  -29,-51,-23,-15,-22,-18,-50,-64,
],
chess.BISHOP: [
  -14,-21,-11, -8, -7, -9,-17,-24,
   -8, -4,  7,-12, -3,-13, -4,-14,
    2, -8,  0, -1, -2,  6,  0,  4,
   -3,  9, 12,  9, 14, 10,  3,  2,
   -6,  3, 13, 19,  7, 10, -3, -9,
  -12, -3,  8, 10, 13,  3, -7,-15,
  -14,-18, -7, -1,  4, -9,-15,-27,
  -23, -9,-23, -5, -9,-16, -5,-17,
],
chess.ROOK: [
   13, 10, 18, 15, 12, 12,  8,  5,
   11, 13, 13, 11, -3,  3,  8,  3,
    7,  7,  7,  5,  4, -3, -5, -3,
    4,  3, 13,  1,  2,  1, -1,  2,
    3,  5,  8,  4, -5, -6, -8,-11,
   -4,  0, -5, -1, -7,-12, -8,-16,
   -6, -6,  0,  2, -9, -9,-11, -3,
   -9,  2,  3, -1, -5,-13,  4,-20,
],
chess.QUEEN: [
   -9, 22, 22, 27, 27, 19, 10, 20,
  -17, 20, 32, 41, 58, 25, 30,  0,
  -20,  6,  9, 49, 47, 35, 19,  9,
    3, 22, 24, 45, 57, 40, 57, 36,
  -18, 28, 19, 47, 31, 34, 39, 23,
  -16,-27, 15,  6,  9, 17, 10,  5,
  -22,-23,-30,-16,-16,-23,-36,-32,
  -33,-28,-22,-43, -5,-32,-20,-41,
],
chess.KING: [
  -74,-35,-18,-18,-11, 15,  4,-17,
  -12, 17, 14, 17, 17, 38, 23, 11,
   10, 17, 23, 15, 20, 45, 44, 13,
   -8, 22, 24, 27, 26, 33, 26,  3,
  -18, -4, 21, 24, 27, 23,  9,-11,
  -19, -3, 11, 21, 23, 16,  7, -9,
  -27,-11,  4, 13, 14,  4, -5,-17,
  -53,-34,-21,-11,-28,-14,-24,-43,
],
}

# bonus for knights on advanced central squares that enemy pawns can't attack
OUTPOST_BONUS = [0] * 64
for _sq in chess.SQUARES:
    _f, _r = chess.square_file(_sq), chess.square_rank(_sq)
    if 2 <= _f <= 5 and _r >= 4:
        OUTPOST_BONUS[_sq] = 15 + (_r - 4) * 8


def _msq(sq: int) -> int:
    # flip a square vertically so black's pieces use the same tables as white
    return sq ^ 56


def game_phase(board: chess.Board) -> float:
    # used to blend MG and EG scores as pieces come off the board
    p = (len(board.pieces(chess.KNIGHT, chess.WHITE)) +
         len(board.pieces(chess.KNIGHT, chess.BLACK)) +
         len(board.pieces(chess.BISHOP, chess.WHITE)) +
         len(board.pieces(chess.BISHOP, chess.BLACK)) +
         2 * (len(board.pieces(chess.ROOK, chess.WHITE)) +
              len(board.pieces(chess.ROOK, chess.BLACK))) +
         4 * (len(board.pieces(chess.QUEEN, chess.WHITE)) +
              len(board.pieces(chess.QUEEN, chess.BLACK))))
    return min(p, 24) / 24.0


def _pst_score(board: chess.Board) -> Tuple[int, int]:
    # sum up material + square bonuses for every piece
    mg = eg = 0
    for sq in chess.SQUARES:
        pc = board.piece_at(sq)
        if pc is None:
            continue
        idx = sq if pc.color == chess.WHITE else _msq(sq)
        s = 1 if pc.color == chess.WHITE else -1
        mg += s * (MG[pc.piece_type] + MG_PST[pc.piece_type][idx])
        eg += s * (EG[pc.piece_type] + EG_PST[pc.piece_type][idx])
    return mg, eg


def _pawn_structure(board: chess.Board) -> int:
    score = 0
    wpawns = board.pieces(chess.PAWN, chess.WHITE)
    bpawns = board.pieces(chess.PAWN, chess.BLACK)

    for color in (chess.WHITE, chess.BLACK):
        s      = 1 if color == chess.WHITE else -1
        mine   = wpawns if color == chess.WHITE else bpawns
        theirs = bpawns if color == chess.WHITE else wpawns
        files  = [chess.square_file(sq) for sq in mine]

        for sq in mine:
            f = chess.square_file(sq)
            r = chess.square_rank(sq)
            adj = {f - 1, f + 1} & set(range(8))

            # two pawns on the same file block each other
            if files.count(f) > 1:
                score -= s * 12

            # no friendly pawns on either side — easy target
            if not any(ff in adj for ff in files):
                score -= s * 22

            # backward pawn — no support from behind
            behind = r - 1 if color == chess.WHITE else r + 1
            if 0 <= behind <= 7:
                if not any(chess.square_file(p) in adj and chess.square_rank(p) == behind
                           for p in mine):
                    score -= s * 8

            # passed pawn — nothing blocking it from promoting
            if color == chess.WHITE:
                passed = all(chess.square_rank(p) <= r for p in theirs
                             if chess.square_file(p) in ({f} | adj))
                bonus = [0, 5, 10, 20, 40, 65, 95, 0][r]
            else:
                passed = all(chess.square_rank(p) >= r for p in theirs
                             if chess.square_file(p) in ({f} | adj))
                bonus = [0, 95, 65, 40, 20, 10, 5, 0][r]
            if passed:
                score += s * bonus

            # phalanx — two pawns next to each other are hard to break
            if any(chess.square_file(p) in adj and chess.square_rank(p) == r for p in mine):
                score += s * 8

    return score


def _king_safety(board: chess.Board, phase: float) -> int:
    # not relevant in the endgame where the king should be active
    if phase < 0.15:
        return 0

    score = 0
    for color in (chess.WHITE, chess.BLACK):
        s      = 1 if color == chess.WHITE else -1
        ksq    = board.king(color)
        if ksq is None:
            continue
        kf     = chess.square_file(ksq)
        kr     = chess.square_rank(ksq)
        mine   = board.pieces(chess.PAWN, color)
        theirs = board.pieces(chess.PAWN, not color)

        # pawns in front of the king act as a shield
        shield = 0
        for f in range(max(0, kf - 1), min(8, kf + 2)):
            for dr in (1, 2):
                pr = kr + (dr if color == chess.WHITE else -dr)
                if 0 <= pr <= 7 and chess.square(f, pr) in mine:
                    shield += 12 - dr * 2
                    break

        # open files near the king let enemy rooks/queens in
        exposed = 0
        for f in range(max(0, kf - 1), min(8, kf + 2)):
            own = any(chess.square_file(p) == f for p in mine)
            opp = any(chess.square_file(p) == f for p in theirs)
            if not own:
                exposed += 18 if not opp else 9

        # enemy pieces nearby are a danger
        attack_weight = 0
        for pt, w in ((chess.QUEEN, 4), (chess.ROOK, 2), (chess.BISHOP, 1), (chess.KNIGHT, 1)):
            for sq in board.pieces(pt, not color):
                dist = max(abs(chess.square_file(sq) - kf), abs(chess.square_rank(sq) - kr))
                if dist <= 2:
                    attack_weight += w * (3 - dist)

        score += s * int(phase * (shield - exposed - attack_weight * 6))

    return score


def _mobility(board: chess.Board) -> int:
    # more squares reachable = better piece activity
    # using attacks_mask so we never accidentally change board.turn
    w_mob = b_mob = 0
    for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
        for sq in board.pieces(pt, chess.WHITE):
            w_mob += bin(board.attacks_mask(sq) & ~board.occupied_co[chess.WHITE]).count('1')
        for sq in board.pieces(pt, chess.BLACK):
            b_mob += bin(board.attacks_mask(sq) & ~board.occupied_co[chess.BLACK]).count('1')
    return (w_mob - b_mob) * 3


def _outposts(board: chess.Board) -> int:
    # a knight on an advanced central square that can't be attacked by pawns is very strong
    score = 0
    bpawns = board.pieces(chess.PAWN, chess.BLACK)
    wpawns = board.pieces(chess.PAWN, chess.WHITE)

    for sq in board.pieces(chess.KNIGHT, chess.WHITE):
        f = chess.square_file(sq)
        adj = {f - 1, f + 1} & set(range(8))
        if not any(chess.square_file(p) in adj and chess.square_rank(p) > chess.square_rank(sq)
                   for p in bpawns):
            score += OUTPOST_BONUS[sq]

    for sq in board.pieces(chess.KNIGHT, chess.BLACK):
        f = chess.square_file(sq)
        r = chess.square_rank(sq)
        adj = {f - 1, f + 1} & set(range(8))
        if not any(chess.square_file(p) in adj and chess.square_rank(p) < r for p in wpawns):
            score -= OUTPOST_BONUS[_msq(sq)]

    return score


def _rook_bonuses(board: chess.Board) -> int:
    # rooks love open files with no pawns blocking them
    # two rooks defending each other are much harder to attack
    score = 0
    wpawns = board.pieces(chess.PAWN, chess.WHITE)
    bpawns = board.pieces(chess.PAWN, chess.BLACK)

    for color in (chess.WHITE, chess.BLACK):
        s      = 1 if color == chess.WHITE else -1
        mine   = wpawns if color == chess.WHITE else bpawns
        theirs = bpawns if color == chess.WHITE else wpawns
        rooks  = list(board.pieces(chess.ROOK, color))

        for sq in rooks:
            f = chess.square_file(sq)
            own = any(chess.square_file(p) == f for p in mine)
            opp = any(chess.square_file(p) == f for p in theirs)
            if not own and not opp:
                score += s * 25  # fully open file
            elif not own:
                score += s * 12  # semi-open file

        if len(rooks) == 2:
            r0, r1 = rooks
            if (chess.square_rank(r0) == chess.square_rank(r1) or
                    chess.square_file(r0) == chess.square_file(r1)):
                score += s * 10  # connected rooks

    return score


def _bishop_bonuses(board: chess.Board) -> int:
    # two bishops together cover both colors — very powerful in open positions
    # a bishop blocked by its own pawns is nearly useless
    score = 0
    if len(board.pieces(chess.BISHOP, chess.WHITE)) >= 2:
        score += 35
    if len(board.pieces(chess.BISHOP, chess.BLACK)) >= 2:
        score -= 35

    for color in (chess.WHITE, chess.BLACK):
        s = 1 if color == chess.WHITE else -1
        for bsq in board.pieces(chess.BISHOP, color):
            blight = (chess.square_file(bsq) + chess.square_rank(bsq)) % 2
            blocked = sum(1 for p in board.pieces(chess.PAWN, color)
                          if (chess.square_file(p) + chess.square_rank(p)) % 2 == blight)
            score -= s * blocked * 3

    return score


def _threats(board: chess.Board) -> int:
    # attacking undefended pieces is worth more — we might just win them for free
    score = 0
    for color in (chess.WHITE, chess.BLACK):
        s = 1 if color == chess.WHITE else -1
        for sq in chess.SQUARES:
            pc = board.piece_at(sq)
            if pc is None or pc.color == color:
                continue
            if board.is_attacked_by(color, sq):
                val = MG[pc.piece_type]
                defended = board.is_attacked_by(not color, sq)
                score += s * (val // 12 if defended else val // 6)
    return score


def _space(board: chess.Board, phase: float) -> int:
    # who controls the center? only worth tracking in the middlegame
    if phase < 0.2:
        return 0
    w_space = b_space = 0
    centre = {chess.C4, chess.D4, chess.E4, chess.F4,
              chess.C5, chess.D5, chess.E5, chess.F5}
    for sq in centre:
        r = chess.square_rank(sq)
        if r >= 4 and board.is_attacked_by(chess.WHITE, sq):
            w_space += 1
        if r <= 3 and board.is_attacked_by(chess.BLACK, sq):
            b_space += 1
    return int((w_space - b_space) * 4 * phase)


def _tempo(board: chess.Board) -> int:
    # small bonus for whoever's turn it is — they get to make the next threat
    return 10 if board.turn == chess.WHITE else -10


def evaluate(board: chess.Board) -> int:
    """Returns centipawns from white's perspective.
    Positive means white is better, negative means black is better."""
    if board.is_checkmate():
        return -900_000 if board.turn == chess.WHITE else 900_000
    if board.is_stalemate() or board.is_insufficient_material() or board.is_fifty_moves():
        return 0

    phase = game_phase(board)
    mg, eg = _pst_score(board)

    # blend MG and EG scores based on how many pieces are left
    score = int(mg * phase + eg * (1.0 - phase))

    score += _pawn_structure(board)
    score += _king_safety(board, phase)
    score += _mobility(board)
    score += _outposts(board)
    score += _rook_bonuses(board)
    score += _bishop_bonuses(board)
    score += _threats(board)
    score += _space(board, phase)
    score += _tempo(board)

    return score