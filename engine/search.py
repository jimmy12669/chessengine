import chess
import time
import math
from typing import Optional, Tuple, Dict, List

INF        = 32_000
MATE_SCORE = 30_000
DRAW       = 0
MAX_PLY    = 128

# piece values 
SEE_VAL = [0, 100, 320, 330, 500, 900, 20000]

TT_EXACT = 0
TT_ALPHA = 1
TT_BETA  = 2


def _all_attackers(board: chess.Board, sq: int, occ: int) -> int:
    mask = 0

    # pawns (note: attack directions are from the defender's perspective)
    mask |= chess.BB_PAWN_ATTACKS[chess.WHITE][sq] & board.pawns & board.occupied_co[chess.BLACK]
    mask |= chess.BB_PAWN_ATTACKS[chess.BLACK][sq] & board.pawns & board.occupied_co[chess.WHITE]

    mask |= chess.BB_KNIGHT_ATTACKS[sq] & board.knights
    mask |= chess.BB_KING_ATTACKS[sq] & board.kings

    diag = chess.BB_DIAG_ATTACKS[sq][chess.BB_DIAG_MASKS[sq] & occ]
    rank = chess.BB_RANK_ATTACKS[sq][chess.BB_RANK_MASKS[sq] & occ]
    file = chess.BB_FILE_ATTACKS[sq][chess.BB_FILE_MASKS[sq] & occ]

    mask |= diag & (board.bishops | board.queens)
    mask |= (rank | file) & (board.rooks | board.queens)

    return mask & occ


def see_ge(board: chess.Board, move: chess.Move, threshold: int) -> bool:
    """
    Returns True if SEE(move) >= threshold.
    Uses the full iterative capture-chain method.
    """
    to_sq  = move.to_square
    fr_sq  = move.from_square
    moving = board.piece_type_at(fr_sq)

    if not moving:
        return False

    captured = board.piece_type_at(to_sq)
    cap_val  = SEE_VAL[captured] if captured else 0

    # en passant: the captured pawn isn't on to_sq
    if not captured and board.is_en_passant(move):
        cap_val = SEE_VAL[chess.PAWN]

    value = cap_val - threshold
    if value < 0:
        return False

    value -= SEE_VAL[moving]
    if value >= 0:
        return True

    occ  = board.occupied ^ (1 << fr_sq)
    occ &= ~(1 << to_sq)
    occ |=  (1 << to_sq)

    attackers = _all_attackers(board, to_sq, occ)
    stm = not board.turn

    while True:
        stm_att = attackers & board.occupied_co[stm] & occ
        if not stm_att:
            break

        # pick the least valuable attacker
        for pt in range(1, 7):
            pt_mask = board.pieces_mask(pt, stm) & stm_att
            if pt_mask:
                break
        else:
            break

        stm   = not stm
        value = -value - 1 - SEE_VAL[pt]

        if value >= 0:
            if pt == chess.KING and (attackers & board.occupied_co[stm] & occ):
                stm = not stm
            break

        # remove the piece we just used, then reveal sliders behind it
        lsb  = pt_mask & (-pt_mask)
        occ ^= lsb

        diag = chess.BB_DIAG_ATTACKS[to_sq][chess.BB_DIAG_MASKS[to_sq] & occ]
        rank = chess.BB_RANK_ATTACKS[to_sq][chess.BB_RANK_MASKS[to_sq] & occ]
        file = chess.BB_FILE_ATTACKS[to_sq][chess.BB_FILE_MASKS[to_sq] & occ]

        attackers |= (diag & (board.bishops | board.queens))
        attackers |= ((rank | file) & (board.rooks | board.queens))

    return stm != board.turn


class PVTable:
    def __init__(self):
        self.pv:     List[List[Optional[chess.Move]]] = [[None] * MAX_PLY for _ in range(MAX_PLY)]
        self.length: List[int] = [0] * MAX_PLY

    def update(self, ply: int, move: chess.Move):
        self.pv[ply][ply] = move
        nxt = self.length[ply + 1]
        for i in range(ply + 1, nxt):
            self.pv[ply][i] = self.pv[ply + 1][i]
        self.length[ply] = nxt

    def get_pv(self) -> List[chess.Move]:
        return [m for m in self.pv[0][:self.length[0]] if m]


class TT:
    def __init__(self, mb: int = 128):
        self.slots = max(65536, (mb * 1024 * 1024) // 32)
        self.data: Dict[int, tuple] = {}

    @staticmethod
    def hash_board(board: chess.Board) -> int:
        return hash(board._transposition_key())

    def probe(self, key: int, depth: int, alpha: int, beta: int,
              ply: int) -> Tuple[Optional[int], Optional[chess.Move]]:
        e = self.data.get(key)
        if e is None:
            return None, None

        d, flag, score, move = e

        # adjust mate scores for current ply
        if score >= MATE_SCORE - MAX_PLY:
            score -= ply
        elif score <= -(MATE_SCORE - MAX_PLY):
            score += ply

        if d >= depth:
            if flag == TT_EXACT:
                return score, move
            if flag == TT_ALPHA and score <= alpha:
                return alpha, move
            if flag == TT_BETA and score >= beta:
                return beta, move

        return None, move

    def store(self, key: int, depth: int, flag: int, score: int,
              move: Optional[chess.Move], ply: int):
        if score >= MATE_SCORE - MAX_PLY:
            score += ply
        elif score <= -(MATE_SCORE - MAX_PLY):
            score -= ply

        e = self.data.get(key)
        if e is None or e[0] <= depth or flag == TT_EXACT:
            self.data[key] = (depth, flag, score, move)

        # evict oldest entry if we're over capacity
        if len(self.data) > self.slots:
            try:
                del self.data[next(iter(self.data))]
            except StopIteration:
                pass

    def best_move(self, key: int) -> Optional[chess.Move]:
        e = self.data.get(key)
        return e[3] if e else None

    def clear(self):
        self.data.clear()

    def hashfull(self) -> int:
        return min(1000, len(self.data) * 1000 // self.slots)


# precompute LMR reduction table
_LMR = [[0] * 64 for _ in range(64)]
for _d in range(1, 64):
    for _m in range(1, 64):
        _LMR[_d][_m] = max(0, round(0.67 + math.log(_d) * math.log(_m) / 2.0))


class TimeManager:
    def __init__(self):
        self.start      = 0.0
        self.soft_limit = 0.0
        self.hard_limit = 0.0

    def init(self, move_time: float = 0.0,
             remaining: float = None, increment: float = 0.0,
             movestogo: int = None):
        self.start = time.time()

        if remaining is not None:
            if movestogo and movestogo > 0:
                alloc = remaining / movestogo + increment * 0.8
            else:
                alloc = remaining / 25.0 + increment * 0.8

            # soft = 60% of budget, hard = 160% but capped to avoid flagging
            self.soft_limit = self.start + alloc * 0.60
            self.hard_limit = self.start + min(alloc * 1.6, remaining * 0.8 / 1000)
        else:
            self.soft_limit = self.start + move_time * 0.65
            self.hard_limit = self.start + move_time

    def elapsed(self) -> float:
        return time.time() - self.start

    def soft_expired(self) -> bool:
        return time.time() > self.soft_limit

    def hard_expired(self) -> bool:
        return time.time() > self.hard_limit


class Engine:
    def __init__(self, evaluator):
        self.ev  = evaluator
        self.tt  = TT(mb=128)
        self.tm  = TimeManager()
        self.pv  = PVTable()

        self.reset_heuristics()

        self.nodes     = 0
        self.sel_depth = 0
        self.stop      = False
        self.eval_stack: List[int] = [0] * MAX_PLY

    def reset_heuristics(self):
        # quiet move history indexed [color][from][to]
        self.qhist: List[List[List[int]]] = \
            [[[0] * 64 for _ in range(64)] for _ in range(2)]

        # capture history [color][from][to][captured piece type]
        self.chist: List[List[List[List[int]]]] = \
            [[[[0] * 7 for _ in range(64)] for _ in range(64)] for _ in range(2)]

        # continuation history [prev_pt][prev_to][this_pt][this_to]
        self.cont: List[List[List[List[int]]]] = \
            [[[[0] * 64 for _ in range(7)] for _ in range(64)] for _ in range(7)]

        # countermove table [from][to] -> best response move
        self.counter: List[List[Optional[chess.Move]]] = \
            [[None] * 64 for _ in range(64)]

        # killers [ply][0/1]
        self.killers: List[List[Optional[chess.Move]]] = \
            [[None, None] for _ in range(MAX_PLY)]

    @staticmethod
    def _grav(cur: int, delta: int, max_val: int = 16384) -> int:
        """gravity update — keeps history scores from growing unbounded"""
        return cur + delta - cur * abs(delta) // max_val

    def _bonus(self, depth: int) -> int:
        return min(2048, depth * depth + 2 * depth)

    def _update_qhist(self, color: int, fr: int, to: int, b: int):
        self.qhist[color][fr][to] = self._grav(self.qhist[color][fr][to], b)

    def _update_chist(self, color: int, fr: int, to: int, cap: int, b: int):
        self.chist[color][fr][to][cap] = self._grav(self.chist[color][fr][to][cap], b)

    def _update_cont(self, prev_pt: int, prev_to: int,
                     this_pt: int, this_to: int, b: int):
        v = self.cont[prev_pt][prev_to][this_pt][this_to]
        self.cont[prev_pt][prev_to][this_pt][this_to] = self._grav(v, b)

    def _store_killer(self, ply: int, move: chess.Move):
        k = self.killers[ply]
        if k[0] != move:
            k[1] = k[0]
            k[0] = move

    def _score_move(self, board: chess.Board, move: chess.Move,
                    tt_move: Optional[chess.Move], ply: int,
                    prev_move: Optional[chess.Move]) -> int:
        if move == tt_move:
            return 20_000_000

        color = int(board.turn)

        if board.is_capture(move):
            cap_pt = board.piece_type_at(move.to_square) or chess.PAWN
            good   = see_ge(board, move, 0)
            base   = 10_000_000 if good else -10_000_000
            base  += SEE_VAL[cap_pt] * 8
            base  += self.chist[color][move.from_square][move.to_square][cap_pt] // 8
            return base

        if move.promotion == chess.QUEEN:
            return 9_000_000

        # killer moves
        k = self.killers[ply]
        if move == k[0]: return 8_000_000
        if move == k[1]: return 7_900_000

        if prev_move:
            cm = self.counter[prev_move.from_square][prev_move.to_square]
            if move == cm:
                return 7_800_000

        score = self.qhist[color][move.from_square][move.to_square]

        if prev_move:
            prev_pt = board.piece_type_at(prev_move.to_square) or 1
            this_pt = board.piece_type_at(move.from_square) or 1
            score  += self.cont[prev_pt][prev_move.to_square][this_pt][move.to_square]

        return score

    def _order(self, board: chess.Board, moves,
               tt_move: Optional[chess.Move], ply: int,
               prev_move: Optional[chess.Move]) -> List[chess.Move]:
        scored = [(self._score_move(board, m, tt_move, ply, prev_move), m) for m in moves]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored]

    def _do_cutoff_update(self, board: chess.Board, move: chess.Move,
                          depth: int, ply: int, prev_move: Optional[chess.Move],
                          quiets_failed: List[chess.Move],
                          caps_failed: List[chess.Move]):
        color = int(board.turn)
        b     = self._bonus(depth)

        if board.is_capture(move):
            cap_pt = board.piece_type_at(move.to_square) or chess.PAWN
            self._update_chist(color, move.from_square, move.to_square, cap_pt, b)
            for m in caps_failed:
                ct = board.piece_type_at(m.to_square) or chess.PAWN
                self._update_chist(color, m.from_square, m.to_square, ct, -b)
        else:
            self._store_killer(ply, move)
            self._update_qhist(color, move.from_square, move.to_square, b)
            if prev_move:
                self.counter[prev_move.from_square][prev_move.to_square] = move
                prev_pt = board.piece_type_at(prev_move.to_square) or 1
                this_pt = board.piece_type_at(move.from_square) or 1
                self._update_cont(prev_pt, prev_move.to_square, this_pt, move.to_square, b)
            for m in quiets_failed:
                self._update_qhist(color, m.from_square, m.to_square, -b)

    def qsearch(self, board: chess.Board, alpha: int, beta: int, ply: int) -> int:
        self.nodes += 1
        self.sel_depth = max(self.sel_depth, ply)

        # draw checks
        if board.is_repetition(2) or board.is_fifty_moves(): return DRAW
        if board.is_insufficient_material():                  return DRAW
        if board.is_stalemate():                              return DRAW

        in_check = board.is_check()

        if not in_check:
            stand_pat = self.ev.score(board)
            if stand_pat >= beta:
                return stand_pat
            if stand_pat > alpha:
                alpha = stand_pat
            if stand_pat + 1050 < alpha:
                return alpha  # delta pruning

        if in_check:
            moves = list(board.generate_pseudo_legal_moves())
        else:
            moves = list(board.generate_pseudo_legal_captures())

        for move in moves:
            if not board.is_legal(move):
                continue
            if not in_check and not see_ge(board, move, -50):
                continue

            self.ev.push(board, move)
            board.push(move)
            self.ev.after_push(board)

            score = -self.qsearch(board, -beta, -alpha, ply + 1)

            board.pop()
            self.ev.pop()

            if score >= beta:
                return score
            if score > alpha:
                alpha = score

        # if we're in check and had no legal moves, it's checkmate
        if in_check and not any(board.is_legal(m) for m in board.generate_pseudo_legal_moves()):
            return -(MATE_SCORE - ply)

        return alpha

    def _search(self, board: chess.Board, depth: int, alpha: int, beta: int,
                ply: int, cut_node: bool,
                prev_move: Optional[chess.Move] = None,
                skip_move: Optional[chess.Move] = None) -> int:

        self.nodes += 1

        # check the clock every 2048 nodes
        if self.nodes & 2047 == 0:
            if self.tm.hard_expired():
                self.stop = True
                return 0

        if self.stop:
            return 0

        pv_node = alpha + 1 < beta
        root    = ply == 0

        if not root:
            if board.is_repetition(2) or board.is_fifty_moves(): return DRAW
            if board.is_insufficient_material():                  return DRAW

        in_check = board.is_check()
        if in_check:
            depth = max(depth, 1)  # check extension

        if depth <= 0:
            return self.qsearch(board, alpha, beta, ply)

        # mate distance pruning
        if not root:
            alpha = max(alpha, -(MATE_SCORE - ply))
            beta  = min(beta, MATE_SCORE - ply - 1)
            if alpha >= beta:
                return alpha

        # TT probe
        h = TT.hash_board(board)
        tt_score, tt_move = self.tt.probe(h, depth, alpha, beta, ply)
        if tt_score is not None and not root and skip_move is None and not pv_node:
            return tt_score

        # static eval
        if in_check:
            static_eval = -(MATE_SCORE - ply)
        else:
            static_eval = self.ev.score(board)
            # if TT gives us a better bound, use it
            if tt_score is not None and tt_move is not None:
                e = self.tt.data.get(h, (0, TT_EXACT, 0, None))
                if (e[1] == TT_BETA  and tt_score > static_eval) or \
                   (e[1] == TT_ALPHA and tt_score < static_eval):
                    static_eval = tt_score

        if ply < MAX_PLY:
            self.eval_stack[ply] = static_eval

        improving = (not in_check and ply >= 2
                     and static_eval > self.eval_stack[max(0, ply - 2)])

        # razoring
        if (not pv_node and not in_check and skip_move is None
                and depth <= 3 and static_eval + 350 * depth <= alpha):
            q = self.qsearch(board, alpha - 1, alpha, ply)
            if q < alpha:
                return q

        # reverse futility pruning
        if (not pv_node and not in_check and skip_move is None
                and depth <= 9 and abs(beta) < MATE_SCORE - MAX_PLY
                and static_eval - 75 * (depth - improving) >= beta):
            return (static_eval + beta) // 2

        # null move pruning
        if (not pv_node and not in_check and skip_move is None
                and depth >= 3 and static_eval >= beta
                and any(board.pieces(pt, board.turn)
                        for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN))):

            R = 3 + depth // 3 + min(3, (static_eval - beta) // 150)

            self.ev.push(board, chess.Move.null())
            board.push(chess.Move.null())
            self.ev.after_push(board)

            null_s = -self._search(board, depth - R - 1, -beta, -beta + 1,
                                   ply + 1, not cut_node)

            board.pop()
            self.ev.pop()

            if self.stop: return 0

            if null_s >= beta:
                if null_s >= MATE_SCORE - MAX_PLY: return beta
                if depth < 12: return null_s
                # verification search at shallow depth
                vs = self._search(board, depth - R - 1, beta - 1, beta,
                                  ply, cut_node, prev_move)
                if vs >= beta: return vs

        # probcut
        if (not pv_node and not in_check and depth >= 5 and skip_move is None
                and abs(beta) < MATE_SCORE - MAX_PLY):
            pc_beta = beta + 180
            for move in board.generate_pseudo_legal_captures():
                if not board.is_legal(move): continue
                if not see_ge(board, move, pc_beta - static_eval): continue

                self.ev.push(board, move)
                board.push(move)
                self.ev.after_push(board)

                pcs = -self.qsearch(board, -pc_beta, -pc_beta + 1, ply + 1)
                if pcs >= pc_beta:
                    pcs = -self._search(board, depth - 4, -pc_beta, -pc_beta + 1,
                                        ply + 1, not cut_node, move)

                board.pop()
                self.ev.pop()

                if self.stop: return 0
                if pcs >= pc_beta: return pcs

        # IIR: if we don't have a tt move in a pv node, do a quick pre-search
        if tt_move is None and pv_node and depth >= 5:
            self._search(board, depth - 2, alpha, beta, ply, cut_node, prev_move)
            if self.stop: return 0
            _, tt_move = self.tt.probe(h, 0, alpha, beta, ply)

        legal = list(board.legal_moves)
        if not legal:
            return -(MATE_SCORE - ply) if in_check else DRAW

        moves      = self._order(board, legal, tt_move, ply, prev_move)
        best_score = -INF
        best_move  = None
        orig_alpha = alpha
        moves_done = 0
        quiets_failed: List[chess.Move] = []
        caps_failed:   List[chess.Move] = []
        self.pv.length[ply] = ply

        for move in moves:
            if move == skip_move:
                continue

            is_cap   = board.is_capture(move)
            is_promo = bool(move.promotion)
            is_quiet = not is_cap and not is_promo

            # pruning for later moves
            if not root and moves_done > 0 and abs(alpha) < MATE_SCORE - MAX_PLY:

                # futility pruning
                if (not pv_node and not in_check and is_quiet and depth <= 7
                        and static_eval + 80 + 50 * depth <= alpha):
                    continue

                # late move pruning
                lmp_thresh = 3 + depth * depth // (2 - improving)
                if (not pv_node and not in_check and is_quiet
                        and depth <= 8 and moves_done >= lmp_thresh):
                    continue

                # SEE pruning
                see_thresh = (-50 * depth if is_quiet else -SEE_VAL[min(5, depth)] * depth)
                if depth <= 10 and not see_ge(board, move, see_thresh):
                    continue

            extension = 0

            # singular extension
            if (move == tt_move and not root and skip_move is None
                    and depth >= 7 and tt_score is not None
                    and abs(tt_score) < MATE_SCORE - MAX_PLY):
                tt_entry = self.tt.data.get(h)
                if tt_entry and tt_entry[0] >= depth - 3:
                    s_beta  = tt_score - depth * 2
                    s_score = self._search(board, (depth - 1) // 2, s_beta - 1, s_beta,
                                           ply, cut_node, prev_move, skip_move=move)
                    if self.stop: return 0
                    if s_score < s_beta:
                        extension = 2 if (not pv_node and s_score < s_beta - 30) else 1
                    elif s_score >= beta:
                        return s_beta  # multi-cut shortcut

            self.ev.push(board, move)
            board.push(move)
            self.ev.after_push(board)

            gives_check = board.is_check()
            new_depth   = depth - 1 + extension

            if moves_done >= 2 and depth >= 3 and not gives_check and not extension:
                r = _LMR[min(63, depth)][min(63, moves_done)]

                if not pv_node:    r += 1
                if cut_node:       r += 1
                if not improving:  r += 1
                if is_cap:         r -= 1
                if move == tt_move: r -= 1

                r = max(0, min(r, new_depth - 1))
                lmr_d = new_depth - r

                score = -self._search(board, lmr_d, -alpha - 1, -alpha,
                                      ply + 1, True, move)

                # if it beat alpha with reductions, research at full depth
                if not self.stop and score > alpha and r > 0:
                    score = -self._search(board, new_depth, -alpha - 1, -alpha,
                                          ply + 1, not cut_node, move)

            elif not pv_node or moves_done > 0:
                score = -self._search(board, new_depth, -alpha - 1, -alpha,
                                      ply + 1, not cut_node, move)
            else:
                score = -INF

            # full window search for pv nodes
            if pv_node and (moves_done == 0 or (score > alpha and not self.stop)):
                score = -self._search(board, new_depth, -beta, -alpha,
                                      ply + 1, False, move)

            board.pop()
            self.ev.pop()

            if self.stop: return 0

            moves_done += 1
            if is_cap:     caps_failed.append(move)
            elif is_quiet: quiets_failed.append(move)

            if score > best_score:
                best_score = score
                best_move  = move

            if score > alpha:
                alpha = score
                if pv_node:
                    self.pv.update(ply, move)

            if alpha >= beta:
                self._do_cutoff_update(
                    board, move, depth, ply, prev_move,
                    quiets_failed[:-1] if is_quiet else quiets_failed,
                    caps_failed[:-1]   if is_cap   else caps_failed)
                break

        if not self.stop:
            flag = (TT_EXACT if orig_alpha < best_score < beta
                    else (TT_BETA if best_score >= beta else TT_ALPHA))
            self.tt.store(h, depth, flag, best_score, best_move, ply)

        return best_score

    def go(self, board: chess.Board,
           move_time: float = 5.0,
           max_depth: int   = 64,
           wtime: int       = None,
           btime: int       = None,
           winc:  int       = 0,
           binc:  int       = 0,
           movestogo: int   = None) -> Tuple[Optional[chess.Move], int, int]:

        if wtime is not None or btime is not None:
            remaining = (wtime if board.turn == chess.WHITE else btime) or 30_000
            inc       = (winc  if board.turn == chess.WHITE else binc) or 0
            self.tm.init(remaining=remaining, increment=inc, movestogo=movestogo)
        else:
            self.tm.init(move_time=move_time)

        self.stop       = False
        self.nodes      = 0
        self.sel_depth  = 0
        self.eval_stack = [0] * MAX_PLY
        self.ev.prepare(board)

        best_move:  Optional[chess.Move] = None
        best_score: int = 0
        depth_done: int = 0
        prev_score: int = 0
        asp_delta:  int = 15

        for depth in range(1, max_depth + 1):
            if self.tm.soft_expired() and depth > 1:
                break

            self.sel_depth    = 0
            self.pv.length[0] = 0

            if depth >= 5:
                a, b = prev_score - asp_delta, prev_score + asp_delta
            else:
                a, b = -INF, INF

            fails = 0
            while True:
                score = self._search(board, depth, a, b, 0, False)
                if self.stop: break

                if score <= a:
                    b = (a + b) // 2
                    a -= asp_delta
                    fails += 1
                elif score >= b:
                    b += asp_delta
                    fails += 1
                else:
                    break

                asp_delta = int(asp_delta * (1.5 + fails * 0.2))
                if a < -MATE_SCORE: a = -INF
                if b >  MATE_SCORE: b =  INF

            if self.stop: break

            prev_score = score
            asp_delta  = max(12, abs(score - prev_score) // 2 + 8)
            depth_done = depth

            pv = self.pv.get_pv()
            if pv:
                best_move = pv[0]
            else:
                bm = self.tt.best_move(TT.hash_board(board))
                if bm: best_move = bm

            best_score = score
            elapsed    = self.tm.elapsed()
            nps        = int(self.nodes / max(elapsed, 1e-9))
            pv_str     = ' '.join(m.uci() for m in pv[:12]) or str(best_move)

            if abs(score) >= MATE_SCORE - MAX_PLY:
                m2m = (MATE_SCORE - abs(score) + 1) // 2
                sc  = f"mate {m2m if score > 0 else -m2m}"
            else:
                sc = f"cp {score}"

            print(f"info depth {depth} seldepth {self.sel_depth} "
                  f"score {sc} nodes {self.nodes} nps {nps} "
                  f"hashfull {self.tt.hashfull()} "
                  f"time {int(elapsed * 1000)} pv {pv_str}", flush=True)

            if abs(score) >= MATE_SCORE - MAX_PLY:
                break

        return best_move, best_score, depth_done