import chess
import numpy as np
import torch
import torch.nn as nn
import os
from typing import Optional, Tuple, List, NamedTuple

# halfkp input size (king sq * 768 + piece_sq * 12 + piece_type)
HALFKP = 64 * 64 * 12

L1 = 1024
L2 = 64
L3 = 32

CP_SCALE = 200.0 


def _pidx(pt: int, color: bool) -> int:
    return (pt - 1) + (0 if color == chess.WHITE else 6)


def halfkp_indices(board: chess.Board) -> Tuple[np.ndarray, np.ndarray]:
    wk = board.king(chess.WHITE)
    bk = board.king(chess.BLACK)

    w, b = [], []

    for sq in chess.SQUARES:
        pc = board.piece_at(sq)
        if pc is None or pc.piece_type == chess.KING:
            continue

        p = _pidx(pc.piece_type, pc.color)
        w.append(wk * 768 + sq * 12 + p)
        b.append((bk ^ 56) * 768 + (sq ^ 56) * 12 + p)  # mirror for black

    return np.array(w, dtype=np.int32), np.array(b, dtype=np.int32)


class NNUE(nn.Module):

    def __init__(self):
        super().__init__()

        self.ft  = nn.Linear(HALFKP, L1 // 2)
        self.l1  = nn.Linear(L1, L2)
        self.l2  = nn.Linear(L2, L3)
        self.out = nn.Linear(L3, 1)

        self._init()

    def _init(self):
        nn.init.kaiming_normal_(self.ft.weight, nonlinearity='relu')
        nn.init.zeros_(self.ft.bias)

        for layer in (self.l1, self.l2, self.out):
            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
            nn.init.zeros_(layer.bias)

    def forward(self, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        w_acc = torch.clamp(self.ft(w), 0.0, 1.0)
        b_acc = torch.clamp(self.ft(b), 0.0, 1.0)

        x = torch.cat([w_acc, b_acc], dim=-1)
        x = torch.clamp(self.l1(x), 0.0, 1.0)
        x = torch.clamp(self.l2(x), 0.0, 1.0)

        return self.out(x).squeeze(-1) * CP_SCALE

    def save(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save({'state': self.state_dict()}, path)

    @classmethod
    def load(cls, path: str) -> 'NNUE':
        ck = torch.load(path, map_location='cpu', weights_only=False)
        m = cls()

        sd = ck['state'] if isinstance(ck, dict) and 'state' in ck else ck

        m.load_state_dict(sd)
        m.eval()
        return m


def clamp_score(x: float) -> float:
    # anything past 30 pawns is winning
    return max(-3000.0, min(3000.0, float(x)))


class _Entry(NamedTuple):
    w_acc: torch.Tensor
    b_acc: torch.Tensor
    needs_refresh: bool


class NNUEInference:
    """Handles fast inference with incremental accumulator updates so we
    dont have to recompute everything on every move."""

    def __init__(self, model: NNUE):
        sd = model.state_dict()

        self.ft_W  = sd['ft.weight']   # shape: (L1//2, HALFKP)
        self.ft_b  = sd['ft.bias']     # shape: (L1//2,)
        self.l1_W  = sd['l1.weight']
        self.l1_b  = sd['l1.bias']
        self.l2_W  = sd['l2.weight']
        self.l2_b  = sd['l2.bias']
        self.out_W = sd['out.weight']
        self.out_b = sd['out.bias']

        # start accumulators at the bias
        self.w_acc = self.ft_b.clone()
        self.b_acc = self.ft_b.clone()

        self._stack: List[_Entry] = []

    def full_refresh(self, board: chess.Board):
        """recompute accumulators from scratch for the given position"""
        wi, bi = halfkp_indices(board)

        self.w_acc = self.ft_b.clone()
        self.b_acc = self.ft_b.clone()

        if len(wi):
            self.w_acc = self.w_acc + self.ft_W[:, wi].sum(dim=1)
        if len(bi):
            self.b_acc = self.b_acc + self.ft_W[:, bi].sum(dim=1)

    def push(self, board: chess.Board, move: chess.Move):
        """save accumulator state before making a move (call BEFORE board.push)"""
        moving = board.piece_at(move.from_square)

        # king moves invalidate all features so we need a full refresh after
        needs_refresh = (
            moving is None or
            moving.piece_type == chess.KING or
            board.is_castling(move)
        )

        self._stack.append(_Entry(self.w_acc.clone(), self.b_acc.clone(), needs_refresh))

        if not needs_refresh:
            self._apply_delta(board, move, moving)

    def _apply_delta(self, board: chess.Board, move: chess.Move, moving):
        """incrementally update accumulators for a non-king move"""
        wk = board.king(chess.WHITE)
        bk = board.king(chess.BLACK)
        color = moving.color

        p_from = _pidx(moving.piece_type, color)
        p_to   = _pidx(move.promotion if move.promotion else moving.piece_type, color)

        # compute feature indices for the from/to squares
        w_from = wk * 768 + move.from_square * 12 + p_from
        w_to   = wk * 768 + move.to_square   * 12 + p_to
        b_from = (bk ^ 56) * 768 + (move.from_square ^ 56) * 12 + p_from
        b_to   = (bk ^ 56) * 768 + (move.to_square   ^ 56) * 12 + p_to

        self.w_acc = self.w_acc - self.ft_W[:, w_from] + self.ft_W[:, w_to]
        self.b_acc = self.b_acc - self.ft_W[:, b_from] + self.ft_W[:, b_to]

        if board.is_capture(move):
            if board.is_en_passant(move):
                cap_sq  = move.to_square + (-8 if color == chess.WHITE else 8)
                cap_pt  = chess.PAWN
                cap_col = not color
            else:
                cap = board.piece_at(move.to_square)
                if cap is None:
                    return  # shouldnt happen but just in case
                cap_sq  = move.to_square
                cap_pt  = cap.piece_type
                cap_col = cap.color

            cp = _pidx(cap_pt, cap_col)
            wc = wk * 768 + cap_sq * 12 + cp
            bc = (bk ^ 56) * 768 + (cap_sq ^ 56) * 12 + cp

            self.w_acc = self.w_acc - self.ft_W[:, wc]
            self.b_acc = self.b_acc - self.ft_W[:, bc]

    def needs_full_refresh(self) -> bool:
        return bool(self._stack) and self._stack[-1].needs_refresh

    def after_push(self, board: chess.Board):
        """call this AFTER board.push if the move was a king move"""
        if self.needs_full_refresh():
            self.full_refresh(board)

    def pop(self):
        e = self._stack.pop()
        self.w_acc = e.w_acc
        self.b_acc = e.b_acc

    def evaluate(self, white_to_move: bool) -> int:
        """run the network forward from current accumulators, returns centipawns from side-to-move perspective"""
        w = torch.clamp(self.w_acc, 0.0, 1.0)
        b = torch.clamp(self.b_acc, 0.0, 1.0)

        # order depends on who's to move
        x = torch.cat([w, b] if white_to_move else [b, w])

        x = torch.clamp(self.l1_W @ x + self.l1_b, 0.0, 1.0)
        x = torch.clamp(self.l2_W @ x + self.l2_b, 0.0, 1.0)

        return int(((self.out_W @ x) + self.out_b).item() * CP_SCALE)


class Evaluator:

    def __init__(self, model_path: Optional[str] = None):
        self._inf: Optional[NNUEInference] = None
        self.use_nnue = False

        if model_path and os.path.exists(model_path):
            try:
                model = NNUE.load(model_path)
                self._inf = NNUEInference(model)
                self.use_nnue = True
                print(f"[NNUE] loaded: {model_path}")
            except Exception as e:
                print(f"[NNUE] failed to load ({e}), falling back to classical eval")
        else:
            print("[Eval] no model found, using classical eval")

    def prepare(self, board: chess.Board):
        if self.use_nnue:
            self._inf.full_refresh(board)

    def push(self, board: chess.Board, move: chess.Move):
        if self.use_nnue:
            self._inf.push(board, move)

    def after_push(self, board: chess.Board):
        if self.use_nnue:
            self._inf.after_push(board)

    def pop(self):
        if self.use_nnue:
            self._inf.pop()

    def score(self, board: chess.Board) -> int:
        if self.use_nnue:
            return self._inf.evaluate(board.turn == chess.WHITE)

        from engine.evaluate import evaluate
        v = evaluate(board)
        return v if board.turn == chess.WHITE else -v