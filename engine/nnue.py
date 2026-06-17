import chess
import numpy as np
import os
import ctypes
from typing import Optional, Tuple, List, NamedTuple

try:
    import torch
    import torch.nn as nn
    _TORCH = True
except ImportError:
    _TORCH = False

HALFKP  = 64 * 64 * 12
L1      = 1024
L2      = 64
L3      = 32
L1_HALF = L1 // 2
CP_SCALE = 200.0

_LIB = None


def _load_lib():
    global _LIB
    if _LIB is not None:
        return _LIB

    here = os.path.dirname(os.path.abspath(__file__))
    name = 'nnue.dll' if os.name == 'nt' else 'nnue.so'
    so = os.path.join(here, name)
    if not os.path.exists(so):
        return None

    lib = ctypes.CDLL(so)
    fp  = ctypes.POINTER(ctypes.c_float)

    lib.accumulator_init.restype  = None
    lib.accumulator_init.argtypes = [fp, fp, fp]

    lib.accumulator_add.restype  = None
    lib.accumulator_add.argtypes = [fp, ctypes.c_int, fp]

    lib.accumulator_sub.restype  = None
    lib.accumulator_sub.argtypes = [fp, ctypes.c_int, fp]

    lib.accumulator_add_sub.restype  = None
    lib.accumulator_add_sub.argtypes = [fp, ctypes.c_int, ctypes.c_int, fp]

    lib.forward.restype  = ctypes.c_float
    lib.forward.argtypes = [fp, fp, fp, fp, fp, fp, fp, ctypes.c_float, ctypes.c_int]

    _LIB = lib
    return lib


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
        b.append((bk ^ 56) * 768 + (sq ^ 56) * 12 + p)
    return np.array(w, dtype=np.int32), np.array(b, dtype=np.int32)


def _ptr(arr: np.ndarray):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


if _TORCH:
    class NNUE(nn.Module):
        def __init__(self):
            super().__init__()
            self.ft  = nn.Linear(HALFKP, L1_HALF)
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

        def forward(self, w, b):
            import torch
            w_acc = torch.clamp(self.ft(w), 0.0, 1.0)
            b_acc = torch.clamp(self.ft(b), 0.0, 1.0)
            x = torch.cat([w_acc, b_acc], dim=-1)
            x = torch.clamp(self.l1(x), 0.0, 1.0)
            x = torch.clamp(self.l2(x), 0.0, 1.0)
            return self.out(x).squeeze(-1) * CP_SCALE

        def save(self, path: str):
            import torch
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            torch.save({'state': self.state_dict()}, path)

        @classmethod
        def load(cls, path: str) -> 'NNUE':
            import torch
            ck = torch.load(path, map_location='cpu', weights_only=False)
            m  = cls()
            sd = ck['state'] if isinstance(ck, dict) and 'state' in ck else ck
            m.load_state_dict(sd)
            m.eval()
            return m


class _Entry(NamedTuple):
    w_acc: np.ndarray
    b_acc: np.ndarray
    needs_refresh: bool


class NNUEInference:
    def __init__(self, npz_path: str):
        d = np.load(npz_path)

        self.ft_W  = np.ascontiguousarray(d['ft_W'].astype(np.float32).T)
        self.ft_b  = np.ascontiguousarray(d['ft_b'].astype(np.float32))
        self.l1_W  = np.ascontiguousarray(d['l1_W'].astype(np.float32))
        self.l1_b  = np.ascontiguousarray(d['l1_b'].astype(np.float32))
        self.l2_W  = np.ascontiguousarray(d['l2_W'].astype(np.float32))
        self.l2_b  = np.ascontiguousarray(d['l2_b'].astype(np.float32))
        self.out_W = np.ascontiguousarray(d['out_W'].astype(np.float32).ravel())
        self.out_b = float(d['out_b'].ravel()[0])

        self._lib  = _load_lib()
        self.w_acc = np.ascontiguousarray(self.ft_b.copy())
        self.b_acc = np.ascontiguousarray(self.ft_b.copy())

        if not self._lib:
            self.ft_cols = [np.ascontiguousarray(self.ft_W[i]) for i in range(self.ft_W.shape[0])]
            self._buf    = np.empty(L1, dtype=np.float32)

        self._stack: List[_Entry] = []

    def full_refresh(self, board: chess.Board):
        wi, bi = halfkp_indices(board)

        if self._lib:
            lib = self._lib
            ft  = _ptr(self.ft_W)
            ftb = _ptr(self.ft_b)
            w   = _ptr(self.w_acc)
            b   = _ptr(self.b_acc)
            lib.accumulator_init(ftb, w, b)
            for i in wi: lib.accumulator_add(ft, int(i), w)
            for i in bi: lib.accumulator_add(ft, int(i), b)
        else:
            self.w_acc = self.ft_b.copy()
            self.b_acc = self.ft_b.copy()
            for i in wi: self.w_acc += self.ft_cols[i]
            for i in bi: self.b_acc += self.ft_cols[i]

    def push(self, board: chess.Board, move: chess.Move):
        moving = board.piece_at(move.from_square)
        needs_refresh = (
            moving is None or
            moving.piece_type == chess.KING or
            board.is_castling(move)
        )
        self._stack.append(_Entry(self.w_acc.copy(), self.b_acc.copy(), needs_refresh))
        if not needs_refresh:
            self._apply_delta(board, move, moving)

    def _apply_delta(self, board: chess.Board, move: chess.Move, moving):
        wk    = board.king(chess.WHITE)
        bk    = board.king(chess.BLACK)
        color = moving.color

        p_from = _pidx(moving.piece_type, color)
        p_to   = _pidx(move.promotion if move.promotion else moving.piece_type, color)

        w_from = wk * 768 + move.from_square * 12 + p_from
        w_to   = wk * 768 + move.to_square   * 12 + p_to
        b_from = (bk ^ 56) * 768 + (move.from_square ^ 56) * 12 + p_from
        b_to   = (bk ^ 56) * 768 + (move.to_square   ^ 56) * 12 + p_to

        if self._lib:
            ft = _ptr(self.ft_W)
            w  = _ptr(self.w_acc)
            b  = _ptr(self.b_acc)
            self._lib.accumulator_add_sub(ft, w_to,   w_from, w)
            self._lib.accumulator_add_sub(ft, b_to,   b_from, b)
        else:
            cols = self.ft_cols
            np.add(self.w_acc,      cols[w_to],   out=self.w_acc)
            np.subtract(self.w_acc, cols[w_from], out=self.w_acc)
            np.add(self.b_acc,      cols[b_to],   out=self.b_acc)
            np.subtract(self.b_acc, cols[b_from], out=self.b_acc)

        if board.is_capture(move):
            if board.is_en_passant(move):
                cap_sq  = move.to_square + (-8 if color == chess.WHITE else 8)
                cap_pt  = chess.PAWN
                cap_col = not color
            else:
                cap = board.piece_at(move.to_square)
                if cap is None:
                    return
                cap_sq  = move.to_square
                cap_pt  = cap.piece_type
                cap_col = cap.color

            cp = _pidx(cap_pt, cap_col)
            wc = wk * 768 + cap_sq * 12 + cp
            bc = (bk ^ 56) * 768 + (cap_sq ^ 56) * 12 + cp

            if self._lib:
                self._lib.accumulator_sub(_ptr(self.ft_W), wc, _ptr(self.w_acc))
                self._lib.accumulator_sub(_ptr(self.ft_W), bc, _ptr(self.b_acc))
            else:
                np.subtract(self.w_acc, self.ft_cols[wc], out=self.w_acc)
                np.subtract(self.b_acc, self.ft_cols[bc], out=self.b_acc)

    def needs_full_refresh(self) -> bool:
        return bool(self._stack) and self._stack[-1].needs_refresh

    def after_push(self, board: chess.Board):
        if self.needs_full_refresh():
            self.full_refresh(board)

    def pop(self):
        e = self._stack.pop()
        self.w_acc = e.w_acc
        self.b_acc = e.b_acc

    def evaluate(self, white_to_move: bool) -> int:
        if self._lib:
            return int(self._lib.forward(
                _ptr(self.w_acc), _ptr(self.b_acc),
                _ptr(self.l1_W),  _ptr(self.l1_b),
                _ptr(self.l2_W),  _ptr(self.l2_b),
                _ptr(self.out_W), ctypes.c_float(self.out_b),
                ctypes.c_int(1 if white_to_move else 0),
            ))

        half = L1_HALF
        buf  = self._buf
        if white_to_move:
            np.clip(self.w_acc, 0.0, 1.0, out=buf[:half])
            np.clip(self.b_acc, 0.0, 1.0, out=buf[half:])
        else:
            np.clip(self.b_acc, 0.0, 1.0, out=buf[:half])
            np.clip(self.w_acc, 0.0, 1.0, out=buf[half:])
        x = (self.l1_W @ buf + self.l1_b).clip(0.0, 1.0)
        x = (self.l2_W @ x   + self.l2_b).clip(0.0, 1.0)
        return int((self.out_W @ x + self.out_b) * CP_SCALE)


class Evaluator:
    def __init__(self, model_path: Optional[str] = None):
        self._inf: Optional[NNUEInference] = None
        self.use_nnue = False

        if model_path:
            npz = model_path if model_path.endswith('.npz') else \
                  os.path.join(os.path.dirname(model_path), 'weights.npz')

            if os.path.exists(npz):
                try:
                    self._inf  = NNUEInference(npz)
                    self.use_nnue = True
                    backend    = "C backend" if self._inf._lib else "NumPy fallback"
                    print(f"[NNUE] loaded: {npz} ({backend})")
                except Exception as e:
                    print(f"[NNUE] failed to load ({e}), falling back to classical eval")

            elif _TORCH and os.path.exists(model_path):
                try:
                    model   = NNUE.load(model_path)
                    sd      = model.state_dict()
                    npz_out = os.path.join(os.path.dirname(model_path), 'weights.npz')
                    np.savez(npz_out,
                        ft_W  = sd['ft.weight'].numpy(),
                        ft_b  = sd['ft.bias'].numpy(),
                        l1_W  = sd['l1.weight'].numpy(),
                        l1_b  = sd['l1.bias'].numpy(),
                        l2_W  = sd['l2.weight'].numpy(),
                        l2_b  = sd['l2.bias'].numpy(),
                        out_W = sd['out.weight'].numpy(),
                        out_b = sd['out.bias'].numpy(),
                    )
                    self._inf = NNUEInference(npz_out)
                    self.use_nnue = True
                    backend   = "C backend" if self._inf._lib else "NumPy fallback"
                    print(f"[NNUE] loaded from pt ({backend})")
                except Exception as e:
                    print(f"[NNUE] failed to load ({e}), falling back to classical eval")
            else:
                print("[Eval] no model found, using classical eval")
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