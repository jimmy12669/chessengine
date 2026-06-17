import argparse, os, sys, time, random, math, copy
import chess
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from engine.nnue     import NNUE, HALFKP, halfkp_indices, Evaluator, CP_SCALE
from engine.evaluate import evaluate as classical_eval
from engine.search   import Engine

DATA_DIR  = os.path.join(ROOT, 'data')
MODEL_DIR = os.path.join(ROOT, 'models')
os.makedirs(DATA_DIR,  exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

MAX_SCORE      = 3000
FILTER_SCORE   = 800
MIN_MOVE       = 6
MAX_MOVE       = 120


def clamp_score(score: float) -> float:
    return max(-MAX_SCORE, min(MAX_SCORE, float(score)))


def _bar(done: int, total: int, width: int = 40, prefix: str = ''):
    pct  = done / max(total, 1)
    fill = int(width * pct)
    bar  = '█' * fill + '░' * (width - fill)
    print(f'\r{prefix}|{bar}| {done}/{total} ({pct*100:.1f}%)', end='', flush=True)


def _bar_done():
    print()


def _is_quiet(board: chess.Board) -> bool:
    if board.is_check():
        return False
    if any(board.is_capture(m) for m in board.legal_moves):
        captures = [m for m in board.legal_moves if board.is_capture(m)]
        if len(captures) > 3:
            return False
    return True


class PositionDataset(Dataset):
    def __init__(self, files, max_positions: int = None):
        seen = {}

        if isinstance(files, str):
            files = [files]

        skipped = 0
        for path in files:
            if not os.path.exists(path):
                print(f"  [Warning] data file not found: {path}")
                continue
            with open(path, encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or '|' not in line:
                        continue
                    fen, sc = line.rsplit('|', 1)
                    try:
                        score = float(sc)
                        fen   = fen.strip()

                        if abs(score) > FILTER_SCORE:
                            skipped += 1
                            continue

                        board = chess.Board(fen)

                        if board.fullmove_number < MIN_MOVE or board.fullmove_number > MAX_MOVE:
                            skipped += 1
                            continue

                        if board.is_check():
                            skipped += 1
                            continue

                        seen[fen] = clamp_score(score)
                    except Exception:
                        pass

        samples = list(seen.items())
        random.shuffle(samples)

        if max_positions:
            samples = samples[:max_positions]

        self.samples = samples

        n_files = sum(1 for f in files if os.path.exists(f))
        print(f"  [Dataset] {len(self.samples):,} unique positions from {n_files} file(s) ({skipped:,} filtered)")

        if self.samples:
            scores = [s for _, s in self.samples]
            avg    = sum(scores) / len(scores)
            print(f"  [Dataset] score range: {min(scores):.0f} to {max(scores):.0f} cp | avg: {avg:.0f} cp")
            extreme = sum(1 for s in scores if abs(s) > 400)
            print(f"  [Dataset] |score| > 400cp: {extreme:,} ({100*extreme/len(scores):.1f}%)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fen, score = self.samples[idx]
        board      = chess.Board(fen)
        wi, bi     = halfkp_indices(board)

        w = np.zeros(HALFKP, dtype=np.float32)
        b = np.zeros(HALFKP, dtype=np.float32)
        if len(wi): w[wi] = 1.0
        if len(bi): b[bi] = 1.0

        target = score if board.turn == chess.WHITE else -score

        return (
            torch.from_numpy(w),
            torch.from_numpy(b),
            torch.tensor(target, dtype=torch.float32),
        )


def wdl_loss(pred: torch.Tensor, target: torch.Tensor,
             lam: float = 0.7, scale: float = CP_SCALE,
             label_smooth: float = 0.05) -> torch.Tensor:
    pred = pred.squeeze()
    mse  = nn.functional.mse_loss(pred, target)
    pw   = torch.sigmoid(pred   / scale)
    tw   = torch.sigmoid(target / scale)
    tw   = tw * (1.0 - label_smooth) + 0.5 * label_smooth
    ce   = nn.functional.binary_cross_entropy(pw, tw)
    return (1.0 - lam) * mse / (scale ** 2) + lam * ce


OPENINGS = [
    "e2e4 e7e5",
    "e2e4 c7c5",
    "e2e4 e7e6",
    "e2e4 c7c6",
    "d2d4 d7d5",
    "d2d4 g8f6",
    "d2d4 f7f5",
    "c2c4",
    "g1f3 d7d5",
    "g1f3 g8f6",
    "e2e4 e7e5 g1f3 b8c6",
    "e2e4 e7e5 g1f3 g8f6",
    "d2d4 d7d5 c2c4",
    "e2e4 e7e5 f2f4",
    "e2e4 c7c5 g1f3",
    "d2d4 g8f6 c2c4 e7e6",
    "e2e4 e7e6 d2d4 d7d5",
    "g1f3 d7d5 d2d4 g8f6 c2c4",
]


def _play_one_game(engine: Engine, seconds_per_move: float) -> list:
    board    = chess.Board()
    recorded = []

    opening = random.choice(OPENINGS).split()
    for uci in opening:
        try:
            mv = chess.Move.from_uci(uci)
            if mv in board.legal_moves:
                board.push(mv)
        except Exception:
            pass

    for _ in range(MAX_MOVE * 2):
        if board.is_game_over():
            break

        mv, score, _ = engine.go(board, move_time=seconds_per_move)
        if mv is None:
            break

        move_num = board.fullmove_number
        clamped  = clamp_score(float(score))

        if (MIN_MOVE <= move_num <= MAX_MOVE
                and abs(clamped) <= FILTER_SCORE
                and not board.is_check()):
            recorded.append((board.fen(), clamped))

        board.push(mv)

    res     = board.result()
    outcome = {"1-0": 1.0, "0-1": -1.0}.get(res, 0.0)

    LAM = 0.5
    return [
        (fen, clamp_score(LAM * cp + (1 - LAM) * outcome * 600.0))
        for fen, cp in recorded
    ]


def run_selfplay(num_games: int, seconds_per_move: float,
                 model_path: str = None, stockfish_path: str = None) -> str:
    ev     = Evaluator(model_path)
    engine = Engine(ev)

    out_path = os.path.join(DATA_DIR, f"sp_{int(time.time())}.txt")
    all_data = {}

    print(f"\n{'━'*60}")
    print(f"  Self-play: {num_games} games | {seconds_per_move}s/move")
    print(f"  Filtering: moves {MIN_MOVE}-{MAX_MOVE} | |score| <= {FILTER_SCORE}cp | no checks")
    print(f"  Model: {model_path or 'classical eval'}")
    print(f"  Output: {out_path}")
    print(f"{'━'*60}")

    games_done = 0
    try:
        with open(out_path, 'w', encoding='utf-8') as f:
            for i in range(num_games):
                pairs = _play_one_game(engine, seconds_per_move)
                for fen, score in pairs:
                    all_data[fen] = score
                    f.write(f"{fen}|{score:.2f}\n")
                f.flush()
                games_done += 1
                _bar(i + 1, num_games, prefix='  Games ')
    except KeyboardInterrupt:
        _bar_done()
        print(f"\n  Stopped — {games_done}/{num_games} games saved")
        print(f"  {len(all_data):,} positions written to: {out_path}")
        return out_path

    _bar_done()
    print(f"  {len(all_data):,} unique positions saved to: {out_path}")
    return out_path


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay  = decay
        self.shadow = copy.deepcopy(model.state_dict())

    def update(self, model: nn.Module):
        with torch.no_grad():
            for k, v in model.state_dict().items():
                self.shadow[k] = self.decay * self.shadow[k] + (1 - self.decay) * v.float()

    def apply_to(self, model: nn.Module):
        model.load_state_dict(self.shadow)


def train(data_files, model_path: str = None,
          epochs: int = 20, batch_size: int = 4096,
          lr: float = 3e-4, lam: float = 0.7,
          max_pos: int = None):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'━'*60}")
    print(f"  Training NNUE")
    print(f"  Device: {device} {'(GPU)' if device.type == 'cuda' else '(CPU)'}")
    print(f"{'━'*60}\n")

    if model_path and os.path.exists(model_path):
        model = NNUE.load(model_path)
        print(f"  Resuming from: {model_path}")
    else:
        model = NNUE()
        print("  Starting fresh model")
    model = model.to(device)

    ds = PositionDataset(data_files, max_positions=max_pos)
    if len(ds) == 0:
        print("  No valid training data found — run --selfplay first.")
        return None

    n_val   = max(1024, len(ds) // 10)
    n_train = len(ds) - n_val
    tr_ds, va_ds = torch.utils.data.random_split(ds, [n_train, n_val])

    accum  = max(1, batch_size // 2048)
    actual = batch_size // accum

    tr_ld = DataLoader(tr_ds, batch_size=actual, shuffle=True,
                       num_workers=0, pin_memory=(device.type == 'cuda'))
    va_ld = DataLoader(va_ds, batch_size=actual, shuffle=False, num_workers=0)

    opt   = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    total = epochs * len(tr_ld)
    warm  = max(1, total // 20)
    ema   = EMA(model, decay=0.9995)

    def lr_fn(step):
        if step < warm:
            return step / warm
        p = (step - warm) / max(1, total - warm)
        return 0.5 * (1 + math.cos(math.pi * p))

    sched = optim.lr_scheduler.LambdaLR(opt, lr_fn)

    best_val  = float('inf')
    best_path = os.path.join(MODEL_DIR, 'nnue_best.pt')
    step      = 0

    print(f"  Train: {n_train:,} | Val: {n_val:,} | Batch: {batch_size} ({actual}×{accum}) | LR: {lr}")
    print()

    for epoch in range(1, epochs + 1):
        model.train()
        ep_loss    = 0.0
        ep_batches = 0
        opt.zero_grad()

        for bi, (w, b, tgt) in enumerate(tr_ld):
            w   = w.to(device)
            b   = b.to(device)
            tgt = tgt.to(device)

            pred = model(w, b)
            loss = wdl_loss(pred, tgt, lam=lam) / accum
            loss.backward()

            if (bi + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                ema.update(model)
                opt.zero_grad()
                step += 1

            ep_loss    += loss.item() * accum
            ep_batches += 1
            _bar(bi + 1, len(tr_ld), prefix=f'  Epoch {epoch:3d} ')

        _bar_done()

        ema_model = copy.deepcopy(model)
        ema.apply_to(ema_model)
        ema_model.eval()

        val_loss = 0.0
        with torch.no_grad():
            for w, b, tgt in va_ld:
                w, b, tgt = w.to(device), b.to(device), tgt.to(device)
                val_loss += wdl_loss(ema_model(w, b), tgt, lam=lam).item()

        avg_tr  = ep_loss  / max(ep_batches, 1)
        avg_val = val_loss / max(len(va_ld), 1)
        cur_lr  = sched.get_last_lr()[0] if sched.get_last_lr() else lr

        is_best = avg_val < best_val
        marker  = " ← best!" if is_best else ""
        print(f"  Epoch {epoch:3d}/{epochs} | train: {avg_tr:.5f} | val: {avg_val:.5f} | "
              f"lr: {cur_lr:.2e}{marker}")

        if is_best:
            best_val = avg_val
            ema_model.save(best_path)

    latest = os.path.join(MODEL_DIR, 'nnue_latest.pt')
    ema_model.save(latest)

    print(f"\n  Done! Best val loss: {best_val:.5f}")
    print(f"  Best model:   {best_path}")
    print(f"  Latest model: {latest}")
    print(f"\n  To play: python play.py --model models/nnue_best.pt")
    return best_path


def benchmark(model_path=None, seconds=10.0):
    ev  = Evaluator(model_path)
    eng = Engine(ev)
    b   = chess.Board()

    print(f"\nBenchmarking for {seconds}s...")
    mv, sc, depth = eng.go(b, move_time=seconds)

    elapsed = eng.tm.elapsed()
    nps     = int(eng.nodes / max(elapsed, 1e-9))

    print(f"\n  Best move: {mv}")
    print(f"  Score:     {sc:+d} cp ({sc/100:+.2f} pawns)")
    print(f"  Depth:     {depth}")
    print(f"  Nodes:     {eng.nodes:,}")
    print(f"  NPS:       {nps:,}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--selfplay',  action='store_true')
    ap.add_argument('--train',     action='store_true')
    ap.add_argument('--benchmark', action='store_true')
    ap.add_argument('--games',     type=int,   default=200)
    ap.add_argument('--epochs',    type=int,   default=20)
    ap.add_argument('--batch',     type=int,   default=4096)
    ap.add_argument('--lr',        type=float, default=3e-4)
    ap.add_argument('--lam',       type=float, default=0.7)
    ap.add_argument('--time',      type=float, default=0.05)
    ap.add_argument('--model',     type=str,   default=None)
    ap.add_argument('--data',      type=str,   default=None)
    ap.add_argument('--stockfish', type=str,   default=None)
    ap.add_argument('--max-pos',   type=int,   default=None)
    args = ap.parse_args()

    best = args.model or os.path.join(MODEL_DIR, 'nnue_best.pt')
    if not os.path.exists(best):
        best = None

    if not any([args.selfplay, args.train, args.benchmark]):
        ap.print_help()
        sys.exit(0)

    new_data = None

    if args.selfplay:
        new_data = run_selfplay(
            num_games        = args.games,
            seconds_per_move = args.time,
            model_path       = best,
            stockfish_path   = args.stockfish,
        )

    if args.train:
        if args.data:
            data_files = [args.data]
        else:
            data_files = sorted(
                os.path.join(DATA_DIR, f)
                for f in os.listdir(DATA_DIR)
                if f.endswith('.txt')
            )

        if not data_files:
            print("No training data found — run --selfplay first.")
            sys.exit(1)

        print(f"  Using {len(data_files)} data file(s)")
        best = train(
            data_files = data_files,
            model_path = best,
            epochs     = args.epochs,
            batch_size = args.batch,
            lr         = args.lr,
            lam        = args.lam,
            max_pos    = args.max_pos,
        )

    if args.benchmark:
        benchmark(model_path=best, seconds=10.0)