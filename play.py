import sys, os, argparse
import chess

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from engine.nnue     import Evaluator
from engine.search   import Engine
from engine.evaluate import evaluate as classical_eval


def _supports_color() -> bool:
    if sys.platform == "win32":
        try:
            import ctypes
            kernel = ctypes.windll.kernel32
            kernel.SetConsoleMode(kernel.GetStdHandle(-11), 7)
            return True
        except Exception:
            return False
    return hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()

USE_COLOR = _supports_color()

RESET    = "\033[0m"
BOLD     = "\033[1m"
LIGHT_BG = "\033[48;5;229m"  # light squares
DARK_BG  = "\033[48;5;136m"  # dark squares
WHITE_FG = "\033[97;1m"
BLACK_FG = "\033[30;1m"


def _col(text: str, *codes: str) -> str:
    if not USE_COLOR:
        return text
    return "".join(codes) + text + RESET


PIECE_NAMES = {
    chess.PAWN:   'P',
    chess.KNIGHT: 'N',
    chess.BISHOP: 'B',
    chess.ROOK:   'R',
    chess.QUEEN:  'Q',
    chess.KING:   'K',
}


def render(board: chess.Board, flip: bool = False, ascii_mode: bool = False):
    print()

    ranks  = range(7, -1, -1) if not flip else range(8)
    files  = range(8)         if not flip else range(7, -1, -1)

    print("    +" + "----+" * 8)

    for rank in ranks:
        print(f"  {rank+1} |", end="")

        for file in files:
            sq       = chess.square(file, rank)
            pc       = board.piece_at(sq)
            is_light = (rank + file) % 2 == 1

            if pc:
                side   = 'w' if pc.color == chess.WHITE else 'b'
                letter = PIECE_NAMES[pc.piece_type]
                cell   = f"{side}{letter}"  # e.g. "wK", "bQ"

                if USE_COLOR:
                    bg = LIGHT_BG if is_light else DARK_BG
                    fg = WHITE_FG if pc.color == chess.WHITE else BLACK_FG
                    print(f"{bg}{fg} {cell} {RESET}|", end="")
                else:
                    print(f" {cell} |", end="")
            else:
                fill = "    " if is_light else " ·· "
                if USE_COLOR:
                    bg = LIGHT_BG if is_light else DARK_BG
                    print(f"{bg}{fill}{RESET}|", end="")
                else:
                    print(f"{fill}|", end="")

        print()
        print("    +" + "----+" * 8)

    files_label = "abcdefgh" if not flip else "hgfedcba"
    print("      " + "    ".join(files_label))
    print()
    print("  w = White pieces   b = Black pieces")
    print("  P=Pawn N=Knight B=Bishop R=Rook Q=Queen K=King")
    print()


def _eval_bar(score: int, width: int = 20) -> str:
    clamped = max(-500, min(500, score))
    center  = width // 2
    fill    = int(center * clamped / 500)

    bar = ['-'] * width
    if fill > 0:
        for i in range(center, center + fill):
            bar[i] = '█'
    elif fill < 0:
        for i in range(center + fill, center):
            bar[i] = '█'
    bar[center] = '|'

    return f"W {''.join(bar)} B"


def parse_move(board: chess.Board, text: str):
    text = text.strip()

    # try UCI first (e.g. e2e4), then SAN (e.g. Nf3)
    try:
        mv = chess.Move.from_uci(text)
        if mv in board.legal_moves:
            return mv
    except Exception:
        pass

    try:
        mv = board.parse_san(text)
        if mv in board.legal_moves:
            return mv
    except Exception:
        pass

    return None


def play(model_path=None, move_time=2.0, human_color=chess.WHITE, ascii_mode=False):
    ev      = Evaluator(model_path)
    engine  = Engine(ev)
    board   = chess.Board()
    history = []
    flip    = (human_color == chess.BLACK)  # put human's pieces at the bottom

    print(f"\n{'━'*54}")
    print(f"  PyNNUE Chess Engine")
    print(f"{'━'*54}")
    print(f"  Playing as: {'White ♙' if human_color == chess.WHITE else 'Black ♟'}")
    print(f"  Engine time: {move_time}s per move")
    print(f"  Model: {model_path or 'classical evaluation'}")
    print(f"  Color mode: {'ANSI color' if USE_COLOR else 'plain text'}")
    if not USE_COLOR and not ascii_mode:
        print(f"  Tip: if pieces look wrong, try --ascii flag")
    print(f"\n  Move format: e2e4  OR  e4  OR  Nf3  OR  O-O")
    print(f"  Commands:    undo | eval | new | quit")
    print(f"{'━'*54}\n")

    while True:
        render(board, flip=flip, ascii_mode=ascii_mode)

        ev.prepare(board)
        raw_score   = ev.score(board)
        white_score = raw_score if board.turn == chess.WHITE else -raw_score
        bar         = _eval_bar(white_score)
        turn_name   = "White" if board.turn == chess.WHITE else "Black"

        print(f"  {bar}")
        print(f"  Eval: {white_score:+d} cp | Move {board.fullmove_number} | {turn_name} to move")

        if board.is_game_over():
            res  = board.result()
            msgs = {"1-0": "White wins!", "0-1": "Black wins!", "1/2-1/2": "Draw!"}
            print(f"\n  Game over — {res} — {msgs.get(res, '')}")

            cmd = input("\n  [new / quit] → ").strip().lower()
            if cmd == 'new':
                board = chess.Board()
                history = []
                engine.tt.clear()
                engine.reset_heuristics()
                continue
            break

        if board.turn == human_color:
            try:
                raw = input("\n  Your move → ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n  Quit.")
                break

            cmd = raw.lower()

            if not cmd:
                continue
            elif cmd == 'quit':
                print("  Goodbye!")
                break
            elif cmd == 'new':
                board = chess.Board()
                history = []
                engine.tt.clear()
                engine.reset_heuristics()
                print("  New game started!")
                continue
            elif cmd == 'undo':
                n = min(2, len(history))
                for _ in range(n):
                    board.pop()
                    history.pop()
                if n == 0:
                    print("  Nothing to undo.")
                else:
                    print(f"  Undid {n} move(s).")
                continue
            elif cmd == 'eval':
                ev.prepare(board)
                s  = ev.score(board)
                ce = classical_eval(board)
                print(f"  Eval: {s:+d} cp ({s/100:.2f} pawns)")
                print(f"  Classical: {ce:+d} cp | NNUE: {'yes' if ev.use_nnue else 'no (classical)'}")
                continue

            mv = parse_move(board, raw)
            if mv is None:
                legal_sample = [board.san(m) for m in list(board.legal_moves)[:8]]
                print(f"  ✗ '{raw}' is not a valid move.")
                print(f"  Some legal moves: {', '.join(legal_sample)}...")
                continue

            board.push(mv)
            history.append(mv)

        else:
            print(f"  Engine is thinking...", end="\r", flush=True)
            mv, score, depth = engine.go(board, move_time=move_time)

            if mv is None:
                print("  Engine has no legal moves!")
                break

            san = board.san(mv)
            board.push(mv)
            history.append(mv)

            score_pawns = score / 100.0
            print(f"  Engine plays: {san} ({mv.uci()}) | "
                  f"eval: {score:+d} cp ({score_pawns:+.2f}p) | depth: {depth}   ")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Play chess against PyNNUE')
    ap.add_argument('--model', default=None,         help='path to NNUE model .pt file')
    ap.add_argument('--time',  type=float, default=2.0, help='engine thinking time in seconds (default 2)')
    ap.add_argument('--black', action='store_true',  help='play as Black')
    ap.add_argument('--ascii', action='store_true',  help='use ASCII pieces if unicode looks broken')
    args = ap.parse_args()

    model_path = args.model
    if model_path is None:
        default = os.path.join(ROOT, 'models', 'nnue_best.pt')
        if os.path.exists(default):
            model_path = default

    play(
        model_path  = model_path,
        move_time   = args.time,
        human_color = chess.BLACK if args.black else chess.WHITE,
        ascii_mode  = args.ascii,
    )