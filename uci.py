import sys, os, threading, argparse
import chess

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from engine.nnue   import Evaluator
from engine.search import Engine

NAME   = "PyNNUE"
AUTHOR = "You"


class UCIEngine:
    def __init__(self, model_path=None):
        self.ev      = Evaluator(model_path)
        self.engine  = Engine(self.ev)
        self.board   = chess.Board()
        self._thread = None

        # engine options with defaults
        self.opts = {
            'Hash':     128,
            'Threads':  1,
            'MoveTime': 1000,
            'Depth':    64,
        }

    def out(self, msg: str):
        print(msg, flush=True)

    def cmd_uci(self):
        self.out(f"id name {NAME}")
        self.out(f"id author {AUTHOR}")
        self.out("option name Hash     type spin default 128 min 1 max 4096")
        self.out("option name MoveTime type spin default 1000 min 50 max 60000")
        self.out("option name Depth    type spin default 64 min 1 max 128")
        self.out("uciok")

    def cmd_position(self, tokens):
        i = 0

        if i < len(tokens) and tokens[i] == 'startpos':
            self.board = chess.Board()
            i += 1
        elif i < len(tokens) and tokens[i] == 'fen':
            i += 1
            fen_parts = []
            while i < len(tokens) and tokens[i] != 'moves':
                fen_parts.append(tokens[i])
                i += 1
            self.board = chess.Board(' '.join(fen_parts))

        if i < len(tokens) and tokens[i] == 'moves':
            i += 1
            while i < len(tokens):
                try:
                    mv = chess.Move.from_uci(tokens[i])
                    if mv in self.board.legal_moves:
                        self.board.push(mv)
                except Exception:
                    pass
                i += 1

    def cmd_go(self, tokens):
        params = {}
        i = 0

        while i < len(tokens):
            k = tokens[i]
            if k in ('wtime', 'btime', 'winc', 'binc', 'movestogo',
                     'depth', 'movetime', 'nodes') and i + 1 < len(tokens):
                try:
                    params[k] = int(tokens[i + 1])
                except Exception:
                    pass
                i += 2
            elif k == 'infinite':
                params['infinite'] = True
                i += 1
            else:
                i += 1

        movetime  = params.get('movetime', self.opts['MoveTime']) / 1000.0
        max_depth = params.get('depth', self.opts['Depth'])
        wtime     = params.get('wtime')
        btime     = params.get('btime')
        winc      = params.get('winc', 0)
        binc      = params.get('binc', 0)

        if params.get('infinite'):
            movetime = 3600.0  # just search for an hour, stop will interrupt it

        board_copy = self.board.copy()

        def worker():
            move, score, depth = self.engine.go(
                board_copy,
                move_time = movetime,
                max_depth = max_depth,
                wtime=wtime, btime=btime,
                winc=winc,   binc=binc,
            )
            self.out(f"bestmove {move.uci() if move else '0000'}")

        self.engine.stop = False
        self._thread = threading.Thread(target=worker, daemon=True)
        self._thread.start()

    def cmd_stop(self):
        self.engine.stop = True
        if self._thread:
            self._thread.join(timeout=2.0)

    def cmd_setoption(self, tokens):
        try:
            ni    = tokens.index('name')
            vi    = tokens.index('value')
            name  = ' '.join(tokens[ni + 1:vi])
            value = ' '.join(tokens[vi + 1:])
            if name in self.opts:
                self.opts[name] = int(value)
        except Exception:
            pass

    def run(self):
        while True:
            try:
                line = input().strip()
            except EOFError:
                break

            if not line:
                continue

            tok  = line.split()
            cmd  = tok[0]
            rest = tok[1:]

            if cmd == 'uci':
                self.cmd_uci()
            elif cmd == 'isready':
                self.out('readyok')
            elif cmd == 'ucinewgame':
                self.board = chess.Board()
                self.engine.tt.clear()
                self.engine.reset_heuristics()
            elif cmd == 'position':
                self.cmd_position(rest)
            elif cmd == 'go':
                self.cmd_go(rest)
            elif cmd == 'stop':
                self.cmd_stop()
            elif cmd == 'setoption':
                self.cmd_setoption(rest)
            elif cmd == 'quit':
                self.cmd_stop()
                break
            elif cmd == 'd':
                # debug: print board + fen
                print(self.board, flush=True)
                print(f"FEN: {self.board.fen()}", flush=True)
            elif cmd == 'eval':
                self.ev.prepare(self.board)
                s = self.ev.score(self.board)
                print(f"Eval: {s:+d} cp (side to move)", flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=None, help='path to NNUE model .pt file')
    args = ap.parse_args()

    model_path = args.model
    if model_path is None:
        default = os.path.join(ROOT, 'models', 'nnue_best.pt')
        if os.path.exists(default):
            model_path = default

    UCIEngine(model_path).run()