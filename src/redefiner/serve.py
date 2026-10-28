"""HTTP server for both modes.

    redefiner serve --model PATH_OR_HF_REPO --port 8000

    POST /extract  {"rulebook": TEXT_OR_DICT, "texts": [..], "threshold"?: 0.5, "none_bias"?: 2}
                   -> {"results": [[{"start", "end", "text", "label", "confidence"}, ..], ..], "ms"}
    POST /label    {"rulebook": TEXT_OR_DICT, "texts": [..], "spans": [[[start, end], ..], ..],
                    "allow_none"?: false, "snap"?: false, "none_bias"?: 2}
                   -> {"results": [[{"start", "end", "text", "label", "confidence", "alternatives"}
                                    or {"start", "end", "text", "unsupported"}, ..], ..], "ms"}
    GET  /info     -> the model's config

Requests are served one at a time (the model holds one rulebook cache). Binds to 127.0.0.1 unless --host is given.
"""

import argparse
import json
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import __version__
from .api import Tagger


class BadRequest(ValueError):
    pass


def _texts(req: dict) -> list[str]:
    texts = req.get("texts")
    if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
        raise BadRequest('"texts" must be a list of strings')
    return texts


def _rulebook(req: dict) -> str | dict:
    rulebook = req.get("rulebook")
    if not isinstance(rulebook, (str, dict)):
        raise BadRequest('"rulebook" must be rulebook text or a rulebook object')
    return rulebook


def extract(tagger: Tagger, req: dict) -> list:
    return tagger.extract(_texts(req), _rulebook(req), threshold=req.get("threshold"), none_bias=req.get("none_bias"))


def label(tagger: Tagger, req: dict) -> list:
    texts = _texts(req)
    spans = req.get("spans")
    if not isinstance(spans, list) or len(spans) != len(texts):
        raise BadRequest('"spans" must be a list with one list of [start, end] pairs per text')
    return tagger.label(texts, spans, _rulebook(req), allow_none=bool(req.get("allow_none")),
                        snap=bool(req.get("snap")), none_bias=req.get("none_bias"))


ROUTES = {"/extract": extract, "/label": label}


class Server(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], tagger: Tagger):
        super().__init__(address, Handler)
        self.tagger = tagger
        self.lock = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    server: Server

    def send(self, code: int, obj: dict) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/info":
            self.send(200, self.server.tagger.config)
        else:
            self.send(404, {"error": "GET /info, POST /extract or POST /label"})

    def do_POST(self):
        route = ROUTES.get(self.path)
        if route is None:
            return self.send(404, {"error": "POST /extract or POST /label"})
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            if not isinstance(req, dict):
                raise BadRequest("the request body must be a JSON object")
            with self.server.lock:
                t0 = time.perf_counter()
                results = route(self.server.tagger, req)
            self.send(200, {"results": results, "ms": round(1000 * (time.perf_counter() - t0))})
        except (ValueError, KeyError, TypeError) as err:   # bad JSON, bad fields, or a rulebook the model can't read
            self.send(400, {"error": str(err)})
        except Exception as err:
            traceback.print_exc()
            self.send(500, {"error": f"internal error: {type(err).__name__}"})

    def log_request(self, code="-", size="-"):
        pass   # no line per request; errors are still logged


def main() -> None:
    ap = argparse.ArgumentParser(prog="redefiner")
    ap.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="serve /extract and /label over HTTP")
    s.add_argument("--model", required=True, help="local folder or Hugging Face repo")
    s.add_argument("--device", default=None, help="cpu, cuda or cuda:N (default: cuda if available)")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    tagger = Tagger.from_pretrained(args.model, device=args.device)
    print(f"serving /extract and /label on http://{args.host}:{args.port}", flush=True)
    Server((args.host, args.port), tagger).serve_forever()


if __name__ == "__main__":
    main()
