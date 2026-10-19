"""Tiny JSON API (stdlib http.server only). Binds 127.0.0.1.

  GET  /judges            list judge profiles
  POST /grade             {"judge": name, "item_ids": [...]?} -> verdicts
  GET  /report?judge=...  full scorecard for one judge
  GET  /health            {"ok": true}
"""

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs

from .corpus import BY_ID, load_corpus
from .judges import get_judge, list_judges
from .report import grade_all, scorecard


def _json(handler, obj, status=200):
    body = json.dumps(obj, sort_keys=True, indent=2).encode()
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


class Handler(BaseHTTPRequestHandler):
    server_version = "JudgeLab/1.0"

    def log_message(self, *args):  # keep serve output quiet
        pass

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            return _json(self, {"ok": True})
        if parsed.path == "/judges":
            return _json(self, {"judges": list_judges()})
        if parsed.path == "/report":
            name = parse_qs(parsed.query).get("judge", ["calibrated"])[0]
            try:
                judge = get_judge(name)
            except ValueError as e:
                return _json(self, {"error": str(e)}, 400)
            return _json(self, scorecard(judge))
        return _json(self, {"error": "not found"}, 404)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/grade":
            return _json(self, {"error": "not found"}, 404)
        try:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length).decode() or "{}")
        except (ValueError, json.JSONDecodeError):
            return _json(self, {"error": "invalid JSON"}, 400)
        try:
            judge = get_judge(payload.get("judge", "calibrated"))
        except ValueError as e:
            return _json(self, {"error": str(e)}, 400)
        ids = payload.get("item_ids")
        items = load_corpus()
        if ids:
            unknown = [i for i in ids if i not in BY_ID]
            if unknown:
                return _json(self, {"error": "unknown item_ids: %s" % unknown}, 400)
            items = [BY_ID[i] for i in ids]
        verdicts = grade_all(judge, items)
        return _json(self, {"judge": judge.name,
                            "verdicts": [{"item_id": i["id"], **v}
                                         for i, v in zip(items, verdicts)]})


def serve(host="127.0.0.1", port=8078):
    httpd = HTTPServer((host, port), Handler)
    print("judge-lab serving on http://%s:%d (Ctrl-C to stop)" % (host, port))
    httpd.serve_forever()
