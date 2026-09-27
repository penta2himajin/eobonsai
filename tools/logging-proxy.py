#!/usr/bin/env python3
"""Logging proxy in front of llama-server, to see what a client actually sends.

The model's behaviour only tells you whether a request *had an effect*; it cannot tell you
whether a client omitted a field or sent one the server ignores. This records the raw request
bodies so the question is answered directly.

Usage:
  python3 tools/logging-proxy.py --listen 8081 --target http://127.0.0.1:8080 \
      --out out/proxy.jsonl            # then point the client at http://127.0.0.1:8081/v1
"""
import argparse
import http.server
import json
import pathlib
import socketserver
import time
import urllib.error
import urllib.request


class Handler(http.server.BaseHTTPRequestHandler):
    target = ""
    out = None
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):  # keep the harness output clean
        pass

    def _proxy(self, method):
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""

        record = {
            "t": time.time(),
            "method": method,
            "path": self.path,
            "body": None,
            "note": None,
        }
        if body:
            try:
                parsed = json.loads(body)
                record["body"] = parsed
                # What we actually care about, hoisted so it is greppable.
                record["note"] = {
                    "reasoning_effort": parsed.get("reasoning_effort"),
                    "thinking_budget_tokens": parsed.get("thinking_budget_tokens"),
                    "chat_template_kwargs": parsed.get("chat_template_kwargs"),
                    "max_tokens": parsed.get("max_tokens"),
                    "stream": parsed.get("stream"),
                    "message_count": len(parsed.get("messages") or []),
                    "tool_count": len(parsed.get("tools") or []),
                }
            except (json.JSONDecodeError, UnicodeDecodeError):
                record["body"] = body[:2000].decode("utf-8", "replace")
        else:
            record["note"] = {"reasoning_effort": None, "path_query": self.path}

        if record["body"] is not None or "note" in record:
            with open(Handler.out, "a") as fh:
                fh.write(json.dumps(record) + "\n")

        req = urllib.request.Request(Handler.target + self.path, data=body or None, method=method)
        for k, v in self.headers.items():
            if k.lower() not in ("host", "content-length", "connection"):
                req.add_header(k, v)
        try:
            resp = urllib.request.urlopen(req, timeout=1800)
            code, payload, headers = resp.status, resp.read(), resp.headers
        except urllib.error.HTTPError as e:
            code, payload, headers = e.code, e.read(), e.headers

        self.send_response(code)
        for k, v in headers.items():
            if k.lower() not in ("transfer-encoding", "content-length", "connection"):
                self.send_header(k, v)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self):
        self._proxy("POST")

    def do_GET(self):
        self._proxy("GET")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--listen", type=int, default=8081)
    ap.add_argument("--target", default="http://127.0.0.1:8080")
    ap.add_argument("--out", default="out/proxy.jsonl")
    args = ap.parse_args()

    Handler.target = args.target.rstrip("/")
    Handler.out = pathlib.Path(args.out)
    Handler.out.parent.mkdir(parents=True, exist_ok=True)
    Handler.out.write_text("")

    socketserver.ThreadingTCPServer.allow_reuse_address = True
    with socketserver.ThreadingTCPServer(("127.0.0.1", args.listen), Handler) as httpd:
        print(f"proxying 127.0.0.1:{args.listen} -> {Handler.target}, logging to {Handler.out}",
              flush=True)
        httpd.serve_forever()


if __name__ == "__main__":
    main()
