#!/usr/bin/env python3
"""Tiny OpenAI-compatible proxy in front of llama-server.
If a non-streaming chat completion returns empty `content` but non-empty `reasoning_content`
(thinking model that never closed its think block), copy reasoning into content so agent
harnesses still see the answer. Same fallback used by eval.py for HumanEval.
Usage: proxy.py <listen_port> <upstream_port>
"""
import json, sys, urllib.request, urllib.error
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

LISTEN, UP = int(sys.argv[1]), int(sys.argv[2])


class H(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, *a):
        pass

    def _fwd(self, method):
        n = int(self.headers.get('Content-Length') or 0)
        body = self.rfile.read(n) if n else None
        stream = False
        if body:
            try:
                stream = bool(json.loads(body).get('stream'))
            except Exception:
                pass
        req = urllib.request.Request(f'http://127.0.0.1:{UP}{self.path}', data=body, method=method,
                                     headers={'Content-Type': self.headers.get('Content-Type', 'application/json')})
        try:
            r = urllib.request.urlopen(req, timeout=3600)
            code = r.status
        except urllib.error.HTTPError as e:
            r, code = e, e.code
        if stream:
            self.send_response(code)
            self.send_header('Content-Type', r.headers.get('Content-Type', 'text/event-stream'))
            self.send_header('Transfer-Encoding', 'chunked')
            self.end_headers()
            while True:
                chunk = r.read1(65536) if hasattr(r, 'read1') else r.read(65536)
                if not chunk:
                    break
                self.wfile.write(b'%x\r\n%s\r\n' % (len(chunk), chunk))
            self.wfile.write(b'0\r\n\r\n')
            return
        data = r.read()
        if self.path.endswith('/chat/completions') and code == 200:
            try:
                o = json.loads(data)
                for ch in o.get('choices', []):
                    m = ch.get('message') or {}
                    if not (m.get('content') or '').strip() and (m.get('reasoning_content') or '').strip() \
                            and not m.get('tool_calls'):
                        m['content'] = m['reasoning_content']
                data = json.dumps(o).encode()
            except Exception:
                pass
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        self._fwd('POST')

    def do_GET(self):
        self._fwd('GET')


ThreadingHTTPServer(('127.0.0.1', LISTEN), H).serve_forever()
