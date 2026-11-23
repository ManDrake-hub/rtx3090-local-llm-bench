#!/usr/bin/env python3
"""127.0.0.1:<listen> -> vLLM container. Forces model name, maps `reasoning` -> `reasoning_content`,
and (like agentic/proxy.py) copies reasoning into empty content only on the --fill port.
Usage: adapter.py <listen> <upstream host:port> [--fill]"""
import json, sys, urllib.request, urllib.error
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
LISTEN, UP, FILL = int(sys.argv[1]), sys.argv[2], '--fill' in sys.argv


class H(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    def log_message(self, *a): pass

    def fwd(self, method):
        n = int(self.headers.get('Content-Length') or 0)
        body = self.rfile.read(n) if n else None
        stream = False
        if body:
            try:
                o = json.loads(body)
                o['model'] = 'inkling-2-mini'
                stream = bool(o.get('stream'))
                body = json.dumps(o).encode()
            except Exception:
                pass
        req = urllib.request.Request(f'http://{UP}{self.path}', data=body, method=method, headers={'Content-Type': 'application/json'})
        try:
            r = urllib.request.urlopen(req, timeout=3600); code = r.status
        except urllib.error.HTTPError as e:
            r, code = e, e.code
        except Exception as e:
            data = json.dumps({'error': {'message': repr(e)}}).encode()
            self.send_response(502); self.send_header('Content-Length', str(len(data))); self.end_headers(); self.wfile.write(data); return
        if stream and code == 200:
            self.send_response(code)
            self.send_header('Content-Type', 'text/event-stream'); self.send_header('Transfer-Encoding', 'chunked'); self.end_headers()
            while True:
                c = r.read1(65536)
                if not c: break
                self.wfile.write(b'%x\r\n%s\r\n' % (len(c), c)); self.wfile.flush()
            self.wfile.write(b'0\r\n\r\n'); return
        data = r.read()
        if code == 200 and self.path.endswith('/chat/completions'):
            try:
                o = json.loads(data)
                for ch in o.get('choices', []):
                    m = ch.get('message') or {}
                    if m.get('reasoning') and not m.get('reasoning_content'):
                        m['reasoning_content'] = m['reasoning']
                    if FILL and not (m.get('content') or '').strip() and (m.get('reasoning_content') or '').strip() and not m.get('tool_calls'):
                        m['content'] = m['reasoning_content']
                data = json.dumps(o).encode()
            except Exception:
                pass
        self.send_response(code); self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data))); self.end_headers(); self.wfile.write(data)

    def do_GET(self): self.fwd('GET')
    def do_POST(self): self.fwd('POST')


ThreadingHTTPServer(('127.0.0.1', LISTEN), H).serve_forever()
