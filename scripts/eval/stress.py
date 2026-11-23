#!/usr/bin/env python3
"""Agentic-client stress test against the live server (/v1/messages): many messages, tool use, irregular sizes.
Usage: stress.py <ctx>. Exit 0 only if every request succeeds and llama-server PID never changes."""
import base64, glob, json, os, random, struct, subprocess, sys, threading, time, urllib.request, zlib

CTX = int(sys.argv[1])
KEY = open(os.path.expanduser('~/.config/llama/api-key')).read().strip()
URL = 'http://127.0.0.1:8080/v1/messages'
random.seed(7)
SRC = ''.join(open(f, errors='ignore').read() for f in sorted(glob.glob(os.path.expanduser('~/llama.cpp-prod/src/*.cpp'))))
pos = 0
TOOLS = [{'name': 'read_file', 'description': 'Read a file', 'input_schema': {
    'type': 'object', 'properties': {'path': {'type': 'string'}}, 'required': ['path']}}]


def pid():
    return subprocess.run(['systemctl', '--user', 'show', 'llama-qwen', '-p', 'MainPID', '--value'],
                          capture_output=True, text=True).stdout.strip()


def free():
    return int(subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                              capture_output=True, text=True).stdout.split()[0])


def text(n_tok):
    global pos
    n = int(n_tok * 3.2)
    s = SRC[pos:pos + n]
    pos += n
    if pos > len(SRC) - 700000:
        pos = 1000
    return s


P0, minfree, done = pid(), free(), False
print(f'start pid={P0} free={minfree}', flush=True)


def watch():
    global minfree
    while not done:
        minfree = min(minfree, free())
        time.sleep(0.4)


threading.Thread(target=watch, daemon=True).start()


def call(msgs, label, max_tokens=500, tools=True):
    body = {'model': 'qwen3.8-27b', 'max_tokens': max_tokens, 'system': 'You are a coding agent.', 'messages': msgs}
    if tools:
        body['tools'] = TOOLS
    req = urllib.request.Request(URL, json.dumps(body).encode(), {
        'Content-Type': 'application/json', 'x-api-key': KEY, 'Authorization': 'Bearer ' + KEY,
        'anthropic-version': '2023-06-01'})
    t = time.time()
    try:
        r = json.load(urllib.request.urlopen(req, timeout=1200))
    except Exception as e:
        detail = e.read()[:200] if hasattr(e, 'read') else ''
        print(f'FAIL {label}: {e!r} {detail} pid={pid()} minfree={minfree}', flush=True)
        sys.exit(1)
    if pid() != P0:
        print(f'FAIL {label}: PID changed', flush=True)
        sys.exit(1)
    u = r['usage']
    tot = u.get('input_tokens', 0) + u.get('cache_read_input_tokens', 0) + u.get('cache_creation_input_tokens', 0)
    print(f'ok {label}: prompt={tot} out={u["output_tokens"]} {time.time() - t:.0f}s free={free()} minfree={minfree}', flush=True)
    return r, tot


def add_tool_turn(msgs, i, n):
    msgs.append({'role': 'assistant', 'content': [{'type': 'tool_use', 'id': f't{i}', 'name': 'read_file', 'input': {'path': f'f{i}.cpp'}}]})
    msgs.append({'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': f't{i}', 'content': text(n)}]})


def png():
    raw = b''.join(b'\x00' + bytes((255, 0, 0)) * 256 for _ in range(256))
    c = lambda t, d: struct.pack('>I', len(d)) + t + d + struct.pack('>I', zlib.crc32(t + d))
    return base64.b64encode(b'\x89PNG\r\n\x1a\n' + c(b'IHDR', struct.pack('>IIBBBBB', 256, 256, 8, 2, 0, 0, 0))
                            + c(b'IDAT', zlib.compress(raw)) + c(b'IEND', b'')).decode()


# 1) cold ~90k prompt built from many irregular messages (the crashing shape)
msgs, i, est = [{'role': 'user', 'content': 'Review these sources one by one, then list bugs. Be detailed.'}], 0, 0
while est < 88000:
    n = random.choice([37, 180, 640, 1500, 2900, 5200, 9100])
    add_tool_turn(msgs, i, n)
    est += n
    i += 1
_, tot = call(msgs, 'cold-90k', 700)
# 2) grow irregularly to near ctx, always leaving room for output; side requests in between
limit = CTX - 9000
turn = 0
while tot < limit - 600 and turn < 40:
    n = random.choice([60, 333, 1100, 2700, 4800, 7600])
    n = min(n, int((limit - tot) * 0.7))
    add_tool_turn(msgs, i, n)
    i += 1
    turn += 1
    _, tot = call(msgs, f'turn{turn}', random.choice([64, 250, 600, 1000]))
    if turn % 5 == 0:
        call([{'role': 'user', 'content': 'Title for: ' + text(random.choice([50, 900, 3300]))}], 'side', 120, tools=False)
        call(msgs, 'back-to-long', 200)
# 3) vision + tool call sanity on the same process
r, _ = call([{'role': 'user', 'content': [
    {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png', 'data': png()}},
    {'type': 'text', 'text': 'What color is this image? One word.'}]}], 'vision', 1500, tools=False)
print('vision answer:', [b.get('text') for b in r['content'] if b['type'] == 'text'], flush=True)
r, _ = call([{'role': 'user', 'content': 'Read the file main.py using your tool.'}], 'tool', 1500)
print('tool answer:', [b.get('input') for b in r['content'] if b['type'] == 'tool_use'], flush=True)
call(msgs, 'final-long', 1200)
done = True
print(f'PASS ctx={CTX} max_prompt={tot} minfree={minfree} pid={pid()}', flush=True)
