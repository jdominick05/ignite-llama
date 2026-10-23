#!/usr/bin/env python3
# Generates agentic calibration conversations with real tool use (HOPE expert pruning): the model behind a running
# llama-server answers questions with tools served by an MCP server (e.g. Context7's resolve-library-id / query-docs),
# and every turn, tool call and tool result is kept. Output lines: {"messages", "tools", "q", "_gen": true}, read by
# hope_calib.py with an "oai" source and "tools": "tools".
#
#   hope_gen_agent.py --questions Q.txt --out OUT.jsonl --mcp context7 [--settings ~/.qwen/settings.json]
#                     [--url http://127.0.0.1:8091] [--key-file KEYS] [--max-turns 8] [--limit N]
#
# The MCP endpoint and its headers come from the Qwen Code settings file (never printed); tools get Qwen Code's
# mcp__<server>__<tool> names. Resumable: questions already in OUT are skipped.
import argparse, json, os, sys, time, urllib.error, urllib.request

SYSTEM = ('You are a coding and security research assistant working in a terminal. When a question depends on a '
          "library's or framework's API, check the current documentation with the documentation tools before "
          'answering, and base code on what they return.')


class MCP:
    def __init__(self, url, headers):
        self.url, self.headers, self.sid, self.n = url, headers, None, 0
        self.call('initialize', {'protocolVersion': '2025-06-18', 'capabilities': {},
                                 'clientInfo': {'name': 'hope-gen-agent', 'version': '0'}})

    def call(self, method, params):
        self.n += 1
        h = {'Content-Type': 'application/json', 'Accept': 'application/json, text/event-stream', **self.headers}
        if self.sid:
            h['Mcp-Session-Id'] = self.sid
        req = urllib.request.Request(self.url, method='POST', headers=h, data=json.dumps(
            {'jsonrpc': '2.0', 'id': self.n, 'method': method, 'params': params}).encode())
        with urllib.request.urlopen(req, timeout=120) as r:
            body = r.read().decode('utf8', 'replace')
            self.sid = r.headers.get('Mcp-Session-Id') or self.sid
        if 'data:' in body[:200]:
            body = [l[5:].strip() for l in body.splitlines() if l.startswith('data:')][-1]
        d = json.loads(body)
        if 'error' in d:
            raise RuntimeError(d['error'].get('message', 'MCP error'))
        return d['result']


def chat(url, key, messages, tools, max_tokens):
    body = json.dumps({'model': 'flash-next', 'messages': messages, 'tools': tools, 'max_tokens': max_tokens}).encode()
    req = urllib.request.Request(url + '/v1/chat/completions', data=body, method='POST',
                                 headers={'Content-Type': 'application/json', 'Authorization': f'Bearer {key}'})
    with urllib.request.urlopen(req, timeout=1200) as r:
        d = json.load(r)
    return d['choices'][0]['message'], d.get('usage', {}).get('completion_tokens', 0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--questions', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--mcp', default='context7')
    p.add_argument('--settings', default=os.path.expanduser('~/.qwen/settings.json'))
    p.add_argument('--url', default='http://127.0.0.1:8091')
    p.add_argument('--key-file', default=os.path.expanduser('~/llm/fn-server.keys'))
    p.add_argument('--max-turns', type=int, default=8)
    p.add_argument('--max-tokens', type=int, default=6144)
    p.add_argument('--max-result-chars', type=int, default=24000)
    p.add_argument('--limit', type=int, default=0)
    a = p.parse_args()

    cfg = json.load(open(a.settings))['mcpServers'][a.mcp]
    mcp = MCP(cfg['httpUrl'], cfg.get('headers', {}))
    raw = mcp.call('tools/list', {})['tools']
    prefix = f'mcp__{a.mcp}__'
    tools = [{'type': 'function', 'function': {'name': prefix + t['name'], 'description': t.get('description', ''),
                                               'parameters': t.get('inputSchema', {'type': 'object'})}} for t in raw]
    key = open(a.key_file).readline().strip() if os.path.exists(a.key_file) else ''

    done = set()
    if os.path.exists(a.out):
        for line in open(a.out, encoding='utf8'):
            try:
                done.add(json.loads(line)['q'])
            except (ValueError, KeyError):
                pass
    qs = [q.strip() for q in open(a.questions, encoding='utf8') if q.strip() and q.strip() not in done]
    if a.limit:
        qs = qs[:a.limit]
    t0, n_tok, n_calls = time.time(), 0, 0
    for i, q in enumerate(qs):
        msgs = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': q}]
        try:
            for _ in range(a.max_turns):
                m, n = chat(a.url, key, msgs, tools, a.max_tokens)
                n_tok += n
                turn = {'role': 'assistant', 'content': m.get('content') or '', 'reasoning_content': m.get('reasoning_content') or ''}
                calls = m.get('tool_calls') or []
                if calls:
                    turn['tool_calls'] = calls
                msgs.append(turn)
                if not calls:
                    break
                for c in calls:
                    name = c['function']['name']
                    try:
                        args = c['function'].get('arguments') or '{}'
                        args = json.loads(args) if isinstance(args, str) else args
                        res = mcp.call('tools/call', {'name': name[len(prefix):] if name.startswith(prefix) else name, 'arguments': args})
                        text = '\n'.join(x.get('text', '') for x in res.get('content', []) if x.get('type') == 'text')
                    except (RuntimeError, ValueError, urllib.error.URLError) as e:
                        text = f'Error: {e}'
                    n_calls += 1
                    msgs.append({'role': 'tool', 'tool_call_id': c.get('id', ''), 'content': text[:a.max_result_chars]})
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            print(f'question {i}: server error ({type(e).__name__}), skipped', flush=True)
            continue
        with open(a.out, 'a', encoding='utf8') as f:
            f.write(json.dumps({'q': q, 'messages': msgs, 'tools': tools, '_gen': True}, ensure_ascii=False) + '\n')
        if (i + 1) % 10 == 0 or i + 1 == len(qs):
            print(f'{i + 1}/{len(qs)} conversations, {n_calls} tool calls, {n_tok / (time.time() - t0):.1f} gen tok/s', flush=True)


if __name__ == '__main__':
    main()
