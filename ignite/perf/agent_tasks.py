#!/usr/bin/env python3
# Agent tasks run to completion against llama-server, scored for correctness and cost (total completion tokens incl.
# reasoning, turns). The model's tool calls execute in a sandbox: a fresh temp dir per run, bubblewrap with no network,
# read-only system, 10 s per command, and only `python3 <file>.py`, `ls` and `cat <file>` allowed.
# Same tasks and seeds for every model, so two models' token costs compare directly.
#
#   agent_tasks.py LABEL [--port 8091] [--seeds 8] [--max-turns 10] [--out ~/llm/logs/agent-tasks.jsonl]
import argparse, hashlib, json, os, re, shlex, shutil, subprocess, tempfile, time, urllib.request

TOOLS = [
    {'type': 'function', 'function': {'name': 'terminal', 'description': 'Run a shell command in the current directory and return its output.',
     'parameters': {'type': 'object', 'properties': {'command': {'type': 'string', 'description': 'The command to run'}}, 'required': ['command']}}},
    {'type': 'function', 'function': {'name': 'write_file', 'description': 'Write a file in the current directory.',
     'parameters': {'type': 'object', 'properties': {'path': {'type': 'string'}, 'content': {'type': 'string'}}, 'required': ['path', 'content']}}},
]
SYSTEM = 'You are a helpful agent with terminal and file tools. Use tools to complete tasks, then give a short final answer.'
SCORES = 'name,score\nalice,82\nbob,91\ncarol,77\ndave,88\nerin,91.5\n'
FILES = {'notes.txt': 'patch tuesday checklist\n', 'blob.bin': bytes(range(256)) * 3, 'config.ini': '[core]\nmode=strict\n'}


def setup(task, d):
    if task == 'scores':
        open(os.path.join(d, 'scores.csv'), 'w').write(SCORES)
    if task == 'hashes':
        for n, c in FILES.items():
            open(os.path.join(d, n), 'wb').write(c.encode() if isinstance(c, str) else c)


TASKS = {
    'fizzbuzz': 'Create fizzbuzz.py that prints FizzBuzz for 1 to 30, run it, and tell me the last 3 lines of output.',
    'scores': 'There is a file scores.csv in the current directory. Write and run a Python script that prints the average score and the name of the highest scorer.',
    'hashes': 'Write hashes.py that prints the SHA-256 of every file in the current directory as a table, then run it.',
}


def check(task, final, outputs):
    # final: the last assistant text; outputs: all command outputs of the run
    if task == 'fizzbuzz':
        return all(re.search(rf'(?<![\w.]){w}(?![\w.])', final) for w in ('28', '29', 'FizzBuzz'))
    if task == 'scores':
        avg = sum(float(l.split(',')[1]) for l in SCORES.strip().split('\n')[1:]) / 5
        nums = [float(x) for x in re.findall(r'\d+(?:\.\d+)?', final)]
        return 'erin' in final.lower() and any(abs(x - avg) < 0.05 for x in nums)
    if task == 'hashes':
        want = [hashlib.sha256(c.encode() if isinstance(c, str) else c).hexdigest() for c in FILES.values()]
        return all(h in outputs for h in want)


def run_cmd(cmd, d):
    try:
        argv = shlex.split(cmd)
    except ValueError:
        return 'error: could not parse the command'
    ok = (len(argv) == 1 and argv[0] == 'ls') or \
         (len(argv) == 2 and argv[0] in ('python3', 'python') and re.fullmatch(r'[\w.-]+\.py', argv[1])) or \
         (len(argv) == 2 and argv[0] == 'cat' and re.fullmatch(r'[\w.-]+', argv[1]))
    if not ok:
        return 'error: this sandbox only allows `python3 <file>.py`, `ls` and `cat <file>` in the current directory'
    if argv[0] == 'python':
        argv[0] = 'python3'
    bw = ['bwrap', '--ro-bind', '/usr', '/usr', '--ro-bind', '/lib', '/lib', '--ro-bind', '/lib64', '/lib64',
          '--ro-bind', '/bin', '/bin', '--proc', '/proc', '--dev', '/dev', '--tmpfs', '/tmp', '--bind', d, '/work',
          '--chdir', '/work', '--unshare-all', '--die-with-parent', '--clearenv', '--setenv', 'PATH', '/usr/bin:/bin']
    try:
        p = subprocess.run(bw + argv, capture_output=True, timeout=10)
    except subprocess.TimeoutExpired:
        return 'error: timed out after 10 s'
    out = (p.stdout + p.stderr).decode(errors='replace')
    return out[-4000:] if out else f'(no output, exit code {p.returncode})'


def episode(task, seed, a, key):
    d = tempfile.mkdtemp(prefix='agent-task-')
    try:
        setup(task, d)
        msgs = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': TASKS[task]}]
        tokens = turns = 0
        outputs, final, status = [], '', 'max-turns'
        for turns in range(1, a.max_turns + 1):
            body = {'messages': msgs, 'tools': TOOLS, 'max_tokens': a.max_tokens, 'seed': seed * 100 + turns}
            req = urllib.request.Request(f'http://127.0.0.1:{a.port}/v1/chat/completions', data=json.dumps(body).encode(),
                                         headers={'Content-Type': 'application/json', 'Authorization': f'Bearer {key}'})
            r = json.load(urllib.request.urlopen(req, timeout=900))
            c = r['choices'][0]; m = c['message']
            tokens += r['usage']['completion_tokens']
            if c['finish_reason'] == 'length':
                status = 'runaway'; break
            calls = m.get('tool_calls') or []
            msgs.append({k: v for k, v in m.items() if k in ('role', 'content', 'tool_calls', 'reasoning_content')})
            if not calls:
                final, status = m.get('content') or '', 'answered'; break
            for tc in calls:
                try:
                    args = json.loads(tc['function']['arguments'])
                except (json.JSONDecodeError, TypeError):
                    args = {}
                name = tc['function']['name']
                if name == 'write_file' and re.fullmatch(r'[\w.-]+', str(args.get('path', ''))):
                    open(os.path.join(d, args['path']), 'w').write(str(args.get('content', '')))
                    res = f'wrote {args["path"]}'
                elif name == 'terminal':
                    res = run_cmd(str(args.get('command', '')), d)
                    outputs.append(res)
                else:
                    res = 'error: invalid tool call'
                msgs.append({'role': 'tool', 'tool_call_id': tc.get('id', ''), 'content': res})
        correct = status == 'answered' and check(task, final, '\n'.join(outputs))
        return {'task': task, 'seed': seed, 'status': status, 'correct': correct, 'tokens': tokens, 'turns': turns}
    finally:
        shutil.rmtree(d, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('label')
    ap.add_argument('--port', type=int, default=8091)
    ap.add_argument('--seeds', type=int, default=8)
    ap.add_argument('--max-turns', type=int, default=10)
    ap.add_argument('--max-tokens', type=int, default=3000)
    ap.add_argument('--key-file', default=os.path.expanduser('~/llm/fn-server.keys'))
    ap.add_argument('--out', default=os.path.expanduser('~/llm/logs/agent-tasks.jsonl'))
    a = ap.parse_args()
    key = next(l.strip() for l in open(a.key_file) if l.strip() and not l.lstrip().startswith('#'))
    t0 = time.time()
    rows = [episode(t, s, a, key) for t in TASKS for s in range(a.seeds)]
    n = len(rows); ok = [r for r in rows if r['correct']]
    print(f"{a.label}: correct {len(ok)}/{n}, runaways {sum(r['status'] == 'runaway' for r in rows)}, "
          f"tokens mean {sum(r['tokens'] for r in rows) / n:.0f} (correct runs {sum(r['tokens'] for r in ok) / max(len(ok), 1):.0f}), "
          f"turns mean {sum(r['turns'] for r in rows) / n:.1f}, {time.time() - t0:.0f} s")
    for t in TASKS:
        tr = [r for r in rows if r['task'] == t]
        print(f"  {t:9s} correct {sum(r['correct'] for r in tr)}/{len(tr)}, tokens {[r['tokens'] for r in tr]}")
    with open(a.out, 'a') as f:
        f.write(json.dumps({'label': a.label, 'time': time.strftime('%F %T'), 'rows': rows}) + '\n')


if __name__ == '__main__':
    main()
