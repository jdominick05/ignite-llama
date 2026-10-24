#!/usr/bin/env python3
"""Rare-token copy reliability of a running llama-server under sampling (expert-pruning check).

Deep expert prunes can keep aggregate KLD and greedy benchmarks intact while sampled output garbles rare strings
(sh0wie's Flash-Next REAP ladder: a rare name survived 9/10 seeds at 288 experts, 1/10 at 256). This test puts
rare strings an agent must copy exactly (hashes, CVE ids, addresses, mangled symbols, paths, odd identifiers, rare
names) into short realistic contexts, asks for a reply that needs the string, samples each prompt with several
seeds at the daily settings (temp 1.0, top-k 20, top-p 0.95) and scores exact copies.

    rare_tokens.py LABEL [--port 8092] [--prompts 32] [--seeds 8] [--jobs 4] [--out ~/llm/logs/rare-tokens.jsonl]

--jobs N sends N requests at once (start the server with -np N).

Prompts are synthetic and generated from a fixed seed, so every model sees the same set. Prints exact / near-miss
(edit distance 1-3, i.e. a garbled copy) / missing rates per string kind and overall. Misses are split by cause, since a
pruned model can lose a string without garbling it: refused (refusal phrasing in the reply), cut off (max_tokens hit,
usually while thinking) or omitted; per-template exact rates show prompts that trigger refusals.
"""
import argparse, gzip, threading, base64, json, os, random, re, string, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict

SAMPLING = dict(temperature=1.0, top_k=20, top_p=0.95, min_p=0.0)
REFUSAL = re.compile(r"\b(I can(no|')t|I('m| am) (not able|unable)|I won't|I will not|I don't have access)\b", re.IGNORECASE)


def hexs(r, n):
    return ''.join(r.choice('0123456789abcdef') for _ in range(n))


def ident(r, n):
    return r.choice(string.ascii_lowercase) + ''.join(r.choice(string.ascii_lowercase + string.digits) for _ in range(n - 1))


def rare_name(r):
    # pronounceable but non-dictionary surnames, the failure case reported for pruned Flash-Next builds
    syl = ['wex', 'tor', 'quin', 'bra', 'zel', 'mor', 'vash', 'ilt', 'kae', 'drum', 'oss', 'phy', 'lund', 'grav', 'ent']
    return (''.join(r.choice(syl) for _ in range(3)).capitalize() + '-' + ''.join(r.choice(syl) for _ in range(2)).capitalize())


KINDS = {
    'sha256': lambda r: hexs(r, 64),
    'md5': lambda r: hexs(r, 32),
    'cve': lambda r: f'CVE-{r.randint(2015, 2026)}-{r.randint(1000, 99999)}',
    'addr': lambda r: '0x' + r.choice(['7ffd', '5555', '7f3a', '0040']) + hexs(r, 8),
    'symbol': lambda r: f'_ZN{len(a := ident(r, 7))}{a}{len(b := ident(r, 9))}{b}E{r.choice(["v", "i", "Pc", "RKS_"])}',
    'path': lambda r: f'/opt/{ident(r, 6)}/lib/{ident(r, 5)}_{hexs(r, 4)}/{ident(r, 8)}.so.{r.randint(1, 9)}',
    'b64': lambda r: base64.b64encode(bytes(r.getrandbits(8) for _ in range(18))).decode(),
    'ident': lambda r: f'{ident(r, 5)}_{ident(r, 4)}{r.randint(10, 99)}_{ident(r, 6)}',
    'name': rare_name,
}

TEMPLATES = [
    ('A scanner log contains this line:\n\n  [{ts}] ALERT host=build-07 indicator={s} source=edr\n\n'
     'Write the one-line grep command that finds every log line mentioning this exact indicator.'),
    ('Here is a JSON finding from a triage tool:\n\n  {{"id": {n}, "severity": "high", "artifact": "{s}", "status": "open"}}\n\n'
     'Reply with only the value of the "artifact" field, exactly as written.'),
    ('While reversing a binary I noted: the interesting function is referenced as {s} in the decompiler output.\n'
     'Write a short note for my report that names it exactly and says it needs a second look.'),
    ('Ticket #{n}: please add {s} to the blocklist and confirm.\n\nWrite the confirmation message, quoting the entry exactly.'),
]


def make_prompts(n, seed=1234):
    r = random.Random(seed)
    kinds = list(KINDS)
    out = []
    for i in range(n):
        k = kinds[i % len(kinds)]
        s = KINDS[k](r)
        ti = (i // len(kinds)) % len(TEMPLATES); t = TEMPLATES[ti]
        out.append((k, s, ti, t.format(s=s, n=r.randint(1000, 9999), ts=f'2026-10-{r.randint(10, 28)}T{r.randint(0, 23):02d}:{r.randint(0, 59):02d}Z')))
    return out


def edit_distance_window(s, text, max_d=3):
    # smallest edit distance between s and any substring of text of length len(s)-max_d .. len(s)+max_d
    # only windows anchored where the text repeats the first or last 4 characters (a garbled copy keeps one end)
    best = max_d + 1
    n = len(s)
    starts = set()
    for anchor, off in ((s[:4], 0), (s[-4:], n - 4)):
        i = text.find(anchor)
        while i >= 0:
            for d in range(-max_d, max_d + 1):
                starts.add(i - off + d)
            i = text.find(anchor, i + 1)
    for L in range(max(1, n - max_d), n + max_d + 1):
        for i in sorted(x for x in starts if 0 <= x <= len(text) - 1):
            w = text[i:i + L]
            if abs(len(w) - n) > max_d:
                continue
            # banded Levenshtein
            prev = list(range(len(w) + 1))
            for a in range(1, n + 1):
                cur = [a] + [0] * len(w)
                for b in range(1, len(w) + 1):
                    cur[b] = min(prev[b] + 1, cur[b - 1] + 1, prev[b - 1] + (s[a - 1] != w[b - 1]))
                prev = cur
            best = min(best, prev[-1])
            if best <= 1:
                return best
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('label')
    ap.add_argument('--port', type=int, default=8092)
    ap.add_argument('--prompts', type=int, default=32)
    ap.add_argument('--seeds', type=int, default=8)
    ap.add_argument('--seed-start', type=int, default=0, help='first seed (to continue a probe run without repeats)')
    ap.add_argument('--max-tokens', type=int, default=400)
    ap.add_argument('--jobs', type=int, default=4)
    ap.add_argument('--progress', type=int, default=100, help='print the running score every N samples (0 = off)')
    ap.add_argument('--key-file', default=os.path.expanduser('~/llm/fn-server.keys'))
    ap.add_argument('--out', default=os.path.expanduser('~/llm/logs/rare-tokens.jsonl'))
    ap.add_argument('--dump', help='also write every sample (prompt ids, generated ids, top logprobs) to this .jsonl.gz')
    ap.add_argument('--top-logprobs', type=int, default=32, help='with --dump: top-k raw logprobs per generated token')
    a = ap.parse_args()
    key = open(a.key_file).readline().strip() if os.path.exists(a.key_file) else ''
    stats = defaultdict(lambda: [0, 0, 0])   # kind -> exact, near, miss
    by_tpl = defaultdict(lambda: [0, 0])     # template index -> exact, total
    causes = defaultdict(int)                # miss cause -> count
    t0 = time.time()

    def post(path, body):
        req = urllib.request.Request(f'http://127.0.0.1:{a.port}{path}', data=json.dumps(body).encode(),
                                     headers={'Content-Type': 'application/json', 'Authorization': f'Bearer {key}'})
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.load(r)

    dump = gzip.open(a.dump, 'at') if a.dump else None
    dump_lock = threading.Lock()

    def one(job):
        (k, s, ti, prompt), seed = job
        body = {'messages': [{'role': 'user', 'content': prompt}], 'max_tokens': a.max_tokens, 'seed': seed,
                'reasoning_effort': 'low', **SAMPLING}
        if dump: body.update(logprobs=True, top_logprobs=a.top_logprobs)   # raw softmax (post_sampling_probs off)
        c = post('/v1/chat/completions', body)['choices'][0]
        text = c['message'].get('content') or ''
        res = 0 if s in text else 1 if edit_distance_window(s, text) <= 3 else 2
        cause = None if res < 2 else 'cut_off' if c.get('finish_reason') == 'length' else \
            'refused' if REFUSAL.search(text) else 'omitted'
        if dump:
            tpl = post('/apply-template', {k2: body[k2] for k2 in ('messages', 'reasoning_effort')})['prompt']
            ids = post('/tokenize', {'content': tpl, 'add_special': False, 'parse_special': True})['tokens']
            lp = c['logprobs']['content']
            rec = {'kind': k, 'tpl': ti, 'seed': seed, 'res': res, 'cause': cause, 'finish': c.get('finish_reason'),
                   'prompt_ids': ids, 'gen_ids': [t['id'] for t in lp], 'gen_lp': [round(t['logprob'], 4) for t in lp],
                   'top_ids': [[u['id'] for u in t['top_logprobs']] for t in lp],
                   'top_lp': [[round(u['logprob'], 4) for u in t['top_logprobs']] for t in lp]}
            with dump_lock: dump.write(json.dumps(rec) + '\n')
        return k, ti, res, cause

    jobs = [(p, seed) for p in make_prompts(a.prompts) for seed in range(a.seed_start, a.seed_start + a.seeds)]
    with ThreadPoolExecutor(a.jobs) as ex:
        for i, (k, ti, res, cause) in enumerate(ex.map(one, jobs), 1):
            stats[k][res] += 1
            by_tpl[ti][0] += res == 0; by_tpl[ti][1] += 1
            if cause: causes[cause] += 1
            if a.progress and (i % a.progress == 0) and i < len(jobs):
                # running score on stderr, so a pipe that keeps only the final line still shows it
                ex_n = sum(v[0] for v in stats.values()); dt = time.time() - t0
                print(f'{a.label}: {i}/{len(jobs)} done, exact so far {ex_n / i * 100:.1f}%, '
                      f'{dt:.0f} s, ETA {dt / i * (len(jobs) - i):.0f} s', file=sys.stderr, flush=True)
    if dump: dump.close()
    tot = [sum(v[i] for v in stats.values()) for i in range(3)]
    n = sum(tot)
    row = {'label': a.label, 'prompts': a.prompts, 'seeds': a.seeds, 'exact': tot[0] / n, 'near_miss': tot[1] / n,
           'missing': tot[2] / n, 'per_kind': {k: v for k, v in stats.items()}, 'miss_causes': dict(causes),
           'per_template': {str(t): v for t, v in sorted(by_tpl.items())}, 'seconds': round(time.time() - t0)}
    with open(a.out, 'a') as f:
        f.write(json.dumps(row) + '\n')
    print(f'{a.label}: exact {tot[0] / n * 100:.1f}%  garbled {tot[1] / n * 100:.1f}%  missing {tot[2] / n * 100:.1f}%  '
          f'({n} samples, {row["seconds"]} s)')
    print('  misses: ' + ', '.join(f'{c} {n}' for c, n in sorted(causes.items())) + '   exact per template: ' +
          ', '.join(f'T{t} {v[0]}/{v[1]}' for t, v in sorted(by_tpl.items())))
    for k, v in sorted(stats.items()):
        m = sum(v)
        print(f'  {k:7s} exact {v[0]}/{m}  garbled {v[1]}  missing {v[2]}')


if __name__ == '__main__':
    main()
