#!/usr/bin/env python3
# Generates model-written replies for the prompt-style calibration sources (HOPE expert pruning): the papers found
# calibration works best when the assistant turns come from the model itself. Prompts go to a running
# llama-server (OpenAI API); each reply and its reasoning are appended to OUT_DIR/<source>.jsonl as
# {"prompt", "reply", "reasoning", "_gen": true}, which hope_calib.py reads with a "pairs" source.
#
#   hope_gen.py --config gen.json --out OUT_DIR [--url http://127.0.0.1:8091] [--key-file KEYS]
#
# Each config source: {"name", "files", "template" (str.format over the row's columns), "tokens" (stop after this
# many generated tokens), optional "where_not", "min_chars", "max_prompt_chars"}. Resumable: prompts already in the
# output file are skipped. The API key is read from the first line of --key-file and never printed.
import argparse, json, os, random, re, sys, time, urllib.error, urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hope_calib import rows


def chat(url, key, model, prompt, max_tokens, timeout):
    body = json.dumps({'model': model, 'messages': [{'role': 'user', 'content': prompt}], 'max_tokens': max_tokens,
                       'stream': False}).encode()
    req = urllib.request.Request(url + '/v1/chat/completions', data=body, method='POST',
                                 headers={'Content-Type': 'application/json', 'Authorization': f'Bearer {key}'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    m = d['choices'][0]['message']
    return m.get('content') or '', m.get('reasoning_content') or '', d.get('usage', {}).get('completion_tokens', 0), \
        d['choices'][0].get('finish_reason')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--url', default='http://127.0.0.1:8091')
    p.add_argument('--key-file', default=os.path.expanduser('~/llm/fn-server.keys'))
    p.add_argument('--model', default='flash-next')
    p.add_argument('--max-tokens', type=int, default=4096)
    p.add_argument('--timeout', type=int, default=900)
    p.add_argument('--seed', type=int, default=1)
    a = p.parse_args()

    key = ''
    if os.path.exists(a.key_file):
        key = open(a.key_file).readline().strip()
    os.makedirs(a.out, exist_ok=True)
    conf = json.load(open(a.config))
    # round-robin over sources so that a run stopped early still covers all of them
    state = []
    for s in conf['sources']:
        path = os.path.join(a.out, s['name'] + '.jsonl')
        done, toks = set(), 0
        if os.path.exists(path):
            for line in open(path, encoding='utf8'):
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                done.add(r['prompt'])
                toks += r.get('tokens', 0)
        it = rows(s['files'], None, random.Random(f'{a.seed}-{s["name"]}'))
        state.append(dict(cfg=s, path=path, done=done, toks=toks, it=it))
    t0, n_req, n_tok = time.time(), 0, 0
    while True:
        active = [st for st in state if st['toks'] < st['cfg']['tokens'] and st['it'] is not None]
        if not active:
            break
        st = min(active, key=lambda st: st['toks'] / st['cfg']['tokens'])
        s = st['cfg']
        prompt = None
        for row in st['it']:
            if s.get('where_not') and any(row.get(k) in v for k, v in s['where_not'].items()):
                continue
            try:
                pr = s['template'].format(**{k: (v if isinstance(v, str) else json.dumps(v, ensure_ascii=False))
                                             for k, v in row.items() if not k.startswith('_')}).strip()
            except (KeyError, IndexError):
                continue
            pr = pr[:s.get('max_prompt_chars', 24000)]
            if len(pr) >= s.get('min_chars', 40) and pr not in st['done']:
                prompt = pr
                break
        if prompt is None:
            st['it'] = None
            continue
        for attempt in range(5):
            try:
                reply, reasoning, n, fin = chat(a.url, key, a.model, prompt, a.max_tokens, a.timeout)
                break
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
                print(f'request failed ({type(e).__name__}), retry in 60 s', flush=True)
                time.sleep(60)
        else:
            sys.exit('server unreachable, stopping (rerun to resume)')
        st['done'].add(prompt)
        st['toks'] += n
        n_req += 1
        n_tok += n
        with open(st['path'], 'a', encoding='utf8') as f:
            f.write(json.dumps({'prompt': prompt, 'reply': reply, 'reasoning': reasoning, 'tokens': n,
                                'finish': fin, '_gen': True}, ensure_ascii=False) + '\n')
        if n_req % 20 == 0:
            el = time.time() - t0
            prog = ', '.join(f'{x["cfg"]["name"]} {x["toks"] / 1e3:.0f}k/{x["cfg"]["tokens"] / 1e3:.0f}k' for x in state)
            print(f'{n_req} replies, {n_tok / el:.1f} gen tok/s; {prog}', flush=True)
    print('done')


if __name__ == '__main__':
    main()
