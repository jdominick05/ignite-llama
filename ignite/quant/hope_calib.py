#!/usr/bin/env python3
# Builds the calibration pool for llama-hope-collect (HOPE expert pruning): agent sessions, agent trajectories,
# security material, code and some general text, rendered with the model's own chat template.
#
#   hope_calib.py --model FN.gguf --tokenizer llama-tokenize --config pool.json --tokens 10000000 --out OUT_DIR
#
# The config names the sources, their share of --tokens and their options (see SOURCES below and
# hope_pool.example.json). Output:
#   OUT_DIR/pool.jsonl   one document per line: {"text", "set", "src"}; set = 2*half + gen (half A=0 / B=1,
#                        gen=1 for documents whose replies were written by the model, see hope_gen.py)
#   OUT_DIR/stats.json   per source: measured chars/token, tokens per set, groups, top group share, drops
#
# Rules:
#   - Documents are interleaved so that every prefix of the pool has the target mix (deficit round-robin).
#   - Chat documents are kept whole up to --max-doc-tokens and split at message boundaries beyond it; plain text
#     and code are split at blank lines. No text between pieces is dropped.
#   - Within a source, no group (project, repo, dataset subset) takes more than its cap of the source budget;
#     the budget a capped group leaves goes to the other groups.
#   - Halves are assigned per source by split key (session, SWE instance, writeup, repo directory), each key to the
#     half that is smaller within that source.
#   - Documents overlapping the --exclude texts (normalized 64-char shingles, more than --exclude-frac of
#     positions) and documents whose assistant turns refuse are dropped.
# Prints aggregate statistics only: the sessions are private.
import argparse, glob, json, math, os, random, re, subprocess, sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- chat template

def load_template(model_path):
    sys.path.insert(0, os.path.join(HERE, '..', '..', 'gguf-py'))
    import gguf
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    r = gguf.GGUFReader(model_path)
    f = r.fields['tokenizer.chat_template']
    src = bytes(f.parts[f.data[0]]).decode()

    def raise_exception(msg):
        raise ValueError(msg)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=['jinja2.ext.loopcontrols'])
    env.globals['raise_exception'] = raise_exception
    env.filters['tojson'] = lambda v, **kw: json.dumps(v, ensure_ascii=False)
    tmpl = env.from_string(src)

    def render(messages, tools=None, effort='medium'):
        # effort: the reasoning_effort the server would pass (the template's own default is xhigh)
        kw = dict(messages=messages, add_generation_prompt=False, bos_token='', eos_token='', reasoning_effort=effort)
        if tools:
            kw['tools'] = tools
        return tmpl.render(**kw)
    return render


def split_think(content):
    m = re.match(r'\s*<think>(.*?)</think>\s*(.*)', content or '', re.S)
    return (m.group(1).strip(), m.group(2)) if m else ('', content or '')


def tool_args(raw):
    if isinstance(raw, dict):
        return raw
    try:
        v = json.loads(raw) if raw else {}
    except (ValueError, TypeError):
        return {'arguments': raw}
    return v if isinstance(v, dict) else {'value': v}


def oai_messages(msgs, keep_system=False):
    # OpenAI-style messages (role/content/tool_calls, optional reasoning_content) -> template messages
    out = []
    for m in msgs or []:
        if not isinstance(m, dict):
            continue
        role = m.get('role') or m.get('from')
        role = {'human': 'user', 'gpt': 'assistant', 'observation': 'tool', 'function': 'tool'}.get(role, role)
        if role == 'system' and not keep_system:
            continue
        if role not in ('system', 'user', 'assistant', 'tool'):
            continue
        content = m.get('content') if 'content' in m else m.get('value')
        if isinstance(content, list):
            content = '\n'.join(c.get('text', '') for c in content if isinstance(c, dict))
        mm = {'role': role, 'content': content or ''}
        if role == 'assistant':
            r = m.get('reasoning_content') or ''
            if not r:
                r, mm['content'] = split_think(mm['content'])
            mm['reasoning_content'] = r
            calls = [{'function': {'name': (tc.get('function') or {}).get('name', ''),
                                   'arguments': tool_args((tc.get('function') or {}).get('arguments'))}}
                     for tc in (m.get('tool_calls') or []) if isinstance(tc, dict)]
            if calls:
                mm['tool_calls'] = calls
        out.append(mm)
    return out


# ---------------------------------------------------------------- readers

def _file_rows(p):
    if p.endswith('.parquet'):
        import pyarrow.parquet as pq
        for b in pq.ParquetFile(p).iter_batches(batch_size=64):
            yield from b.to_pylist()
    elif p.endswith('.json'):
        try:
            v = json.load(open(p, encoding='utf8', errors='replace'))
        except ValueError:
            return
        yield from (v if isinstance(v, list) else [v])
    else:
        with open(p, encoding='utf8', errors='replace') as f:
            for line in f:
                try:
                    yield json.loads(line)
                except ValueError:
                    continue


def rows(files, limit=None, rng=None):
    # parquet / jsonl / json rows from one or more globs; with rng, files are visited in random order and read in
    # rotation (up to 64 open at a time), so an early stop still samples every file
    pats = files if isinstance(files, list) else [files]
    paths = sorted({p for pat in pats for p in glob.glob(os.path.expanduser(pat), recursive=True)})
    n = 0
    if not rng:
        for p in paths:
            for r in _file_rows(p):
                if isinstance(r, dict):
                    r['_path'] = p
                yield r
                n += 1
                if limit and n >= limit:
                    return
        return
    rng.shuffle(paths)
    pending, open_its = list(paths), []
    while pending or open_its:
        while pending and len(open_its) < 64:
            p = pending.pop()
            open_its.append((p, _file_rows(p)))
        for item in list(open_its):
            p, it = item
            for _ in range(8):
                r = next(it, None)
                if r is None:
                    open_its.remove(item)
                    break
                if isinstance(r, dict):
                    r['_path'] = p
                yield r
                n += 1
                if limit and n >= limit:
                    return


# ---------------------------------------------------------------- code trees

CODE_EXT = {'.c', '.h', '.cl', '.cpp', '.hpp', '.cc', '.hh', '.hxx', '.cxx', '.inc', '.rs', '.py', '.pyi', '.java',
            '.kt', '.kts', '.scala', '.swift', '.m', '.mm', '.ts', '.tsx', '.js', '.jsx', '.mjs', '.cjs', '.vue',
            '.go', '.sh', '.bash', '.zsh', '.awk', '.rb', '.lua', '.nse', '.pl', '.pm', '.php', '.cs', '.ps1',
            '.psm1', '.psd1', '.bat', '.cmd', '.sql', '.html', '.htm', '.css', '.scss', '.S', '.s', '.asm', '.sol',
            '.tf', '.hcl', '.proto', '.y', '.l', '.lds', '.ld', '.cmake', '.gradle', '.toml', '.ini', '.conf',
            '.cnf', '.cfg', '.xml', '.yaml', '.yml', '.json', '.j2', '.erb', '.tmpl', '.diff', '.patch',
            '.slaspec', '.sinc', '.cspec', '.pspec', '.yar', '.yara', '.rules'}
DOC_EXT = {'.md', '.rst', '.txt', '.adoc'}
CODE_NAMES = {'Makefile', 'GNUmakefile', 'Kbuild', 'Kconfig', 'CMakeLists.txt', 'Containerfile', 'Jenkinsfile',
              'Vagrantfile', 'Rakefile', 'Gemfile', 'go.mod', 'Cargo.toml', 'pyproject.toml', 'nginx.conf',
              '.htaccess'}
SKIP_DIRS = {'node_modules', 'vendor', '_vendor', 'third_party', 'thirdparty', '3rdparty', 'deps', 'external',
             'extern', 'site-packages', '__pycache__', 'testdata', '.git', '.svn', '.idea', '.vscode'}
SKIP_FILE_RE = re.compile(
    r'(^|/)(package-lock\.json|yarn\.lock|pnpm-lock\.yaml|Cargo\.lock|poetry\.lock|composer\.lock|go\.sum|'
    r'LICENSE[^/]*|COPYING[^/]*|contributors\.json|expected_output\.json|result_output\.json|fp_[^/]*\.txt)$|'
    r'(^|/)(wordlists?|payloads)/[^/]*\.txt$|\.min\.(js|css)$|\.bundle\.js$|\.map$|\.pb\.go$|_pb2\.py$|'
    r'(^|/)z(errors|syscall|types)_[^/]*\.go$|zz_generated|(^|/)asic_reg/|_sh_mask\.h$|_offset\.h$|'
    r'shellcraft/templates/.*/syscalls/')
GENERATED_RE = re.compile(r'Code generated .* DO NOT EDIT|@generated|[Aa]uto-?generated|automatically generated|'
                          r'[Gg]enerated by ')
TEST_RE = re.compile(r'(^|/)(tests?|spec|__tests__)(/|$)|(_test|\.test|\.spec|_spec)\.[A-Za-z]+$|(^|/)test_[^/]*$')
LICENSE_RE = re.compile(r'copyright|license|licence|spdx|permission is hereby granted|all rights reserved|'
                        r'warranty|gnu general public', re.I)


def is_code_file(fn, exts):
    if fn in CODE_NAMES or fn.startswith('Dockerfile') or fn.startswith('docker-compose'):
        return True
    return os.path.splitext(fn)[1] in exts


def strip_license_header(text):
    # drops a leading comment block that talks about copyright / licences
    m = re.match(r'\s*(/\*.*?\*/|(?:(?://|#|--|;)[^\n]*\n)+)', text, re.S)
    if m and LICENSE_RE.search(m.group(1)[:3000]):
        return text[m.end():].lstrip('\n')
    return text


def looks_generated_or_minified(text):
    if GENERATED_RE.search(text[:3000]):
        return True
    lines = text.split('\n')
    return max(map(len, lines)) > 1000 or len(text) / max(1, len(lines)) > 200


def tree_files(root, exts, rng):
    # (path, rel) of eligible files in random order; vendored code and build output are skipped anywhere
    root = os.path.expanduser(root).rstrip('/')
    out = []
    for d, subdirs, files in os.walk(root):
        rel_d = os.path.relpath(d, root)
        keep = []
        for s in subdirs:
            if s in SKIP_DIRS or (s.startswith('.') and s != '.github'):
                continue
            if s in ('build', 'target', 'dist', 'out') and (
                    os.path.exists(os.path.join(d, 'Cargo.toml')) or os.path.exists(os.path.join(d, s, 'CMakeCache.txt'))
                    or os.path.isdir(os.path.join(d, s, 'release')) or os.path.isdir(os.path.join(d, s, 'debug'))):
                continue
            keep.append(s)
        subdirs[:] = keep
        in_github = rel_d.split('/')[0] == '.github'
        for fn in files:
            if not is_code_file(fn, exts):
                continue
            rel = os.path.normpath(os.path.join(rel_d, fn))
            if (in_github and '/workflows/' not in '/' + rel) or SKIP_FILE_RE.search(rel):
                continue
            out.append((os.path.join(d, fn), rel))
    rng.shuffle(out)
    return out


def src_code(cfg, rng, render):
    # per tree (weight ~ sqrt of size): tests <= 15%, at most cfg['per_dir'] files per directory (near-duplicate
    # guard), a slice of <= cfg['file_chars'] chars starting at a blank line; group = tree, key = top-level dir
    exts = CODE_EXT | (DOC_EXT if cfg.get('docs') else set())
    file_chars = cfg.get('file_chars', 8000)
    per_dir = cfg.get('per_dir', 6)
    budget_chars = cfg['_budget_tokens'] * 3.3 * 3   # oversample 3x so caps and filters have room
    trees = [os.path.expanduser(t) for t in cfg['trees']]
    listing = {t: tree_files(t, exts, random.Random(f'{t}')) for t in trees}
    sizes = {t: max(1, sum(os.path.getsize(p) for p, _ in v[:3000]) * max(1, len(v)) / max(1, min(3000, len(v))))
             for t, v in listing.items()}
    wsum = sum(math.sqrt(v) for v in sizes.values())
    for t in trees:
        want = budget_chars * math.sqrt(sizes[t]) / wsum
        name = cfg.get('names', {}).get(t, os.path.basename(t.rstrip('/')))
        got, tests, per = 0, 0, Counter()
        files = listing[t]
        rng.shuffle(files)
        for p, rel in files:
            if got >= want:
                break
            try:
                if not 200 < os.path.getsize(p) < 400_000:
                    continue
                text = open(p, encoding='utf8').read()
            except (OSError, UnicodeDecodeError):
                continue
            if looks_generated_or_minified(text):
                continue
            is_test = bool(TEST_RE.search(rel))
            if is_test and tests > 0.15 * max(1, got):
                continue
            dkey = os.path.dirname(rel)
            if per[dkey] >= per_dir:
                continue
            per[dkey] += 1
            text = strip_license_header(text)
            if len(text) > file_chars:
                starts = [0] + [m.end() for m in re.finditer(r'\n\s*\n', text[:len(text) - file_chars])]
                s0 = rng.choice(starts)
                text = text[s0:s0 + file_chars]
            if len(text) < 200:
                continue
            got += len(text)
            tests += len(text) if is_test else 0
            top = rel.split('/')[0] if '/' in rel else '.'
            yield dict(group=name, key=f'{name}/{top}', path=f'{name}/{rel}', text=text, kind='file')


# ---------------------------------------------------------------- sources

def src_sessions(cfg, rng, render):
    root = os.path.expanduser(cfg['path'])
    for path in sorted(glob.glob(os.path.join(root, '**', '*.jsonl'), recursive=True)):
        msgs, cwd, sid, cur = [], None, path, None

        def flush():
            nonlocal cur
            if cur:
                if not cur['tool_calls']:
                    del cur['tool_calls']
                msgs.append(cur)
                cur = None

        for line in open(path, encoding='utf8', errors='replace'):
            try:
                d = json.loads(line)
            except ValueError:
                continue
            cwd = cwd or d.get('cwd')
            sid = d.get('sessionId') or sid
            m = d.get('message')
            if not isinstance(m, dict):
                continue
            parts = [p for p in (m.get('parts') or []) if isinstance(p, dict)]
            if d.get('type') == 'assistant':
                cur = cur or {'role': 'assistant', 'content': '', 'reasoning_content': '', 'tool_calls': []}
                for p in parts:
                    if 'functionCall' in p:
                        fc = p['functionCall']
                        cur['tool_calls'].append({'function': {'name': fc.get('name', ''), 'arguments': fc.get('args') or {}}})
                    elif isinstance(p.get('text'), str):
                        cur['reasoning_content' if p.get('thought') else 'content'] += p['text']
            elif d.get('type') == 'tool_result':
                flush()
                for p in parts:
                    fr = p.get('functionResponse')
                    if fr:
                        r = fr.get('response') or {}
                        body = r.get('output', r.get('error', ''))
                        msgs.append({'role': 'tool', 'content': body if isinstance(body, str) else json.dumps(body)})
            elif d.get('type') == 'user':
                flush()
                text = '\n'.join(p['text'] for p in parts if isinstance(p.get('text'), str))
                if text:
                    msgs.append({'role': 'user', 'content': text})
        flush()
        if any(mm['role'] == 'user' for mm in msgs):
            yield dict(group=cwd or '?', key=sid, msgs=msgs, kind='chat')


def _gk(cfg, row, i):
    if cfg.get('group_re'):
        m = re.search(cfg['group_re'], row.get('_path', ''))
        g = m.group(1) if m else cfg['name']
    else:
        g = str(row.get(cfg['group'])) if cfg.get('group') else cfg['name']
    k = str(row.get(cfg['key'])) if cfg.get('key') else f'{cfg["name"]}-{i}'
    return g, k


def src_oai(cfg, rng, render):
    # rows holding an OpenAI/ShareGPT-style message list (cfg: files, messages column, group/key columns)
    col = cfg.get('messages', 'messages')
    for i, row in enumerate(rows(cfg['files'], cfg.get('limit'), rng)):
        msgs = row.get(col)
        if isinstance(msgs, str):
            try:
                msgs = json.loads(msgs)
            except ValueError:
                continue
        if not isinstance(msgs, list):
            continue
        if cfg.get('final'):   # e.g. "fncall_response.choices.0.message": the last step kept outside the list
            v = row
            for part in cfg['final'].split('.'):
                v = v[int(part)] if isinstance(v, list) and part.isdigit() and int(part) < len(v) else (v.get(part) if isinstance(v, dict) else None)
            if isinstance(v, dict):
                msgs = msgs + [v]
        if cfg.get('system') and row.get(cfg['system']):
            msgs = [{'role': 'system', 'content': row[cfg['system']]}] + list(msgs)
        msgs = oai_messages(msgs, keep_system=cfg.get('keep_system', bool(cfg.get('system'))))
        if not any(m['role'] == 'user' for m in msgs) or not any(m['role'] == 'assistant' for m in msgs):
            continue
        g, k = _gk(cfg, row, i)
        tools = row.get(cfg['tools']) if cfg.get('tools') else None
        yield dict(group=g, key=k, msgs=msgs, kind='chat', tools=tools or None, gen=int(bool(row.get('_gen'))))


def src_react(cfg, rng, render):
    # ReAct turns (user / thought / action / observation / final): thought + action -> one assistant turn with the
    # action as a shell tool call, observation -> tool result, final -> assistant reply
    for i, row in enumerate(rows(cfg['files'], cfg.get('limit'), rng)):
        msgs, cur = [], None
        for t in row.get(cfg.get('turns', 'turns')) or []:
            role, text = t.get('role'), t.get('text') or ''
            if role == 'user':
                cur = None
                msgs.append({'role': 'user', 'content': text})
            elif role == 'thought':
                cur = {'role': 'assistant', 'content': '', 'reasoning_content': text}
                msgs.append(cur)
            elif role == 'action':
                if cur is None or cur.get('tool_calls'):
                    cur = {'role': 'assistant', 'content': '', 'reasoning_content': ''}
                    msgs.append(cur)
                cur['tool_calls'] = [{'function': {'name': 'run_shell_command', 'arguments': {'command': text}}}]
            elif role == 'observation':
                cur = None
                msgs.append({'role': 'tool', 'content': text})
            elif role == 'final':
                cur = None
                msgs.append({'role': 'assistant', 'content': text, 'reasoning_content': ''})
        if any(m['role'] == 'user' for m in msgs):
            g, k = _gk(cfg, row, i)
            yield dict(group=g, key=k, msgs=msgs, kind='chat')


def src_text(cfg, rng, render):
    # plain text columns joined (cfg: files, columns, optional where {column: [allowed values]})
    for i, row in enumerate(rows(cfg['files'], cfg.get('limit'), rng)):
        if cfg.get('where') and any(row.get(k) not in v for k, v in cfg['where'].items()):
            continue
        if cfg.get('where_not') and any(row.get(k) in v for k, v in cfg['where_not'].items()):
            continue
        parts = []
        for c in cfg['columns']:
            v = row.get(c)
            if c == 'patches' and isinstance(v, list):   # CIRCL: base64 patch texts
                import base64
                v = '\n'.join(base64.b64decode(x.get('patch_text_b64') or '').decode('utf8', 'replace') for x in v)
            elif isinstance(v, (list, dict)):
                v = json.dumps(v, ensure_ascii=False)
            parts.append(str(v or ''))
        text = '\n\n'.join(parts).strip()
        if len(text) < cfg.get('min_chars', 200):
            continue
        g, k = _gk(cfg, row, i)
        yield dict(group=g, key=k, text=text, kind='text')


def src_pairs(cfg, rng, render):
    # prompt / reply columns -> user / assistant turns, cfg['per_doc'] pairs per document; a list-valued prompt
    # column is taken as a whole message list
    per_doc = cfg.get('per_doc', 1)
    doc, n, g = [], 0, cfg['name']
    for i, row in enumerate(rows(cfg['files'], cfg.get('limit'), rng)):
        if cfg.get('where') and any(row.get(k) not in v for k, v in cfg['where'].items()):
            continue
        if cfg.get('where_not') and any(row.get(k) in v for k, v in cfg['where_not'].items()):
            continue
        p = row.get(cfg['prompt'])
        if isinstance(p, list):
            doc += oai_messages(p)
        else:
            rr, cc = split_think(row.get(cfg['reply']) or '')
            if cfg.get('unescape'):   # text stored with literal \\n escapes
                p, cc = (x.replace('\\n', '\n').replace('\\t', '\t') if isinstance(x, str) else x for x in (p, cc))
            if cfg.get('reasoning'):
                rr = row.get(cfg['reasoning']) or rr
            doc += [{'role': 'user', 'content': p or ''}, {'role': 'assistant', 'content': cc, 'reasoning_content': rr}]
        n += 1
        if n >= per_doc:
            g, _ = _gk(cfg, row, i)
            yield dict(group=g, key=f'{cfg["name"]}-{i}', msgs=doc, kind='chat', gen=int(bool(row.get('_gen'))))
            doc, n = [], 0


def src_ctftime(cfg, rng, render):
    # text chunks without writeup ids: a chunk that continues the previous one (shared 100-char overlap) keeps its key
    prev, key = '', None
    for i, row in enumerate(rows(cfg['files'])):
        t = row.get('text_chunk') or ''
        if not key or not prev or prev[-100:] not in t[:2000]:
            key = f'ctf-{i}'
        prev = t
        if len(t) > 300:
            yield dict(group='ctftime', key=key, text=t, kind='text')


def src_files(cfg, rng, render):
    paths = sorted(glob.glob(os.path.expanduser(cfg['files']), recursive=True))
    rng.shuffle(paths)
    for p in paths:
        if not os.path.isfile(p) or os.path.getsize(p) < 200:
            continue
        with open(p, encoding='utf8', errors='replace') as f:
            text = f.read(cfg.get('max_file_chars', 2_000_000))
        g = cfg['name']
        if cfg.get('group_re'):
            m = re.search(cfg['group_re'], p)
            g = m.group(1) if m else g
        yield dict(group=g, key=p, text=text, kind='text')


SOURCES = {'sessions': src_sessions, 'oai': src_oai, 'react': src_react, 'text': src_text, 'pairs': src_pairs,
           'ctftime': src_ctftime, 'files': src_files, 'code': src_code}

# ---------------------------------------------------------------- rendering, splitting

EFFORTS = (('low', 0.45), ('medium', 0.45), ('xhigh', 0.10))   # the user's Qwen Code defaults: fast = low


def pick_effort(rng):
    x, acc = rng.random(), 0.0
    for e, w in EFFORTS:
        acc += w
        if x < acc:
            return e
    return 'medium'


def render_doc(d, render, rng, cfg):
    effort = 'medium' if d.get('gen') else pick_effort(rng)   # generated replies were made at medium
    if d['kind'] == 'chat':
        return render(d['msgs'], d.get('tools'), effort)
    if d['kind'] == 'file' and rng.random() < cfg.get('as_tool_frac', 0.3):
        # the way an agent sees a file: a read_file call and its result
        msgs = [{'role': 'user', 'content': f'Take a look at {d["path"]}.'},
                {'role': 'assistant', 'content': '', 'reasoning_content': '',
                 'tool_calls': [{'function': {'name': 'read_file', 'arguments': {'absolute_path': '/workspace/' + d['path']}}}]},
                {'role': 'tool', 'content': d['text']}]
        return render(msgs, None, effort)
    return d['text']


def split_text(text, max_chars, boundary):
    # pieces of at most max_chars, cut at the last boundary in the second half of the window; nothing is dropped
    if len(text) <= max_chars:
        return [text]
    out, pos = [], 0
    while len(text) - pos > max_chars:
        cut = -1
        for m in boundary.finditer(text, pos + max_chars // 2, pos + max_chars):
            cut = m.start()
        if cut <= pos:
            cut = pos + max_chars
        out.append(text[pos:cut])
        pos = cut
    out.append(text[pos:])
    return out


CHAT_BOUNDARY = re.compile(r'<\|im_start\|>')
TEXT_BOUNDARY = re.compile(r'\n\s*\n')

# ---------------------------------------------------------------- filters

REFUSAL_RE = re.compile(r"I'm sorry, but I can't|I cannot (help|assist) with|I can't (help|assist) with|"
                        r"I am unable to (help|assist)|As an AI(,| language model)", re.I)


def assistant_text(d):
    if d['kind'] != 'chat':
        return ''
    return '\n'.join(m.get('content') or '' for m in d['msgs'] if m['role'] == 'assistant')


class Excluder:
    # whitespace-normalized 64-char shingles of the eval texts; low-entropy and licence shingles left out
    def __init__(self, paths, k=64):
        self.k, self.h = k, set()
        for p in paths:
            t = self.norm(open(os.path.expanduser(p), encoding='utf8', errors='replace').read())
            for i in range(0, max(0, len(t) - k), 4):
                s = t[i:i + k]
                if len(set(s)) >= 16 and not LICENSE_RE.search(s):
                    self.h.add(hash(s))

    @staticmethod
    def norm(t):
        return re.sub(r'\s+', ' ', t)

    def frac(self, text):
        if not self.h:
            return 0.0
        t = self.norm(text)
        pos = range(0, max(0, len(t) - self.k), 4)
        return sum(hash(t[i:i + self.k]) in self.h for i in pos) / max(1, len(pos))


# ---------------------------------------------------------------- tokens

def measure_cpt(tokenizer, model, samples):
    # chars per token of the joined samples with the model's tokenizer (special tokens parsed)
    text = '\n'.join(samples)
    if not text:
        return 3.5
    r = subprocess.run([tokenizer, '-m', model, '--stdin', '--ids', '--log-disable'], input=text.encode(),
                       capture_output=True, timeout=900)
    ids = r.stdout.decode().strip()
    n = ids.count(',') + 1 if ids.startswith('[') and len(ids) > 2 else 0
    return len(text) / n if n else 3.5


# ---------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--tokenizer', required=True, help='path to llama-tokenize')
    p.add_argument('--config', required=True)
    p.add_argument('--tokens', type=int, default=10_000_000)
    p.add_argument('--max-doc-tokens', type=int, default=32768)
    p.add_argument('--exclude', action='append', default=[])
    p.add_argument('--exclude-frac', type=float, default=0.02)
    p.add_argument('--only', help='comma-separated source names (for testing a config)')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--out', required=True)
    a = p.parse_args()

    conf = json.load(open(a.config))
    srcs = [s for s in conf['sources'] if not a.only or s['name'] in a.only.split(',')]
    rng = random.Random(a.seed)
    render = load_template(a.model)
    excl = Excluder(a.exclude)
    shares = {s['name']: s['share'] for s in srcs}
    total_share = sum(shares.values())
    stats, per_src = {}, {}

    for s in srcs:
        name = s['name']
        budget = a.tokens * s['share'] / total_share
        s['_budget_tokens'] = budget
        srng = random.Random(f'{a.seed}-{name}')
        cands = defaultdict(list)
        drops = Counter()
        n_cand = 0
        boundary = CHAT_BOUNDARY if s['type'] in ('sessions', 'oai', 'react', 'pairs') else TEXT_BOUNDARY
        max_chars = int(s.get('max_doc_tokens', a.max_doc_tokens) * s.get('cpt_guess', 3.3))
        for d in SOURCES[s['type']](s, srng, render):
            if d['kind'] == 'chat' and REFUSAL_RE.search(assistant_text(d)):
                drops['refusal'] += 1
                continue
            try:
                text = render_doc(d, render, srng, s)
            except Exception:
                drops['render_failed'] += 1
                continue
            if excl.frac(text) > a.exclude_frac:
                drops['eval_overlap'] += 1
                continue
            pieces = split_text(text, max_chars, boundary)
            mpk = s.get('max_pieces_per_key', 1 if s['type'] == 'files' else 0)
            if mpk and len(pieces) > mpk:   # long single documents: a few random pieces, for variety
                pieces = srng.sample(pieces, mpk)
            for piece in pieces:
                cands[d['group']].append((d['key'], piece, d.get('gen', 0)))
            n_cand += len(text)
            n_docs = sum(len(v) for v in cands.values())
            if n_cand > budget * 3.5 * s.get('oversample', 4) and n_docs >= s.get('min_candidates', 400):
                break   # enough to choose from
        sample = [t[:20000] for v in cands.values() for _, t, _ in v[:3]][:60]
        cpt = measure_cpt(a.tokenizer, a.model, sample)
        budget_chars = budget * cpt


        # round-robin over groups; a group at its cap drops out and the rest of the budget goes to the others
        for v in cands.values():
            srng.shuffle(v)
        groups = list(cands)
        used, per_group, picked = 0, Counter(), []
        cap_chars = s.get('group_cap', 1.0) * budget_chars
        while used < budget_chars and groups:
            srng.shuffle(groups)
            for g in list(groups):
                if not cands[g]:
                    groups.remove(g)
                    continue
                key, text, gen = cands[g][-1]
                if per_group[g] > 0 and per_group[g] + len(text) > cap_chars:
                    groups.remove(g)
                    continue
                cands[g].pop()
                picked.append((g, key, text, gen))
                used += len(text)
                per_group[g] += len(text)
                if used >= budget_chars:
                    break

        # halves per source by split key, then pieces at boundaries
        half_of, half_chars, docs = {}, [0, 0], []
        for g, key, text, gen in picked:
            if key not in half_of:
                half_of[key] = 0 if half_chars[0] <= half_chars[1] else 1
            h = half_of[key]
            half_chars[h] += len(text)
            text = text.encode('utf-8', 'replace').decode('utf-8')   # lone surrogates from some datasets
            docs.append({'text': text, 'set': 2 * h + gen, 'src': name, 'key': str(key)})
        per_src[name] = docs
        top = max(per_group.values()) if per_group else 0
        set_chars = Counter()
        for d in docs:
            set_chars[d['set']] += len(d['text'])
        stats[name] = {'share': s['share'] / total_share, 'chars_per_token': round(cpt, 2),
                       'est_tokens': int(used / cpt), 'budget_tokens': int(budget), 'budget_met': used >= budget_chars,
                       'docs': len(docs), 'groups': len(per_group), 'top_group_share': round(top / used, 3) if used else 0,
                       'tokens_per_set': {str(k): int(v / cpt) for k, v in sorted(set_chars.items())},
                       'dropped': dict(drops)}
        print(f'{name:22s} ~{used / cpt / 1e3:7.0f}k tok of {budget / 1e3:7.0f}k  cpt {cpt:4.2f}  docs {len(docs):6d}  '
              f'groups {len(per_group):5d}  top {top / max(1, used) * 100:3.0f}%  drops {dict(drops)}', flush=True)

    # deficit round-robin: the next document comes from the source furthest below its target share
    for v in per_src.values():
        rng.shuffle(v)
    cpt_of = {n: stats[n]['chars_per_token'] for n in per_src}
    got_share = {n: stats[n]['est_tokens'] for n in per_src}
    got_total = sum(got_share.values()) or 1
    written, order = Counter(), []
    active = [n for n in per_src if per_src[n]]
    while active:
        tot = sum(written.values()) or 1
        n = min(active, key=lambda n: written[n] / tot - got_share[n] / got_total)
        d = per_src[n].pop()
        written[n] += len(d['text']) / cpt_of[n]
        order.append(d)
        if not per_src[n]:
            active.remove(n)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, 'pool.jsonl'), 'w', encoding='utf8') as f:
        for d in order:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')

    # prefix balance: largest gap between a source's share in any 10% prefix and its share of the whole pool
    tot_tokens = sum(len(d['text']) / cpt_of[d['src']] for d in order)
    worst, acc, run, mark = 0.0, Counter(), 0.0, 0.1
    for d in order:
        t = len(d['text']) / cpt_of[d['src']]
        acc[d['src']] += t
        run += t
        if run >= mark * tot_tokens:
            worst = max([worst] + [abs(acc[n] / run - got_share[n] / got_total) for n in per_src])
            mark += 0.1
    stats['_pool'] = {'docs': len(order), 'est_tokens': int(tot_tokens), 'worst_prefix_share_error': round(worst, 4)}
    json.dump(stats, open(os.path.join(a.out, 'stats.json'), 'w'), indent=1)
    print(json.dumps(stats['_pool']))


if __name__ == '__main__':
    main()
