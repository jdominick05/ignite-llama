#!/usr/bin/env python3
# Copies a GGUF while dropping tensors and/or taking tensors from another GGUF (no requantization), e.g. to ship a
# requantized Flash-Next between machines without its 29 GB n-gram table and graft an identical local table back.
#
#   gguf_graft.py IN.gguf OUT.gguf [--drop REGEX] [--take DONOR.gguf --take-re REGEX]
#
# Taken tensors replace same-named ones (or are added when IN lacks them). Metadata comes from IN.
import argparse, os, re, sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'gguf-py'))
import gguf  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('inp')
    p.add_argument('out')
    p.add_argument('--drop')
    p.add_argument('--take')
    p.add_argument('--take-re')
    a = p.parse_args()
    if bool(a.take) != bool(a.take_re):
        sys.exit('--take and --take-re go together')

    r = gguf.GGUFReader(a.inp)
    arch = r.fields['general.architecture'].contents()
    donor = {}
    if a.take:
        donor = {t.name: t for t in gguf.GGUFReader(a.take).tensors if re.search(a.take_re, t.name)}
        if not donor:
            sys.exit(f'nothing in {a.take} matches {a.take_re}')
    plan = []
    for t in r.tensors:
        if t.name in donor:
            plan.append(donor.pop(t.name))
        elif not (a.drop and re.search(a.drop, t.name)):
            plan.append(t)
    plan += donor.values()   # taken tensors IN did not have

    w = gguf.GGUFWriter(a.out, arch=arch, endianess=r.endianess)
    for field in r.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith('GGUF.') or field.name.startswith('split.'):
            continue
        vt = field.types[0]
        sub = field.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
        w.add_key_value(field.name, field.contents(), vt, sub_type=sub)
    total = 0
    for t in plan:
        w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
        total += t.data.nbytes
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    for t in plan:
        w.write_tensor_data(t.data, tensor_endianess=r.endianess)
    w.close()
    print(f'wrote {a.out}: {len(plan)} tensors, {total / 1e9:.1f} GB')


if __name__ == '__main__':
    main()
