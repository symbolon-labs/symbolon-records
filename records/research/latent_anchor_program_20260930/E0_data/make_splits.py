"""E0 split builder (scratch mode), version 2 (2026-10-01, after the harness review).

Balanced rounds reuse latent_binding_v1_20260906/binding.py::make_split with the round size
generalized from a hardcoded 64 to len(pool). Mixed splits pin one unseen ID per table.

Why version 2. The seed-120501 checkerboard pools are block-structured: every known-pool ID
(all 56 trained IDs and the 8 burned IDs) pairs a letter with a digit of the same parity group,
and every novel-pool ID breaks that rule. The v1 DEV-novel used only burned IDs, so it could
not detect a model that learned "this letter only takes these four digits", which would fail
FINAL. Version 2 therefore carves 8 out-of-block (OOB) IDs from the novel pool, by a fixed
rule chosen before anything was scored, as the development proxy for FINAL. The remaining 56
novel IDs stay pristine and form the E3 FINAL pool. Pure splits use only unseen IDs, so a
table offers no trained companions to eliminate.
"""
import hashlib
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = HERE.parents[1]
POOLS = json.loads((R / 'latent_binding_v1_20260906/identity_pools.json').read_text())
COLORS = ['red', 'green', 'blue', 'yellow']
LETTERS, DIGITS = 'ABCDEFGHIJKLMNOP', '01234567'
IDS = [a + d for a in LETTERS for d in DIGITS]
BURNED = ['A0', 'A2', 'C4', 'I2', 'K0', 'K3', 'M0', 'M3']
KNOWN64 = sorted(POOLS['known'])
NOVEL64 = sorted(POOLS['novel'])
TRAINED56 = sorted(set(KNOWN64) - set(BURNED))
assert len(TRAINED56) == 56 and set(BURNED) <= set(KNOWN64) and not set(NOVEL64) & set(KNOWN64)


def block_groups():
    """Letter and digit parity groups such that an ID is in the known pool iff the groups match."""
    a0 = KNOWN64[0][0]
    digits_of = {a: {i[1] for i in KNOWN64 if i[0] == a} for a in LETTERS}
    ga = {a: int(digits_of[a] == digits_of[a0]) for a in LETTERS}
    gd = {d: int(d in digits_of[a0]) for d in DIGITS}
    for i in IDS:
        assert (ga[i[0]] == gd[i[1]]) == (i in KNOWN64), 'pools are not block-structured'
    return ga, gd


def choose_oob8(seed=123010):
    """8 novel-pool IDs with 8 distinct letters (4 per letter group) and all 8 digits once."""
    ga, gd = block_groups()
    rng = random.Random(seed)
    for _ in range(100000):
        letters = rng.sample([a for a in LETTERS if ga[a] == 0], 4) + rng.sample([a for a in LETTERS if ga[a] == 1], 4)
        digits = rng.sample(list(DIGITS), 8)
        ids = [a + d for a, d in zip(letters, digits)]
        if all(i in NOVEL64 for i in ids):
            return sorted(ids)
    raise AssertionError('no OOB8 found')


OOB8 = choose_oob8()
FINAL_NOVEL56 = sorted(set(NOVEL64) - set(OOB8))
assert len(OOB8) == 8 and len(FINAL_NOVEL56) == 56

# balanced: (seed, rounds, pool); mixed: (seed, unseen pool, companion pool)
BALANCED = {
    'train': (122001, 5, TRAINED56),
    'dev_known': (123001, 1, TRAINED56),
    'dev_inblock_pure': (123004, 2, BURNED),
    'dev_oob_pure': (123005, 2, OOB8),
}
MIXED = {
    'dev_inblock_mixed': (123006, BURNED, TRAINED56),
    'dev_oob_mixed': (123007, OOB8, TRAINED56),
}
ORDER = ['train', 'dev_known', 'dev_inblock_pure', 'dev_oob_pure', 'dev_inblock_mixed', 'dev_oob_mixed']
UNSEEN = {'train': [], 'dev_known': [], 'dev_inblock_pure': BURNED, 'dev_oob_pure': OOB8,
          'dev_inblock_mixed': BURNED, 'dev_oob_mixed': OOB8}


def parse_source(source):
    by_color = {}
    for line in source.splitlines():
        identity, color = line.split(' = ')
        assert identity in IDS and color in COLORS and color not in by_color
        by_color[color] = identity
    assert len(set(by_color.values())) == len(by_color) == 4
    return by_color


def canonical(bindings):
    return tuple(bindings[color] for color in COLORS)


def balanced_tuples(rng, pool, rounds):
    n = len(pool)
    tuples = []
    for _ in range(rounds):
        permutations = []
        for color in COLORS:
            for attempt in range(10000):
                perm = rng.sample(pool, n)
                if all(perm[i] not in [p[i] for p in permutations] for i in range(n)):
                    break
            else:
                raise AssertionError('Balanced matching failed')
            permutations.append(perm)
        tuples.extend([[p[i] for p in permutations] for i in range(n)])
    return tuples


def mixed_tuples(rng, unseen, companions):
    tuples = []
    for identity in unseen:
        for color in COLORS:
            others = rng.sample(companions, 3)
            values = []
            it = iter(others)
            for c in COLORS:
                values.append(identity if c == color else next(it))
            tuples.append(values)
    rng.shuffle(tuples)
    return tuples


def build(split, used):
    if split in BALANCED:
        seed, rounds, pool = BALANCED[split]
        rng = random.Random(seed)
        tuples = balanced_tuples(rng, pool, rounds)
        meta = dict(kind='balanced', rounds=rounds, pool=pool)
    else:
        seed, unseen, companions = MIXED[split]
        rng = random.Random(seed)
        tuples = mixed_tuples(rng, unseen, companions)
        meta = dict(kind='mixed', unseen_pool=unseen, companions='TRAINED56')
    unseen = set(UNSEEN[split])
    records = []
    for group, values in enumerate(tuples):
        bindings = dict(zip(COLORS, values))
        key = canonical(bindings)
        assert key not in used, 'Repeated table'
        order = rng.sample(COLORS, 4)
        offset = rng.randrange(1, 4)
        source = '\n'.join(f'{bindings[color]} = {color}' for color in order)
        wrong = '\n'.join(f'{bindings[color]} = {COLORS[(COLORS.index(color) + offset) % 4]}' for color in order)
        wrong_bindings = parse_source(wrong)
        assert canonical(wrong_bindings) not in used
        used.update([key, canonical(wrong_bindings)])
        records.append(dict(group=group, source=source, wrong_source=wrong, bindings=bindings,
                            wrong_bindings=wrong_bindings, unseen_in_table=sum(v in unseen for v in values)))
    rows = []
    for rec in records:
        for color in COLORS:
            rows.append(dict(group=rec['group'], color=color, query=f'Which ID is {color}?',
                             answer=rec['bindings'][color], wrong_answer=rec['wrong_bindings'][color],
                             novel_answer=rec['bindings'][color] in unseen,
                             novel_wrong_answer=rec['wrong_bindings'][color] in unseen))
    rng.shuffle(rows)
    for i, row in enumerate(rows):
        row['id'] = f'{split}-{i:06d}'
    return dict(split=split, seed=seed, unseen=sorted(unseen), records=records, rows=rows, **meta)


def check(data):
    split = data['split']
    unseen = set(UNSEEN[split])
    if data['kind'] == 'balanced':
        counts = {}
        for row in data['rows']:
            counts[(row['answer'], row['color'])] = counts.get((row['answer'], row['color']), 0) + 1
        assert set(counts.values()) == {data['rounds']} and len(counts) == len(data['pool']) * 4
    else:
        assert all(rec['unseen_in_table'] == 1 for rec in data['records'])
        pairs = {(r['answer'], r['color']) for r in data['rows'] if r['novel_answer']}
        assert len(pairs) == len(unseen) * 4 == sum(r['novel_answer'] for r in data['rows'])
    for rec in data['records']:
        assert all(rec['bindings'][c] != rec['wrong_bindings'][c] for c in COLORS)
        assert set(rec['bindings'].values()) == set(rec['wrong_bindings'].values())
        ids = set(rec['bindings'].values())
        assert not ids & set(FINAL_NOVEL56), 'FINAL pool ID in a development split'
        if split in ('train', 'dev_known'):
            assert ids <= set(TRAINED56)


def main():
    out = HERE / 'splits'
    out.mkdir(exist_ok=True)
    used = set()
    manifest = dict(oob8=OOB8, burned=BURNED, final_novel56=FINAL_NOVEL56, block_groups=block_groups(), splits={})
    for split in ORDER:
        data = build(split, used)
        check(data)
        path = out / f'{split}.json'
        path.write_text(json.dumps(data, indent=1) + '\n')
        manifest['splits'][split] = dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                         tables=len(data['records']), rows=len(data['rows']),
                                         novel_answer_rows=sum(r['novel_answer'] for r in data['rows']))
        print(split, manifest['splits'][split])
    train = json.loads((out / 'train.json').read_text())
    assert {r['answer'] for r in train['rows']} == set(TRAINED56)
    (out / 'MANIFEST.json').write_text(json.dumps(manifest, indent=1) + '\n')
    print('OOB8', OOB8)


if __name__ == '__main__':
    main()
