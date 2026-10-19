"""Read-only independent audit of recorded binding artifacts; no inference.

Does not import the training/data implementation. --save writes one new audit
record only; never overwrites an existing audit or historical experiment.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def source_answers(source):
    rows = [line.split(' = ') for line in source.splitlines()]
    assert len(rows) == 4
    assert len({identity for identity, _ in rows}) == 4
    assert {color for _, color in rows} == {'red', 'green', 'blue', 'yellow'}
    return {color: identity for identity, color in rows}


def prediction(record, candidates):
    parts = record['token_log_probabilities']
    scores = record['answer_log_probabilities']
    assert len(parts) == len(scores) == len(candidates) == 128
    assert all(len(pair) == 2 for pair in parts)
    assert all(math.isfinite(x) for pair in parts for x in pair)
    assert all(math.isfinite(x) for x in scores)
    # JSON exposes float32 values; Python's float64 sum may differ by rounding.
    assert max(abs(sum(pair) - score) for pair, score in zip(parts, scores)) < 1e-4
    predicted = candidates[max(range(len(scores)), key=scores.__getitem__)]
    assert predicted == record['prediction']
    return predicted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', type=Path)
    args = parser.parse_args()
    acquisition = read(ROOT / 'acquisition.json')
    for name, digest in acquisition['components'].items():
        assert sha(ROOT / name) == digest == sha(ROOT / 'initial_code' / name)
    pools = read(ROOT / 'identity_pools.json')
    known, novel = set(pools['known']), set(pools['novel'])
    assert len(known) == len(novel) == 64 and not known & novel
    assert known | novel == {a + str(d) for a in 'ABCDEFGHIJKLMNOP' for d in range(8)}
    for pool in (known, novel):
        assert set(Counter(x[0] for x in pool).values()) == {4}
        assert set(Counter(x[1] for x in pool).values()) == {8}
    data_path = ROOT / 'data/tiny/examples.json'
    data = read(data_path)
    manifest = read(ROOT / 'data/tiny/manifest.json')
    assert sha(data_path) == manifest['examples_sha256']
    assert sha(ROOT / 'data/tiny/sender.pt') == manifest['capture_sha256']
    originals, counterfactuals = {}, {}
    for table in data['records']:
        g = table['group']
        originals[g] = source_answers(table['source'])
        counterfactuals[g] = source_answers(table['wrong_source'])
        assert originals[g] == table['bindings']
        assert counterfactuals[g] == table['wrong_bindings']
        assert set(originals[g].values()) == set(counterfactuals[g].values())
        assert all(originals[g][c] != counterfactuals[g][c] for c in originals[g])
        assert set(originals[g].values()) <= known
        assert [line.split(' = ')[0] for line in table['source'].splitlines()] == [
            line.split(' = ')[0] for line in table['wrong_source'].splitlines()]
    rows = {row['id']: row for row in data['rows']}
    assert len(rows) == 32 and len(originals) == 8
    assert len({v for table in originals.values() for v in table.values()}) == 32
    assert set(Counter(row['group'] for row in rows.values()).values()) == {4}
    for row in rows.values():
        assert row['query'] == 'Which ID is ' + row['color'] + '?'
        assert row['answer'] == originals[row['group']][row['color']]
        assert row['wrong_answer'] == counterfactuals[row['group']][row['color']]
    result = dict(status='PASS', scope='Recorded artifacts only, not a research verdict',
                  utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                  code_snapshot_exact=True, pool_components_balanced=True,
                  source_answers_rederived=True, tiny_tables=8, tiny_queries=32,
                  data_directories=sorted(p.name for p in (ROOT / 'data').iterdir()),
                  text=None, evaluations={})
    sender_path = ROOT / 'text_tiny/sender.json'
    receiver_path = ROOT / 'text_tiny/receiver.json'
    if sender_path.exists():
        sender = read(sender_path)
        candidates = sender['candidates']
        sent = {r['id']: prediction(r, candidates) for r in sender['outputs']}
        assert set(sent) == set(rows)
        result['text'] = dict(sender_accuracy=sum(sent[i] == rows[i]['answer'] for i in rows) / 32)
        if receiver_path.exists():
            receiver = read(receiver_path)
            actual, oracle = {}, {}
            for rec in receiver['records']:
                i = rec['id']
                assert rec['answer'] == rows[i]['answer'] and rec['sender_prediction'] == sent[i]
                actual[i] = prediction(rec['actual_text'], candidates)
                oracle[i] = prediction(rec['oracle_text'], candidates)
            assert set(actual) == set(oracle) == set(rows)
            result['text'].update(
                actual_text_accuracy=sum(actual[i] == rows[i]['answer'] for i in rows) / 32,
                oracle_text_accuracy=sum(oracle[i] == rows[i]['answer'] for i in rows) / 32,
                oracle_per_color={c: sum(oracle[i] == rows[i]['answer'] for i in rows if rows[i]['color'] == c) / 8
                                  for c in ['red', 'green', 'blue', 'yellow']})
            summary = read(ROOT / 'text_tiny/summary.json')
            for key, value in result['text'].items():
                assert summary[key] == value
    selection = []
    for path in sorted((ROOT / 'run_tiny').glob('*/per_example.json')):
        saved = read(path)
        metrics = read(path.parent / 'metrics.json')
        assert sha(path.parent / 'outputs.pt') == metrics['output_sha256']
        records = {r['id']: r for r in saved}
        assert set(records) == set(rows)
        predicted = {cond: {i: prediction(r['conditions'][cond], metrics['candidates'])
                            for i, r in records.items()} for cond in metrics['conditions']}
        accuracies = {cond: sum(pred[i] == rows[i]['answer'] for i in rows) / 32
                      for cond, pred in predicted.items()}
        for cond, accuracy in accuracies.items():
            assert accuracy == metrics['conditions'][cond]['accuracy']
            mean_nll = -sum(records[i]['conditions'][cond]['answer_log_probabilities'][
                metrics['candidates'].index(rows[i]['answer'])] for i in rows) / 32
            assert abs(mean_nll - metrics['conditions'][cond]['mean_id_nll']) < 1e-5
            if cond == 'correct' and path.parent.name.startswith('eval_') and path.parent.name != 'eval_0000':
                selection.append((mean_nll, int(path.parent.name.split('_')[1])))
            for color in ['red', 'green', 'blue', 'yellow']:
                observed = sum(predicted[cond][i] == rows[i]['answer'] for i in rows if rows[i]['color'] == color) / 8
                assert observed == metrics['conditions'][cond]['per_color'][color]
        donor_map = {}
        for i, rec in records.items():
            donor = records[rec['donor_id']]
            assert donor['group'] != rec['group'] and donor['query'] == rows[i]['query']
            assert donor_map.setdefault(rec['group'], donor['group']) == donor['group']
            assert rec['donor_answer'] == rows[rec['donor_id']]['answer']
            assert rec['conditions']['shuffled']['token_log_probabilities'] == donor['conditions']['correct']['token_log_probabilities']
            assert rec['conditions']['zero']['token_log_probabilities'] == rec['conditions']['receiver_only']['token_log_probabilities']
        assert set(donor_map) == set(donor_map.values()) == set(originals)
        donor_rates = dict(
            shuffled=sum(predicted['shuffled'][i] == records[i]['donor_answer'] for i in rows) / 32,
            wrong_binding=sum(predicted['wrong_binding'][i] == rows[i]['wrong_answer'] for i in rows) / 32)
        for cond, rate in donor_rates.items():
            assert rate == metrics['conditions'][cond]['donor_following']
        result['evaluations'][path.parent.name] = dict(accuracies=accuracies, donor_following=donor_rates,
                                                       zero_exact=True, shuffled_donor_scores_exact=True)
    if (ROOT / 'run_tiny/decision.json').exists():
        decision = read(ROOT / 'run_tiny/decision.json')
        terminal_name = f"eval_{decision['step']:04d}"
        terminal = read(ROOT / 'run_tiny' / terminal_name / 'per_example.json')
        repeated = read(ROOT / 'run_tiny/terminal_repeat/per_example.json')
        assert terminal == repeated
        assert sha(ROOT / 'run_tiny/best.pt') == decision['best_sha256']
        assert sha(ROOT / 'run_tiny/last.pt') == decision['last_sha256']
        best_nll, best_step = min(selection)
        assert best_step == decision['best_step']
        assert abs(best_nll - decision['best_id_nll']) < 1e-5
        observed = result['evaluations'][terminal_name]
        accuracy = observed['accuracies']
        passed = (accuracy['correct'] >= .95 and observed['donor_following']['shuffled'] >= .95
                  and all(accuracy[c] <= .25 for c in ['shuffled', 'random_71001', 'random_71002'])
                  and accuracy['receiver_only'] <= .125 and observed['zero_exact'])
        assert passed == json.loads((ROOT / 'run_tiny/gates.json').read_text())['pass']
        result['terminal_repeat_exact'] = True
        result['tiny_pass_independently_rederived'] = passed
        result['best_checkpoint_step_from_raw_scores'] = best_step
        result['recorded_verdict'] = decision['verdict']
    if args.save:
        with args.save.open('x') as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write('\n')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
