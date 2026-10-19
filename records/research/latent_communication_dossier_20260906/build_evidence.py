"""Export documentation tables from sealed evidence; no inference or training."""
import csv
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent
RESEARCH = ROOT.parent
RUN = RESEARCH / 'latent_channel_smollm_20260906'
SEALED = ['latent_channel_v1_20260906', 'latent_channel_v2_20260906',
          'latent_channel_crossfamily_20260906', 'receiver_qualification_20260906',
          RUN.name]
HISTORY = ['latent_bridge_recovery_20260905/REPORT.md',
           'latent_bridge_recovery_20260905/recovery_verification.json',
           'latent_bridge_recovery_20260905/history_v2/index.json',
           'latent_bridge_redo_cached_20260906/REPORT.md',
           'latent_bridge_redo_cached_20260906/PREREGISTRATION.md',
           'latent_bridge_redo_audit_20260906/verification.json']


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def write_csv(path, rows):
    assert rows
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    dest = ROOT / 'evidence'
    assert not dest.exists(), 'Documentation exports are append-only; use a new dossier version'
    inventory, seals = {}, {}
    for name in SEALED:
        base = RESEARCH / name
        seal = read(base / 'seal.json')
        seals[name] = {'path': f'{name}/seal.json', 'sha256': sha(base/'seal.json'),
                       'files': len(seal['files'])}
        for relative, digest in seal['files'].items():
            path = base / relative
            assert sha(path) == digest, path
            inventory[f'{name}/{relative}'] = {'sha256': digest, 'bytes': path.stat().st_size,
                                              'kind': 'sealed_original'}
        inventory[f'{name}/seal.json'] = {'sha256': sha(base/'seal.json'),
            'bytes': (base/'seal.json').stat().st_size, 'kind': 'original_seal'}
    for relative in HISTORY:
        path = RESEARCH / relative
        assert path.is_file(), path
        inventory[relative] = {'sha256': sha(path), 'bytes': path.stat().st_size,
                              'kind': 'historical_report_or_recovery_index_not_full_original_payload_audit'}
    a = read(RUN/'run_final/analysis.json')
    assert a['gate4'] == 'PASS'
    assert all(a['statistical_checks'].values()) and all(a['mechanical_checks'].values())
    core = read(RUN/'run_final/evaluation/per_example.json')
    texts = {x['id']: x for x in read(RUN/'text_final/per_example.json')}
    by_id = {x['id']: x for x in core}
    assert len(by_id) == len(core) == len(texts) == 1024
    summary, classes, contrasts, long_rows, donors = [], [], [], [], []
    for name, values in a['accuracy'].items():
        summary.append({'condition': name, 'n': len(core), 'accuracy': values['mean'],
            'ci95_low': values['ci95'][0], 'ci95_high': values['ci95'][1],
            'margin_mean_nats': values['margin_mean'], 'margin_median_nats': values['margin_median'],
            'full_vocab_ce_nats': values['full_vocab_ce']})
        for y in range(4):
            classes.append({'condition': name, 'symbol': 'ABCD'[y], 'n': 256,
                'recall': values['per_class'][y], 'ci95_low': values['per_class_ci95'][y][0],
                'ci95_high': values['per_class_ci95'][y][1]})
        records = [x['conditions'][name] if name in x['conditions'] else texts[x['id']][name] for x in core]
        assert sum(v['prediction'] == x['label'] for v,x in zip(records,core))/len(core) == values['mean']
        for x,v in zip(core, records):
            lp = v['answer_log_probabilities']
            assert v['prediction'] == max(range(4), key=lp.__getitem__)
            long_rows.append({'id': x['id'], 'group': x['group'], 'symbol': 'ABCD'[x['label']],
                'condition': name, 'prediction': 'ABCD'[v['prediction']],
                'correct': int(v['prediction'] == x['label']),
                **{f'logp_{letter}': lp[i] for i,letter in enumerate('ABCD')}})
    for name,v in a['correct_minus'].items():
        contrasts.append({'comparator': name, 'accuracy_gain': v['accuracy_mean'],
            'ci99_low': v['accuracy_ci99'][0], 'ci99_high': v['accuracy_ci99'][1],
            'ci95_low': v['accuracy_ci95'][0], 'ci95_high': v['accuracy_ci95'][1],
            'margin_gain_mean_nats': v['margin_change_mean'],
            'margin_gain_median_nats': v['margin_change_median'],
            'margin_ci99_low': v['margin_change_ci99'][0], 'margin_ci99_high': v['margin_change_ci99'][1]})
    for x in core:
        donor = by_id[x['donor_id']]
        assert donor['label'] == x['donor_label']
        assert x['conditions']['shuffled']['answer_log_probabilities'] == donor['conditions']['correct']['answer_log_probabilities']
        donors.append({'recipient_id': x['id'], 'recipient_symbol': 'ABCD'[x['label']],
            'donor_id': donor['id'], 'donor_symbol': 'ABCD'[donor['label']],
            'swapped_prediction': 'ABCD'[x['conditions']['shuffled']['prediction']],
            'donor_probability_vector_exact': True})
    evaluations = []
    for mode in ['tiny','full']:
        for path in sorted((RUN/f'run_{mode}').glob('eval_*/metrics.json')):
            m = read(path)
            for name,v in m['conditions'].items():
                evaluations.append({'run': mode, 'step': int(path.parent.name[5:]),
                    'condition': name, 'accuracy': v['accuracy'], 'ce': v['full_vocab_ce'],
                    'per_class': json.dumps(v['per_class']), 'gate': m['gate'],
                    'source': str(path.relative_to(RESEARCH))})
    stages = []
    for phase in ['tiny','development','confirmation']:
        j = read(RUN/f'{phase}_job/state.json')
        assert j['status'] == 'complete'
        for s in j['stages']:
            stages.append({'phase': phase, 'stage': s['name'], 'utc_start': s['utc_start'],
                'elapsed_s': s['elapsed_s'], 'pid_at_execution': s['pid'], 'exit_code': s['exit_code'],
                'cumulative_child_peak_rss_kib': s['peak_child_rss_kib'],
                'command_argv': json.dumps(s['command'])})
    freeze = read(RUN/'run_full/FINAL_FREEZE.json')
    for v in freeze['components'].values():
        assert sha(Path(v['path'])) == v['sha256'], v['path']
    dest.mkdir()
    for name,rows in [('conditions',summary),('per_class',classes),('paired_contrasts',contrasts),
                      ('per_example',long_rows),('donor_swaps',donors),('learning_curve',evaluations),('execution_timeline',stages)]:
        write_csv(dest/f'{name}.csv', rows)
    write_json(dest/'result.json', a)
    write_json(dest/'model_and_runtime_identities.json', {
        'models': freeze['models'], 'components': freeze['components'], 'versions': freeze['versions'],
        'checkpoint_sha256': freeze['checkpoint_sha256'], 'final_freeze_sha256': sha(RUN/'run_full/FINAL_FREEZE.json'),
        'note': 'Identities copied from frozen records; model weight rehash requires verify.py --models.'})
    write_json(dest/'source_inventory.json', {'relative_to': str(RESEARCH), 'files': inventory, 'seals': seals,
        'historical_limit': 'Pre-channel report/index hashes are verified here; missing original arrays are not reconstructed.',
        'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
    write_json(dest/'export_verification.json', {'status': 'PASS', 'inference_run': False, 'training_run': False,
        'sealed_original_files_rehashed': sum(s['files'] for s in seals.values()),
        'source_inventory_sha256': sha(dest/'source_inventory.json'), 'generator_sha256': sha(Path(__file__)),
        'per_example_rows': len(long_rows), 'donor_pairs': len(donors),
        'csvs': {p.name:sha(p) for p in sorted(dest.glob('*.csv'))},
        'note': 'Documentation-time export, not another experiment or independent replication.'})
    print('PASS: originals rehashed; full result, class, contrast, donor, learning and timing tables exported')


if __name__ == '__main__':
    main()
