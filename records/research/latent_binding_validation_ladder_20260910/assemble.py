"""Read-only synthesis of sealed validation results; never executes models."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESEARCH = ROOT.parent
PINNED = {
    'latent_binding_order_ladder_20260910': '771d9c8c214fd64959aebb321c0b59c0977f8f69603eb99479f3290d036b07fb',
    'latent_binding_fresh_known_source_20260910': '3310438a4603d40e332dea78ff8e3162a25a0e0458481058e43f748da0f88ee4',
    'latent_binding_fresh_known_native_20260910': '33d50d74532f21e37ff9dc99d4149f3b8b379e527370ba0ca92e15dfadf8a9ee',
    'latent_binding_fresh_known_text_20260910': '76425e21a8890f0360ebcb47e95cbb1bb591024e6e64002b62dba5d5f9aba130',
    'latent_binding_current_exposure_corrected_20260910': '8f2f9f3214a0f5a9b5a616a4e34a4b22adc78b846e48ec678207af1511606faf',
    'latent_binding_novel_whole_source_20260910': '2830801114e0e90028345babd6704489411f4e1ad99869ed888adda1e612e40c',
    'latent_binding_novel_whole_native_20260910': 'a4025f8f941ff2551616c2a4e6464879a0eef1f7cafc47c415474b926c661e98',
}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def inventory(root):
    return {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob('*'))
            if p.is_file() and p != root / 'seal.json'}


def verify_parent(root, expected):
    assert sha(root / 'seal.json') == expected, root
    files = read(root / 'seal.json')['files']
    assert files == inventory(root), root
    return dict(path=str(root), seal_sha256=expected, files=len(files))


def build():
    pins = dict(PINNED)
    pins['latent_binding_novel_whole_text_20260910'] = read(ROOT / 'TEXT_PARENT.json')['seal_sha256']
    seals = [verify_parent(RESEARCH / name, h) for name, h in pins.items()]
    ladder_root = RESEARCH / 'latent_binding_order_ladder_20260910'
    ladder = read(ladder_root / 'SUMMARY.json')
    for parent in ladder['seals']:
        seals.append(verify_parent(Path(parent['path']), parent['seal_sha256']))
    architecture = read(ladder_root / 'ARCHITECTURE_LOCK.json')
    assert sha(Path(architecture['candidate_path'])) == architecture['candidate_sha256']
    models_checked = 0
    for model in architecture['models'].values():
        for name, info in model['files'].items():
            assert sha(Path(model['path']) / name) == info['sha256']
            models_checked += 1
    assert models_checked == 22
    stages = []
    for label, prefix in [('Fresh familiar IDs', 'fresh_known'), ('Novel whole IDs', 'novel_whole')]:
        native_root = RESEARCH / f'latent_binding_{prefix}_native_20260910'
        text_root = RESEARCH / f'latent_binding_{prefix}_text_20260910'
        native, text = read(native_root / 'analysis.json'), read(text_root / 'summary.json')
        assert read(native_root / 'audit.json')['status'] == text['status'] == 'PASS'
        assert native['checkpoint'] == architecture['candidate_sha256']
        counts = [native['original_target']['correct']['correct'],
                  native['shuffled_donor_target']['correct'],
                  native['changed_target']['wrong_binding']['correct']]
        jobs = []
        for job_root in [native_root / 'job', text_root / 'job_sender', text_root / 'job_receiver']:
            launch, completion = read(job_root / 'launch.json'), read(job_root / 'completion.json')
            assert completion['exit_code'] == 0 and not completion['timeout']
            assert not any(Path(f'/proc/{launch[k]}').exists() for k in ['pid', 'wrapper_pid'])
            jobs.append(dict(path=str(job_root), launch=launch, completion=completion))
        stages.append(dict(name=label, native=counts, native_pass=native['gate']['native_view_pass'],
            thresholds=native['gate']['thresholds'], data_seed=native['data_seed'],
            actual_text=[text['counts'][c]['actual_prediction'] for c in ['original', 'changed']],
            oracle_text=[text['counts'][c]['oracle_prediction'] for c in ['original', 'changed']],
            paired_latent_minus_text=text['paired_latent_minus_actual_text'],
            native_report=str(native_root / 'REPORT.md'), text_report=str(text_root / 'REPORT.md'), jobs=jobs))
    assert ladder['source_order_ladder_complete'] and stages[0]['native_pass']
    assert not stages[1]['native_pass'] and stages[1]['native'] == [0, 0, 0]
    novel_root = RESEARCH / 'latent_binding_novel_whole_native_20260910'
    native, data = read(novel_root / 'analysis.json'), read(novel_root / 'run/data.json')
    observations = {}
    known = {'B3', 'C7', 'D4', 'G3', 'L1', 'L7', 'N1', 'O4'}
    for arm, target, source in [('correct', 'original', 'source'), ('wrong_binding', 'changed', 'wrong_source')]:
        stats = native[target + '_target'][arm]
        expected = [next(i for i, line in enumerate(data['records'][row['group']][source].splitlines())
                    if line.split(' = ')[1] == row['color']) for row in data['rows']]
        routes = native['geometry'][arm]['routes']
        observations[target] = dict(prediction_ids=sorted(set(stats['predictions'])),
            familiar_prediction_count=sum(p in known for p in stats['predictions']),
            correct_route_counts=[sum(route[j] == slot for route, slot in zip(routes, expected)) for j in [0, 1]],
            first_prefix_correct=stats['first_prefix_correct'], gold_digit_correct=stats['gold_digit_correct'])
    return dict(verdict='SOURCE-ORDER AND FRESH FAMILIAR-ID PASS; NOVEL WHOLE-ID NO-GO',
        candidate=architecture['candidate_sha256'], architecture=architecture,
        order_views=ladder['views'], later_stages=stages, novel_observations=observations,
        seals=seals, model_tokenizer_files=models_checked, all_recorded_jobs_complete=True,
        novel_binding_established=False, arabic_opened=False, final_opened=False,
        new_external_spend_usd=0,
        novelty_scope='Eight whole IDs absent from 56 ancestral positive training IDs, with older evaluation/design-selection history; new contexts seed191003. Not pristine IDs or novel components.')


def report(result):
    lines = ['# Frozen d0f16cb9 validation ladder', '',
        '**Source-order robustness and fresh familiar-ID contexts passed; novel whole-ID binding failed.**', '',
        'The candidate stayed frozen throughout. The registered decision tree stops at the novel-ID gate; Arabic and the final pool remain unopened.', '',
        'Counts are out of 32. Native columns are original / shuffled donor / changed donor; text columns are original / changed.', '',
        '| Gate | Native | Actual text | Result |', '|---|---:|---:|---|']
    for row in result['order_views']:
        lines.append(f"| {row['name']} | {' / '.join(map(str, row['native']))} | {' / '.join(map(str, row['actual_text']))} | PASS |")
    for row in result['later_stages']:
        verdict = 'PASS' if row['native_pass'] else 'NO-GO'
        lines.append(f"| {row['name']} | {' / '.join(map(str, row['native']))} | {' / '.join(map(str, row['actual_text']))} | {verdict} |")
    lines += ['', 'Cyclic/CORE and later native minima were 31/31/28. Reserved reversal used prospectively frozen 28/28/28. All mandatory controls and paired bounds passed for the passing gates; no thresholds changed after outcomes.', '',
        'On novel IDs, requested-record routing was correct at both query sites in all 32 original and 32 changed cases. All 64 native predictions were familiar B3, C7, D4 or G3; full-ID accuracy was zero. This locates an observed failure after successful record routing, without establishing its internal cause. Mechanical checks passed, so no numerical autopsy or architecture repair was opened.', '',
        result['novelty_scope'], '',
        'The five source views reuse eight CAL tables and do not establish all 24 permutations. Fresh familiar contexts used seed191002 once. The novel bank used fixed seed191003 and structural-only construction before outcomes. These small banks are not independent large-sample transfer evidence.', '',
        'Actual text was freshly executed for each new order and later bank; CORE text retains explicitly audited historical reuse. Text gives the sender source plus query in chat format, whereas latent source encoding is source-only/plain; these access and formatting differences preclude an equal-access efficiency claim. All source-order and fresh-familiar latent-minus-text accuracy intervals include zero.', '',
        '| Later bank | Latent minus text original 95% CI | Changed 95% CI | Oracle original / changed |', '|---|---:|---:|---:|']
    for row in result['later_stages']:
        ci = row['paired_latent_minus_text']
        lines.append(f"| {row['name']} | {ci['original']['ci95']} | {ci['changed']['ci95']} | {' / '.join(map(str, row['oracle_text']))} |")
    lines += ['', 'All endpoint replays, frozen-state restoration, live hashes, independent audits and closing checks passed. The synthesis rechecks every listed sealed inventory, the candidate hash and all 22 model/tokenizer files. All recorded workers are gone. No fit, tuning, optimizer, readout, capacity, threshold, Arabic, final-pool or generation change occurred. New external spend was USD0; the historical approximately USD30 ledger remains unresolved.', '',
        'This answers the first two open validation questions in the [weekly dossier](research/RESEARCH_WEEKLY_ANALYSIS_2026-09-02_TO_2026-09-09.md): the registered order tests passed, fresh familiar contexts passed, and the tested training-unseen whole-ID bank failed. It does not establish a universal obstruction or resolve the internal cause of failed identity transfer.', '',
        'Primary reports:', '']
    for row in result['later_stages']:
        lines.append(f"- {row['name']}: [native]({row['native_report']}), [actual text]({row['text_report']}).")
    lines += [f'- [Source-order ladder]({RESEARCH / "latent_binding_order_ladder_20260910/REPORT.md"}).', '',
        'Machine-readable counts, intervals, seals, job records and architecture identity are in SUMMARY.json. This is a reporting artifact only; no additional inference or analysis-driven fitting was performed.', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['save', 'verify'])
    command = parser.parse_args().command
    result = build()
    rendered = report(result)
    if command == 'save':
        assert not (ROOT / 'seal.json').exists()
        with (ROOT / 'SUMMARY.json').open('x') as handle:
            json.dump(result, handle, indent=2)
            handle.write('\n')
        with (ROOT / 'REPORT.md').open('x') as handle:
            handle.write(rendered)
    else:
        assert read(ROOT / 'SUMMARY.json') == result
        assert (ROOT / 'REPORT.md').read_text() == rendered
        if (ROOT / 'seal.json').exists():
            assert read(ROOT / 'seal.json')['files'] == inventory(ROOT)
    print(json.dumps(dict(status='PASS', parents=len(result['seals']), models=result['model_tokenizer_files'], verdict=result['verdict'])))


if __name__ == '__main__':
    main()
