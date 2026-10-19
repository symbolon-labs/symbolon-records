"""Descriptive tiny-set diagnostics, never a held-out success criterion."""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def describe(values):
    x = np.asarray(values, dtype=float)
    assert len(x) and np.isfinite(x).all()
    return dict(mean=float(x.mean()), median=float(np.median(x)),
                minimum=float(x.min()), maximum=float(x.max()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('evaluation', type=Path)
    parser.add_argument('--save', type=Path)
    parser.add_argument('--curve-csv', type=Path)
    parser.add_argument('--control-csv', type=Path)
    args = parser.parse_args()
    data = read(ROOT / 'data/tiny/examples.json')
    tables = {r['group']: r for r in data['records']}
    records = read(args.evaluation / 'per_example.json')
    by_id = {r['id']: r for r in records}
    metrics = read(args.evaluation / 'metrics.json')
    result = dict(scope='Descriptive tiny-training diagnostics; not generalization evidence',
                  evaluation=args.evaluation.name, gate=metrics['gate'], conditions={})
    for condition in metrics['conditions']:
        id_present, first, second, exact, margins = [], [], [], [], []
        predicted_by_group = defaultdict(list)
        response = defaultdict(list)
        predicted_source_color = Counter()
        for row in records:
            output = row['conditions'][condition]
            pred = output['prediction']
            if condition == 'shuffled':
                table = tables[by_id[row['donor_id']]['group']]
                source = table['bindings']
                target = row['donor_answer']
            elif condition == 'wrong_binding':
                source = tables[row['group']]['wrong_bindings']
                target = row['wrong_answer']
            else:
                source = tables[row['group']]['bindings']
                target = row['answer']
            present = pred in source.values()
            id_present.append(present)
            first.append(pred[0] == target[0])
            second.append(pred[1] == target[1])
            exact.append(pred == target)
            predicted_by_group[row['group']].append(pred)
            color = next((c for c, identity in source.items() if identity == pred), 'outside_table')
            predicted_source_color[row['color'] + '->' + color] += 1
            ix = metrics['candidates'].index(target)
            scores = output['answer_log_probabilities']
            margins.append(scores[ix] - max(v for j, v in enumerate(scores) if j != ix))
            for key, value in output['diagnostics'].items():
                if isinstance(value, (float, int)):
                    response[key].append(value)
        n = len(records)
        result['conditions'][condition] = dict(
            target='donor/counterfactual answer for content swaps; original answer otherwise',
            exact_id_accuracy=sum(exact)/n,
            predicted_id_in_source_rate=sum(id_present)/n,
            exact_given_predicted_id_in_source=sum(exact)/sum(id_present) if sum(id_present) else None,
            predicted_first_character_match=sum(first)/n,
            predicted_second_character_match=sum(second)/n,
            distinct_answers_per_table=describe([len(set(v)) for v in predicted_by_group.values()]),
            binding_confusion=dict(sorted(predicted_source_color.items())),
            correct_minus_best_other_margin=describe(margins),
            query_position_reader_diagnostics={k: describe(v) for k, v in response.items()})
    payload = torch.load(args.evaluation / 'outputs.pt', weights_only=True)
    latent = {}
    representative = {}
    for index, row in enumerate(records):
        representative.setdefault(row['group'], index)
    for name, tensor in payload['memories'].items():
        slot_norm = tensor.norm(dim=-1).numpy()
        flat_norm = tensor.flatten(1).norm(dim=-1).numpy()
        latent[name] = dict(shape=list(tensor.shape), slot_norm=describe(slot_norm.flatten()),
                            example_norm=describe(flat_norm))
        unique = tensor[list(representative.values())].double()
        if name != 'zero':
            flat = unique.flatten(1)
            normalized = flat / flat.norm(dim=-1, keepdim=True)
            cosine = normalized @ normalized.T
            mask = ~torch.eye(len(unique), dtype=torch.bool)
            slots = unique / unique.norm(dim=-1, keepdim=True)
            within = slots @ slots.transpose(-1, -2)
            slot_mask = ~torch.eye(8, dtype=torch.bool)
            centered = flat - flat.mean(0)
            latent[name].update(
                distinct_table_count=len(unique),
                between_table_flat_cosine=describe(cosine[mask].numpy()),
                within_table_slot_cosine=describe(within[:, slot_mask].flatten().numpy()),
                between_table_centered_rms=float(centered.square().mean().sqrt()),
                centered_singular_values=torch.linalg.svdvals(centered).tolist())
    result['latent_norms'] = latent
    correct_memory = payload['memories']['correct'].double().flatten(1)
    wrong_memory = payload['memories']['wrong_binding'].double().flatten(1)
    memory_cosine = torch.nn.functional.cosine_similarity(correct_memory, wrong_memory)
    result['same_id_counterfactual'] = dict(
        all_targets_change=all(row['answer'] != row['wrong_answer'] for row in records),
        prediction_unchanged_rate=sum(row['conditions']['correct']['prediction'] ==
                                      row['conditions']['wrong_binding']['prediction'] for row in records) / len(records),
        paired_memory_cosine=describe(memory_cosine.numpy()),
        absolute_candidate_score_change=describe([
            abs(a-b) for row in records for a, b in zip(
                row['conditions']['correct']['answer_log_probabilities'],
                row['conditions']['wrong_binding']['answer_log_probabilities'])]))
    if args.curve_csv:
        curve = [json.loads(line) for line in (ROOT / 'run_tiny/curve.jsonl').read_text().splitlines()]
        with args.curve_csv.open('x') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(curve[0]))
            writer.writeheader()
            writer.writerows(curve)
    if args.control_csv:
        control_rows = []
        for path in sorted((ROOT / 'run_tiny').glob('eval_*/metrics.json')):
            measured = read(path)
            step = int(path.parent.name.split('_')[1])
            for name, condition in measured['conditions'].items():
                control_rows.append(dict(step=step, condition=name,
                    accuracy=condition['accuracy'], mean_id_nll=condition['mean_id_nll'],
                    donor_following=condition['donor_following'],
                    **{color + '_accuracy': accuracy for color, accuracy in condition['per_color'].items()},
                    reader_gate=measured['gate'],
                    elapsed_s=measured['times'][name]['elapsed_s'],
                    cached=measured['times'][name]['cached']))
        with args.control_csv.open('x') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(control_rows[0]))
            writer.writeheader()
            writer.writerows(control_rows)
    if args.save:
        with args.save.open('x') as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write('\n')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
