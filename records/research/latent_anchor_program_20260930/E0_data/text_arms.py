"""E0 text arms (TEXT_PROTOCOL.md of binding V1, unchanged prompts).

Actual:  Qwen3-1.7B reads SENDER_INSTRUCTION + table + '\\n\\n' + query (chat template, thinking off),
         picks an ID by full 128-candidate two-token scoring; SmolLM2 copies that ID.
Oracle:  SmolLM2 copies the gold ID.
Changed: the same, with the wrong-binding table (answers all change).
First validates the port on the sealed tiny split against latent_binding_v1_20260906/text_tiny.
Usage: python -B text_arms.py   -> E0_data/text/<split>.json + TEXT_SUMMARY.json
"""
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lab  # noqa: E402

HERE = Path(__file__).resolve().parent
V1 = lab.R / 'latent_binding_v1_20260906'
OUT = HERE / 'text'


class Native:
    """Shim so lab.score_cached can score an unhooked endpoint."""
    def __init__(self, model):
        self.model, self.enabled, self.memory = model, False, None


def splits():
    out = {'tiny_check': json.loads((V1 / 'data/tiny/examples.json').read_text())}
    for name in ['dev_known', 'dev_oob_pure', 'dev_inblock_pure', 'dev_oob_mixed', 'dev_inblock_mixed']:
        out[name] = json.loads((lab.SPLITS_DIR / f'{name}.json').read_text())
    return out


def sender_stage(data_by_split, dev):
    model, tok, digest = lab.load('sender', dev)
    bank = lab.AnswerBank(tok, leading=False)
    shim = Native(model)
    results = {}
    for name, data in data_by_split.items():
        res = {}
        for kind, src in [('original', 'source'), ('changed', 'wrong_source')]:
            prompts = [lab.render(tok, lab.SENDER_INSTRUCTION + data['records'][r['group']][src] + '\n\n' + r['query'])
                       for r in data['rows']]
            ids = bank.prompts(prompts)
            t = time.monotonic()
            s = lab.score_cached(shim, ids, bank, None, batch=8, dev=dev)
            res[kind] = dict(prompts=prompts, input_ids=ids, token_lp=s['token_lp'],
                             prediction=[lab.IDS[int(p)] for p in s['prediction']], seconds=time.monotonic() - t)
            lab.event('sender_text', split=name, arm=kind, rows=len(prompts), tokens=ids.shape[1],
                      s=round(res[kind]['seconds'], 1))
        results[name] = res
    del model
    if dev.type == 'mps':
        torch.mps.empty_cache()
    return results, digest


def receiver_stage(data_by_split, sent, dev):
    model, tok, digest = lab.load('receiver', dev)
    bank = lab.AnswerBank(tok, leading=False)
    shim = Native(model)
    identities = sorted(set(lab.IDS))
    prompts = [lab.render(tok, lab.RECEIVER_INSTRUCTION + 'Answer ID: ' + i) for i in identities]
    ids = bank.prompts(prompts)
    s = lab.score_cached(shim, ids, bank, None, batch=32, dev=dev)
    copy = {i: dict(prediction=lab.IDS[int(s['prediction'][k])], token_lp=s['token_lp'][k]) for k, i in enumerate(identities)}
    return copy, prompts, ids, digest


def main():
    dev = lab.device()
    OUT.mkdir(exist_ok=True)
    data_by_split = splits()
    sent, sender_hash = sender_stage(data_by_split, dev)
    copy, rprompts, rids, receiver_hash = receiver_stage(data_by_split, sent, dev)

    # validation on sealed tiny
    sealed_s = json.loads((V1 / 'text_tiny/sender.json').read_text())
    sealed_r = json.loads((V1 / 'text_tiny/receiver.json').read_text())
    tiny = sent['tiny_check']['original']
    sealed_lp = torch.tensor([o['token_log_probabilities'] for o in sealed_s['outputs']])
    val = dict(sender_prompts_equal=tiny['prompts'] == sealed_s['prompts'],
               sender_ids_equal=tiny['input_ids'].tolist() == sealed_s['input_ids'],
               sender_max_abs_token_lp=(tiny['token_lp'] - sealed_lp).abs().max().item(),
               sender_predictions_equal=tiny['prediction'] == [o['prediction'] for o in sealed_s['outputs']],
               receiver_prompts_equal=all(p == rprompts[sorted(lab.IDS).index(i)]
                                          for p, i in zip(sealed_r['prompts'], sealed_r['identities'])),
               receiver_predictions_equal=all(copy[i]['prediction'] == lab.IDS[int(sealed_r['result']['prediction'][k])]
                                              for k, i in enumerate(sealed_r['identities'])))
    val['pass_'] = all(v for k, v in val.items() if k not in ('sender_max_abs_token_lp',)) and val['sender_max_abs_token_lp'] <= 1e-3
    lab.event('text_validation', **val)

    summary = dict(validation=val, sender_parameter_hash=sender_hash, receiver_parameter_hash=receiver_hash, splits={})
    for name, data in data_by_split.items():
        rows = data['rows']
        novel = [bool(r.get('novel_answer', False)) for r in rows]
        res = {}
        for kind, key in [('original', 'answer'), ('changed', 'wrong_answer')]:
            preds = sent[name][kind]['prediction']
            gold = [r[key] for r in rows]
            sender_ok = [p == g for p, g in zip(preds, gold)]
            actual_ok = [copy[p]['prediction'] == g for p, g in zip(preds, gold)]
            oracle_ok = [copy[g]['prediction'] == g for g in gold]
            res[kind] = dict(sender=sum(sender_ok) / len(rows), actual_text=sum(actual_ok) / len(rows),
                             oracle_text=sum(oracle_ok) / len(rows))
            if any(novel) and kind == 'original':
                nr = [i for i, v in enumerate(novel) if v]
                res[kind].update(novel_rows_sender=sum(sender_ok[i] for i in nr) / len(nr),
                                 novel_rows_actual_text=sum(actual_ok[i] for i in nr) / len(nr),
                                 novel_rows_oracle_text=sum(oracle_ok[i] for i in nr) / len(nr), novel_rows_n=len(nr))
            if kind == 'changed' and any(r.get('novel_wrong_answer') for r in rows):
                nr = [i for i, r in enumerate(rows) if r.get('novel_wrong_answer')]
                res[kind].update(novel_target_actual_text=sum(actual_ok[i] for i in nr) / len(nr), novel_target_n=len(nr))
        summary['splits'][name] = res
        per_row = [dict(id=r['id'], answer=r['answer'], wrong_answer=r['wrong_answer'],
                        sender_original=sent[name]['original']['prediction'][i],
                        sender_changed=sent[name]['changed']['prediction'][i],
                        actual_original=copy[sent[name]['original']['prediction'][i]]['prediction'],
                        actual_changed=copy[sent[name]['changed']['prediction'][i]]['prediction'],
                        oracle=copy[r['answer']]['prediction']) for i, r in enumerate(rows)]
        (OUT / f'{name}.json').write_text(json.dumps(dict(results=res, rows=per_row), indent=1) + '\n')
        lab.event('text_split', split=name, **{k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in res.items()})
    (HERE / 'TEXT_SUMMARY.json').write_text(json.dumps(summary, indent=1) + '\n')


if __name__ == '__main__':
    main()
