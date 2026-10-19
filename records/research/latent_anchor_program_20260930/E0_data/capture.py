"""E0 sender capture: Qwen3-1.7B block-13 output for every table and its wrong-binding twin.

Also recaptures the sealed V1 tiny split into captures/tiny_check.pt for port validation.
Usage: python -B capture.py [split ...]   (default: tiny_check train dev_known dev_novel)
FINAL splits are never captured here.
"""
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lab  # noqa: E402

SEALED_TINY = lab.R / 'latent_binding_v1_20260906/data/tiny/examples.json'
ALLOWED = {'tiny_check', 'train', 'dev_known', 'dev_inblock_pure', 'dev_oob_pure', 'dev_inblock_mixed', 'dev_oob_mixed'}


def records(split):
    if split == 'tiny_check':
        return json.loads(SEALED_TINY.read_text())['records']
    return json.loads((lab.SPLITS_DIR / f'{split}.json').read_text())['records']


def main(splits):
    assert set(splits) <= ALLOWED, f'refusing to capture {set(splits) - ALLOWED}'
    dev = lab.device()
    t0 = time.monotonic()
    model, tok, digest = lab.load('sender', dev)
    lab.event('sender_loaded', hash=digest, s=round(time.monotonic() - t0, 1), device=str(dev))
    lab.CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = lab.CAPTURE_DIR / 'MANIFEST.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    captured = []

    def hook(module, args, output):
        captured.append(output.detach().float().cpu().clone())

    handle = model.model.layers[lab.SENDER_LAYER].register_forward_hook(hook)
    try:
        for split in splits:
            recs = records(split)
            sources = [r['source'] for r in recs] + [r['wrong_source'] for r in recs]
            ids = lab.tokens(tok, sources)
            assert ids.shape == (2 * len(recs), lab.SOURCE_TOKENS)
            captured.clear()
            t = time.monotonic()
            with torch.no_grad():
                for i in range(0, len(ids), 16):
                    model.model(input_ids=ids[i:i + 16].to(dev), use_cache=False)
            hidden = torch.cat(captured)
            n = len(recs)
            assert hidden.shape == (2 * n, lab.SOURCE_TOKENS, lab.SENDER_WIDTH) and torch.isfinite(hidden).all()
            path = lab.CAPTURE_DIR / f'{split}.pt'
            torch.save(dict(hidden=hidden[:n], wrong_hidden=hidden[n:], source_ids=ids), path)
            manifest[split] = dict(sha256=lab.sha(path), tables=n, shape=list(hidden.shape),
                                   layer_zero_based=lab.SENDER_LAYER, device=str(dev),
                                   seconds=round(time.monotonic() - t, 2), sender_parameter_hash=digest)
            lab.event('captured', split=split, tables=n, s=manifest[split]['seconds'])
    finally:
        handle.remove()
    assert lab.parameter_hash(model.cpu()) == digest
    manifest_path.write_text(json.dumps(manifest, indent=1) + '\n')


if __name__ == '__main__':
    main(sys.argv[1:] or ['tiny_check', 'train', 'dev_known', 'dev_inblock_pure', 'dev_oob_pure', 'dev_inblock_mixed', 'dev_oob_mixed'])
