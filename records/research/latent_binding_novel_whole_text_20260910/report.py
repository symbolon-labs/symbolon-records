"""Saved-evidence report only; never runs models or changes frozen decisions."""
import argparse
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text())
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def render():
    a=read(ROOT/'summary.json');assert a['status']=='PASS'
    for name,h in read(ROOT/'ANALYSIS_DEPENDENCIES.json')['files'].items():assert sha(Path(name))==h
    native=read(ROOT.parent/'latent_binding_novel_whole_native_20260910/analysis.json')
    counts=[];paired=[];metrics=[];errors=[];compute=[]
    for condition in ['original','changed']:
        v=a['counts'][condition]
        counts.append(f"| {condition} | {a['latent_counts'][condition]} | {v['sender_prediction']} | {v['actual_prediction']} | {v['oracle_prediction']} |")
        v=a['paired_latent_minus_actual_text'][condition]
        paired.append(f"| {condition} | {v['mean']:.6f} | {v['ci95']} | {v['positive_fraction']:.6f} | {v['table_means']} |")
        for arm,measure in a['score_summary'][condition].items():
            metrics.append(f"| {condition}: {arm} | {measure['nll']['mean']:.6f} | {measure['margin']['mean']:.6f} | {measure['margin']['median']:.6f} |")
    for row in a['errors']:
        errors.append(f"| {row['condition']} | {row['example_id']} | {row['query']} | {row['answer']} | {row['sender_prediction']} | {row['actual_prediction']} |")
    if not errors:errors=['| Both conditions | — | — | — | No errors | No errors |']
    total=0
    for role in ['sender','receiver']:
        s=read(ROOT/role/'status.json');j=read(ROOT/f'job_{role}/completion.json');check=read(ROOT/f'{role}_audit.json')
        assert check==a[role+'_audit'] and check['status']=='PASS' and s['completed'] and j['exit_code']==0 and not j['timeout']
        assert check['outputs_sha256']==sha(ROOT/role/'outputs.pt')
        total+=j['elapsed_s']
        compute.append(f"| {role} | {s['contexts']} | {s['prompt_tokens']} | {s['inference_s']:.3f} | {j['elapsed_s']:.3f} | {s['peak_rss_kib']/1024**2:.3f} | {s['parity']['max_difference']:.9g} |")
    return f'''# Novel whole-ID gate: actual text comparison

**{a['verdict']}.** Native comparator is the same frozen d0f16cb9 candidate on
one burned novel DEV191003 trial, eight fresh semantic tables/32queries with eight
training-unseen whole IDs. Native original{native['original_target']['correct']['correct']}/32,
shuffled donor{native['shuffled_donor_target']['correct']}/32, changed donor
{native['changed_target']['wrong_binding']['correct']}/32; minima31/31/28 plus all
controls and paired requirements. Text completion cannot rescue native failure.
No fit, tuning, architecture/readout/capacity/threshold change occurred.

| Condition | Native | Sender ID correct | Actual receiver text | Oracle receiver text |
|---|---:|---:|---:|---:|
{chr(10).join(counts)}

All64 source/query pairs were actually scored afresh, all128 two-token IDs,
full-vocabulary log-softmax/float32 sum/argmax. Actual sender ID string went to
frozen receiver copy prompt; oracle copying is separate. No historical scores
or perfect messages replace actual text. The sender sees S+Q/chat/unprefixed
IDs; latent sees S-only/plain query/leading-space IDs. This access/template
mismatch prevents an equal-access or efficiency superiority interpretation.

| Condition | Latent-minus-text mean |95% CI | Positive fraction | Eight table means |
|---|---|---|---|---|
{chr(10).join(paired)}

Paired eight-table bootstrap,10000 draws, historical DEV seed95002. Four queries
stay together; familiar-ID dependence and tiny-bank size limit population claims.
No superiority claim when intervals include zero. Text has no accuracy gate.

| Condition/arm | Mean NLL | Mean margin | Median margin |
|---|---:|---:|---:|
{chr(10).join(metrics)}

| Condition | Example | Query | Target | Sender | Actual receiver |
|---|---|---|---|---|---|
{chr(10).join(errors)}

Per-color/table/ID and all per-example metrics retained in summary.json. Sender
batch4 saves each chunk and repeats first4 exactly; receiver batch8 repeats all
actual/oracle messages exactly. Both endpoints pass native direct-score<=1e-4,
exact argmax/replay, input/token/message joins and live unchanged-state hashes.
No gradients/hooks/training. Independent audits rehash22snapshot files and the
native parent seal. Eight tests/seven fresh closing checks retained at closure.

| Endpoint | Contexts | Prompt tokens | Scoring s | Job wall s | Peak GiB | Parity max error |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(compute)}

Total endpoint wall{total:.3f}s/{total/60:.2f}min, no timeout/retry, no workers at
seal. User-authorized no text wall-time kill; native remains cap2400s. USD0new
external, historic~USD30unreconciled. No cloud/browser/Rust/commit/push.
Freeze{sha(ROOT/'FREEZE.json')}; independent summary{sha(ROOT/'summary.json')}.

Passing establishes small training-unseen positive whole-ID combination
validation on this new context bank. These eight IDs have older-candidate
evaluation/design-selection history; they are not pristine tests or novel
characters. All128candidates/negative CE/components were exposed; endpoint
pretraining novelty is unknown. The original64FINALpool remains reserved.
Native failure rejects this fixed candidate at the novel whole-ID gate regardless
of text success. No second trial,tuning,new adapter or threshold change follows.
Arabic specialist-to-generalist transfer,freegeneration and capability
preservation remain unproven and do not launch automatically.
'''
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['write','verify']);q=p.parse_args();value=render()
    if q.command=='write':
        with (ROOT/'REPORT.md').open('x') as f:f.write(value)
    else:assert (ROOT/'REPORT.md').read_text()==value
    print('REPORT',q.command,'PASS')
