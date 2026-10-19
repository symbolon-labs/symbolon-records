"""Saved-results report, no endpoint or gate mutation."""
import argparse
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text())
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def render():
    a=read(ROOT/'analysis.json');audit=read(ROOT/'audit.json');s=read(ROOT/'run/status.json');j=read(ROOT/'job/completion.json')
    assert audit['status']=='PASS' and audit['analysis_sha256']==sha(ROOT/'analysis.json')
    table=[]
    for arm,v in a['original_target'].items():
        table.append(f"| {arm} | {v['correct']} | {a['changed_target'][arm]['correct']} | {v['mean_margin']:.6f} |")
    paired=[]
    for part,comparisons in a['paired_controls'].items():
        for arm,v in comparisons.items():paired.append(f"| {part}: active minus {arm} | {v['accuracy']['ci95']} | {v['margin']['ci95']} |")
    errors=[];routes=[];data=read(ROOT/'run/data.json')
    for arm,key,field in [('correct','original','source'),('wrong_binding','changed','wrong_source')]:
        summary=a[key+'_target'][arm];actual=a['geometry'][arm]['routes']
        expected=[next(i for i,line in enumerate(data['records'][r['group']][field].splitlines()) if line.split(' = ')[1]==r['color']) for r in data['rows']]
        routes.append(dict(arm=arm,query_site3=sum(x[0]==y for x,y in zip(actual,expected)),query_site4=sum(x[1]==y for x,y in zip(actual,expected))))
        for i,(pred,target) in enumerate(zip(summary['predictions'],summary['target_ids'])):
            if pred!=target:errors.append(f"| {arm} | {data['rows'][i]['id']} | {target} | {pred} | {actual[i]} | {expected[i]} |")
    if not errors:errors=['| Both | — | — | No errors | — | — |']
    return f'''# Novel whole-ID native binding gate

**{a['verdict']}.** Original{a['original_target']['correct']['correct']}/32,
shuffled donor{a['shuffled_donor_target']['correct']}/32,
changed donor{a['changed_target']['wrong_binding']['correct']}/32 against fixed31/31/28.
Independent source/NumPy/scalar/model audit PASS. Actual text comparison remains
PENDING; no complete novel-binding or Arabic claim follows from this native substage alone.

This is one conservatively burned novel DEV191003 trial: eight fresh semantic tables,
32 queries, eight training-unseen whole IDs, original/changed mappings disjoint from CAL.
This new context bank had not been evaluated. These IDs have older-candidate
evaluation history; they are not pristine tests. All128scored candidates,negative
CE and components were seen. New sender whole/isolated captures ran twice in
reverse order,with known historical sentinels,and were independently sealed.
NOVEL_LOCK.json in this tree burns the bank before the first receiver forward.
The closed historical data tree was not modified. No retry or selection.

Candidate {a['checkpoint']}, all11entries/569857parameters, isolated two-token
source ID states and shared ID queries, typed8x64 float32 channel (2048bytes),
qualified SpanReader and both frozen endpoints are unchanged. Source cache
joining uses raw source ID tokens only. No target/query-selected source field,
prototype lookup, optimizer, fit, tuning, readout/threshold/capacity adjustment.
Full128 A0–P7 native both-token continuation likelihood, leading-space IDs,
float32 vocabulary log-softmax/sum/argmax. Both query sites3/4 intervene; answer
position5 remains unchanged. Candidate prefixes for scoring token2 are ordinary
likelihood scoring, not a supplied answer. No free generation.

Counts /32; changed-target counts are separate from original-target controls.

| Arm | Original correct | Changed correct | Original mean margin |
|---|---:|---:|---:|
{chr(10).join(table)}

Shuffled scores exactly match correct donor rows; zero exactly receiver-only.
Receiver-only/canonical original<=4; all shuffled/changed/random original<=8.
Random wires match slot norms, not necessarily receiver residual energy.
All nine accuracy AND margin paired lower95% bounds must be positive, in
addition to all absolute counts. No component score overrides a failure.

| Active/control comparison | Accuracy95% CI | Margin95% CI |
|---|---|---|
{chr(10).join(paired)}

Same preceding-gate bootstrap: seed95002,10000 paired eight-table bootstrap draws,
all four queries carried together. Tiny-bank and repeated-ID dependence limit
population claims. Replays are deterministic checks, not independent examples.
No paired fresh-minus-CAL statistic is computed across different examples.

Requested-record route counts /32 at sites3/4: {routes}.
Routes are descriptive checks, never a replacement accuracy gate.

| Active arm | Example | Target | Prediction | Routes3/4 | Expected slot |
|---|---|---|---|---|---|
{chr(10).join(errors)}

Every arm and prefill was actually replayed exactly; full score arrays are
retained in run/evaluation and run/repeat. Historical native stock and active
sentinels use their own exact input IDs/labels. Fresh stock5/6 repeated,
disabled/zero/restored exact. Historical source-independent score/hidden
comparisons join identical query tokens, not row indices. Direct scorer error
{a['direct_score_error']:.10g}<=1e-4 with exact argmax; reversed codebook exact.
All11states unchanged, hooks removed, true stock restored, live receiver hash
{s['receiver_after']}. Independent audit rederives all source joins, hybrid
splices, NumPy wire/state equations, scalar score reductions and bootstrap gates;
all22model/tokenizer files rehashed. Exact previously registered kernel-only
migration7.2.2->7.2.3 allowed, all other provenance fixed. Seven harness plus six
qualified-span tests and six fresh closing commands retained at closure.

Actual64-request sender-text-receiver comparison is mandatory after this valid
native run regardless of PASS/NO-GO. S+Q/chat text uses unprefixed full-ID forced
choice; latent uses S-only/plain queries. Actual messages and oracle copying
must be reported separately. No historical text score substitution, equal-access
or latent-over-text superiority claim. No new text outcome exists in this tree.

One cap2400s local receiver job: {j['elapsed_s']:.6f}s wall,
{s['elapsed_s']:.6f}s body, peak{s['peak_rss_kib']}KiB, no timeout/retry.
Torch4/1, seed67108. USD0new external; historic~USD30 remains unreconciled.
No cloud/browser/Rust/commit/push. Freeze{sha(ROOT/'FREEZE.json')}; immutable
closing_checks.json/seal.json record closure. No worker remains at seal.

A native failure rejects this fixed candidate at the novel whole-ID gate. A
passing native result still requires actual text completion before claiming
small training-unseen whole-ID combination validation. Old evaluation history
limits untouched-test claims. The original64FINALpool stays reserved; Arabic
specialist-to-generalist transfer,generation and capability preservation remain
unproven and do not launch automatically. No tuning or new adapter follows.
'''
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['save','verify']);q=p.parse_args();v=render()
    if q.command=='save':
        with (ROOT/'REPORT.md').open('x') as f:f.write(v)
    else:assert (ROOT/'REPORT.md').read_text()==v
    print('REPORT VERIFIED')
