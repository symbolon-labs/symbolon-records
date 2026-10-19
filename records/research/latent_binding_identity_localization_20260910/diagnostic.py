"""Saved-array identity localization. No endpoint execution or fitting."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parent
R=ROOT.parent
CAND=R/'latent_binding_shared_id_query_mechanics_20260908/run/candidate.pt'
BANKS={'familiar':R/'latent_binding_fresh_known_native_20260910','novel':R/'latent_binding_novel_whole_native_20260910'}
PINS={'latent_binding_fresh_known_native_20260910':'33d50d74532f21e37ff9dc99d4149f3b8b379e527370ba0ca92e15dfadf8a9ee',
      'latent_binding_novel_whole_native_20260910':'a4025f8f941ff2551616c2a4e6464879a0eef1f7cafc47c415474b926c661e98',
      'latent_binding_validation_ladder_20260910':'e59dd335b0efc091396f6efbf3f3241b55ff90567b1de877b1b7b37a211be166'}
def read(p):return json.loads(p.read_text())
def load(p):return torch.load(p,weights_only=True)
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,x):
    with p.open('x') as f:json.dump(x,f,indent=2);f.write('\n')
def inventory(root):return {str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*')) if p.is_file() and p!=root/'seal.json'}
def parents():
    for name,h in PINS.items():
        root=R/name;assert sha(root/'seal.json')==h;assert read(root/'seal.json')['files']==inventory(root)
    assert sha(CAND)=='d0f16cb9705ba53daf6cc8f2bfad03a08efa96cbb4f77dd836c9533e948cba87'
def identity():return dict(code={n:sha(ROOT/n) for n in ['diagnostic.py','test_diagnostic.py','PREREGISTRATION.md']},parents=PINS,candidate=sha(CAND),torch=torch.__version__,numpy=np.__version__)
def check():parents();assert identity()==read(ROOT/'FREEZE.json')['identity']
def metric(x,c,cosine=False):
    x=x.double();c=c.double()
    if cosine:return 1-F.normalize(x,dim=-1)@F.normalize(c,dim=-1).T
    return ((x[:,None]-c[None])**2).sum(-1)
def reference_indices(labels):return [labels.index(label) for label in sorted(set(labels))]
def bank(root,state):
    data=read(root/'run/data.json');analysis=read(root/'analysis.json')
    source=load(root/'run/source.pt')['payload'];memory=load(root/'run/memories.pt')['memories']
    stages={};meta=[];checks=[]
    for arm,condition,field,key in [('correct','original','source','hidden'),('wrong_binding','changed','wrong_source','wrong_hidden')]:
        saved=load(root/f'run/prefill_{arm}.pt');cap=saved['first']['capture']
        assert all(torch.equal(v,saved['repeat']['capture'][k]) for k,v in cap.items())
        assert torch.equal(saved['first']['logits'],saved['repeat']['logits'])
        projected=F.linear(memory[arm],state['projector.weight'])
        # Different batch kernels may round float32 projection differently;
        # frozen NumPy relative bound below applies, actual saved memory is used.
        projection_error=float((projected-cap['memory']).abs().max()/cap['memory'].abs().max().clamp_min(1e-12))
        assert projection_error<=1e-5
        value=F.linear(cap['memory'][:,::2].double(),state['reader.value.weight'].double())
        selected=cap['hard'].double()@value
        mapped=F.linear(selected,state['reader.output.weight'].double()).float()
        raw=state['reader.gate']*mapped
        assert torch.equal(raw,cap['raw_delta'])
        boundary=cap['base'][:,3:5]+raw[:,3:5]*state['reader.norm_scale']
        assert torch.equal(boundary,cap['output'][:,3:5])
        # Independent NumPy equations, using full captured arrays, not Torch outputs.
        a=cap['memory'].numpy().astype('float64');v=a[:,::2]@state['reader.value.weight'].numpy().astype('float64').T
        selected_np=cap['hard'].numpy().astype('float64')@v
        mapped_np=(selected_np@state['reader.output.weight'].numpy().astype('float64').T).astype('float32')
        numpy_errors={}
        for name,left,right in [('value',v,value.numpy()),('selected',selected_np,selected.numpy()),('mapped',mapped_np,mapped.numpy()),('projected',memory[arm].numpy().astype('float64')@state['projector.weight'].numpy().astype('float64').T,cap['memory'].numpy())]:
            err=float(np.max(np.abs(left-right))/max(np.max(np.abs(right)),1e-12));assert err<=1e-5;numpy_errors[name]=err
        checks.append(dict(condition=condition,raw_delta_exact=True,boundary_exact=True,saved_repeats_exact=True,projection_relative=projection_error,numpy_relative=numpy_errors))
        for i,row in enumerate(data['rows']):
            table=data['records'][row['group']][field].splitlines()
            slot=next(j for j,line in enumerate(table) if line.split(' = ')[1]==row['color'])
            label=table[slot].split(' = ')[0]
            m=dict(id=label,group=row['group'],condition=condition,query=row['id'],color=row['color'],slot=slot,
                routes=cap['hard'][i,3:5].argmax(-1).tolist(),prediction=analysis[condition+'_target'][arm]['predictions'][i])
            meta.append(m)
            x={'sender_field':source[key][row['group'],[5*slot,5*slot+1]].reshape(-1),'wire64':memory[arm][i,2*slot],
               'projected_memory':cap['memory'][i,2*slot],'reader_value':value[i,slot]}
            for j in [3,4]:x.update({f'selected_{j}':selected[i,j],f'output_map_{j}':mapped[i,j],f'boundary_{j}':cap['output'][i,j]})
            for name,tensor in x.items():stages.setdefault(name,[]).append(tensor.clone())
    return dict(stages={k:torch.stack(v) for k,v in stages.items()},meta=meta,checks=checks)
def summarize(x,rows,familiar,familiar_rows):
    labels=[r['id'] for r in rows];ids=sorted(set(labels));ri=reference_indices(labels);c=x[ri].double();dist=metric(c,c).sqrt()
    pairs=[dict(a=ids[i],b=ids[j],euclidean=float(dist[i,j]),cosine=float(metric(c,c,True)[i,j])) for i in range(len(c)) for j in range(i)]
    spread=max(float((x[i].double()-c[ids.index(label)]).norm()) for i,label in enumerate(labels))
    s=torch.linalg.svdvals(c-c.mean(0));rank={str(dtype):int((s>torch.finfo(dtype).eps*max(c.shape)*s[0]).sum()) for dtype in [torch.float32,torch.float64]}
    eligible=[i for i,r in enumerate(rows) if r['group']!=rows[ri[ids.index(r['id'])]]['group']]
    primitive={};nn={};fids=sorted({r['id'] for r in familiar_rows});fc=familiar[reference_indices([r['id'] for r in familiar_rows])]
    for cosine,name in [(False,'euclidean'),(True,'cosine')]:
        distances=metric(x,c,cosine);pred=distances.argmin(-1);decoded=[ids[i] for i in pred.tolist()]
        if not cosine:
            affine=(2*x.double()@c.T-(c*c).sum(-1)).argmax(-1);assert torch.equal(affine,pred)
        primitive[name]=dict(correct=sum(decoded[i]==labels[i] for i in eligible),n=len(eligible),
            shifted_label_control_correct=sum(ids[(int(pred[i])+1)%len(ids)]==labels[i] for i in eligible),
            predictions=decoded,eligible_indices=eligible,
            ties=sum(int((v==v.min()).sum())>1 for v in distances))
        fd=metric(x,fc,cosine);fp=[fids[i] for i in fd.argmin(-1).tolist()]
        agree=[p==r['prediction'] for p,r in zip(fp,rows)];per={label:dict(n=labels.count(label),agreement=sum(a for a,l in zip(agree,labels) if l==label)) for label in ids}
        nn[name]=dict(predictions=fp,agreement=sum(agree),n=len(rows),per_id=per,
            strong_agreement=sum(agree)>=56 and all(v['agreement']>=.75*v['n'] for v in per.values()),
            ties=sum(int((v==v.min()).sum())>1 for v in fd))
        # Independent scalar/NumPy nearest-reference calculation.
        for refs,predicted in [(c,[ids.index(z) for z in decoded]),(fc,[fids.index(z) for z in fp])]:
            xx=x.double().numpy();cc=refs.double().numpy()
            nd=1-xx@cc.T/(np.linalg.norm(xx,axis=1)[:,None]*np.linalg.norm(cc,axis=1)[None,:]) if cosine else np.array([[np.dot(a-b,a-b) for b in cc] for a in xx])
            assert np.array_equal(nd.argmin(-1),predicted)
    return dict(dimension=x.shape[1],unique_ids=len(ids),occurrences=len(rows),within_id_max_l2=spread,
        minimum_between_id_l2=min(p['euclidean'] for p in pairs),minimum_relative_to_max_norm=min(p['euclidean'] for p in pairs)/float(c.norm(dim=-1).max()),
        exact_cross_id_collisions=[p for p in pairs if p['euclidean']==0],pairs=pairs,singular_values=s.tolist(),centered_rank=rank,
        privileged_reference_decode=primitive,familiar_nearest_neighbor=nn)
def run():
    check();assert not (ROOT/'RESULT.json').exists();start=time.monotonic();state=load(CAND)
    banks={name:bank(path,state) for name,path in BANKS.items()}
    results={name:{stage:summarize(x,banks[name]['meta'],banks['familiar']['stages'][stage],banks['familiar']['meta']) for stage,x in b['stages'].items()} for name,b in banks.items()}
    # Re-extract from saved arrays in reverse bank order; no model loaded.
    for name,path in reversed(list(BANKS.items())):
        repeat=bank(path,state);assert repeat['meta']==banks[name]['meta'] and repeat['checks']==banks[name]['checks']
        assert all(torch.equal(v,repeat['stages'][k]) for k,v in banks[name]['stages'].items())
    assert all(torch.equal(v,load(CAND)[k]) for k,v in state.items());check()
    torch.save(banks,ROOT/'stages.pt')
    result=dict(status='PASS',results=results,checks={k:v['checks'] for k,v in banks.items()},reverse_repeat_exact=True,
        candidate_unchanged=True,no_endpoint_loaded=True,no_fitting=True,no_reset_consumed=True,new_external_spend_usd=0,elapsed_s=time.monotonic()-start)
    write(ROOT/'RESULT.json',result);print(json.dumps(dict(status='PASS',elapsed_s=result['elapsed_s'])))
def report():
    a=read(ROOT/'RESULT.json');n=a['results']['novel'];rows=['# No-fitting identity localization','',
        'Saved d0f16cb9 arrays only; native novel-ID verdict remains0/0/0 NO-GO. No endpoint execution, probe fitting, weight change or usage reset.','',
        '| Stage | Min relative separation | Within-ID L2 spread | Privileged Euclidean decode | Familiar NN/native agreement (Euclidean / cosine) |',
        '|---|---:|---:|---:|---:|']
    for name,v in n.items():
        p=v['privileged_reference_decode']['euclidean'];nn=v['familiar_nearest_neighbor']
        rows.append(f"| {name} | {v['minimum_relative_to_max_norm']:.6g} | {v['within_id_max_l2']:.6g} | {p['correct']}/{p['n']} | {nn['euclidean']['agreement']}/64 / {nn['cosine']['agreement']}/64 |")
    collisions={k:v['exact_cross_id_collisions'] for k,v in n.items() if v['exact_cross_id_collisions']}
    rows+=['',f'Exact cross-ID collisions: {collisions or "none in the measured stage codebooks"}.','',
        'Reference decoding uses one labeled novel-ID vector per ID, with evaluation outside its reference semantic table. This is privileged finite-codebook identification. Isolated same-ID representations repeat across contexts, so success is not learned unseen-ID generalization. The Euclidean classifier is exactly the affine rule2*x·c−||c||²; cosine supplies a fixed nonlinear normalization alternative. No classifier parameters were optimized. Shifted-label controls, all pair distances, ranks, predictions and per-ID agreement are saved in RESULT.json.', '',
        'A distinct codebook shows that total identity erasure has not occurred on these eight IDs. It does not rule out partial loss, poor conditioning, inaccessible semantics, or failure on other IDs. If distinguishability persists through boundary_3/4, attributing failure specifically to the final output map is unsupported: identity distinctions survive that map and enter the frozen receiver. Native output usability and causal localization remain separate questions.', '',
        'Familiar nearest-neighbor agreement is a metric-specific association. The prospectively fixed strong-agreement rule is56/64 overall and at least75% for every novel ID; every stage and both metrics are retained. A matching neighbor is not a causal mechanism, and a mismatch is not proof of information loss.', '',
        'Reader.value, hard selection, output mapping and actual receiver-boundary reconstruction were checked against saved native captures; full raw deltas and query boundary states match exactly. Independent NumPy equations and classifiers, reverse bank extraction, state hashes and sealed parent inventories passed. No live endpoint hash is claimed for this saved-array-only run.', '',
        'These eight IDs are absent from56 ancestral positive training IDs but have older evaluation/design-selection history. The novel bank is already burned; these diagnostics are post-outcome and cannot restore its held-out status. FINAL64 and Arabic remain unopened.', '',
        'No bounded repair is selected or executed by this result. Separability can reject total erasure without identifying a causal repair stage. See PREREGISTRATION.md for the fixed method and RESULT.json/stages.pt for complete evidence.','']
    return '\n'.join(rows)
def verify():
    check();banks=load(ROOT/'stages.pt');a=read(ROOT/'RESULT.json')
    for name,b in banks.items():
        for stage,x in b['stages'].items():assert summarize(x,b['meta'],banks['familiar']['stages'][stage],banks['familiar']['meta'])==a['results'][name][stage]
    assert (ROOT/'REPORT.md').read_text()==report()
    if (ROOT/'seal.json').exists():assert read(ROOT/'seal.json')['files']==inventory(ROOT)
    print('VERIFY PASS')
if __name__=='__main__':
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    p=argparse.ArgumentParser();p.add_argument('command',choices=['freeze','run','report','verify']);q=p.parse_args()
    if q.command=='freeze':parents();write(ROOT/'FREEZE.json',dict(identity=identity(),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())));print('FROZEN',sha(ROOT/'FREEZE.json'))
    elif q.command=='run':run()
    elif q.command=='report':
        with (ROOT/'REPORT.md').open('x') as f:f.write(report())
    else:verify()
