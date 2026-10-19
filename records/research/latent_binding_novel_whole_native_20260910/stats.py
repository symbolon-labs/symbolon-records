"""Saved-array statistics; historical DEV seed95002, no inference change."""
import numpy as np
from core import b

def boot(values,rows):
    values=np.asarray(values,dtype=np.float64)
    groups=np.asarray([row['group'] for row in rows])
    assert values.shape==(32,) and [(groups==g).sum() for g in range(8)]==[4]*8
    means=np.array([values[groups==g].mean() for g in range(8)])
    draws=np.random.default_rng(95002).integers(0,8,(10000,8))
    return dict(mean=float(values.mean()),ci95=np.quantile(means[draws].mean(1),[.025,.975]).tolist(),
                positive_fraction=float((values>0).mean()),table_means=means.tolist(),
                seed=95002,draws=10000,descriptive_only=True)

def vectors(out,targets):
    part=out['token_lp'].numpy()
    assert part.shape==(32,128,2) and np.isfinite(part).all()
    lp=part.sum(-1)
    assert np.array_equal(lp,out['lp'].numpy()) and np.array_equal(lp.argmax(-1),out['prediction'].numpy())
    targets=np.asarray(targets)
    gold=lp[np.arange(32),targets].astype(np.float64)
    other=lp.astype(np.float64);other[np.arange(32),targets]=-np.inf
    return (lp.argmax(-1)==targets).astype(float),gold-other.max(-1),gold

def summary(out,targets,rows,tokens):
    hit,margin,gold=vectors(out,targets)
    targets=np.asarray(targets);part=out['token_lp'].numpy();tokens=np.asarray(tokens)
    groups={}
    for i,p in enumerate(tokens[:,0]):groups.setdefault(int(p),[]).append(i)
    reps=[v[0] for v in groups.values()]
    prefix=np.asarray(reps)[part[:,reps,0].argmax(-1)]
    digit=[]
    for i,t in enumerate(targets):
        ix=groups[int(tokens[t,0])]
        digit.append(ix[int(part[i,ix,1].argmax())]==int(t))
    pred=out['prediction'].tolist()
    return dict(correct=int(hit.sum()),accuracy=boot(hit,rows),mean_margin=float(margin.mean()),
        median_margin=float(np.median(margin)),margin_interval=boot(margin,rows),nll=float(-gold.mean()),
        first_prefix_correct=int((tokens[prefix,0]==tokens[targets,0]).sum()),gold_digit_correct=int(sum(digit)),
        per_color={color:int(hit[[i for i,r in enumerate(rows) if r['color']==color]].sum()) for color in b.COLORS},
        per_group=[int(hit[[i for i,r in enumerate(rows) if r['group']==g]].sum()) for g in range(8)],
        per_id={x:int(hit[targets==b.IDS.index(x)].sum()) for x in sorted({b.IDS[t] for t in targets})},
        predictions=[b.IDS[p] for p in pred],target_ids=[b.IDS[t] for t in targets],margins=margin.tolist())

def paired(outputs,targets,rows):
    result={}
    for group,active,controls in [('original','correct',['zero','shuffled','wrong_binding','random_71001','random_71002']),
                                  ('changed','wrong_binding',['zero','correct','random_71001','random_71002'])]:
        hit,margin,_=vectors(outputs[active],targets[group]);comparisons={}
        for name in controls:
            other,other_margin,_=vectors(outputs[name],targets[group])
            comparisons[name]=dict(accuracy=boot(hit-other,rows),margin=boot(margin-other_margin,rows))
        result[group]=comparisons
    return result
