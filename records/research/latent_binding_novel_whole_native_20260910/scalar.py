"""Independent saved-array DEV statistics; seed95002."""
import numpy as np
IDS=[a+str(d) for a in 'ABCDEFGHIJKLMNOP' for d in range(8)]

def independent_scores(out,target,tokens):
    part=out['token_lp'].numpy();assert part.shape==(32,128,2) and np.isfinite(part).all()
    scores=np.array([[np.float32(x+y) for x,y in row] for row in part],dtype=np.float32)
    predictions=[max(range(128),key=lambda j:float(row[j])) for row in scores]
    assert np.array_equal(scores,out['lp'].numpy()) and predictions==out['prediction'].tolist()
    hits=np.array([p==t for p,t in zip(predictions,target,strict=True)],dtype=float)
    margin=np.array([float(scores[i,t])-max(float(scores[i,j]) for j in range(128) if j!=t) for i,t in enumerate(target)])
    gold=np.array([float(scores[i,t]) for i,t in enumerate(target)])
    prefixes=list(dict.fromkeys(tokens[:,0].tolist()));reps=[int(np.flatnonzero(tokens[:,0]==p)[0]) for p in prefixes]
    first=0;digit=0
    for i,t in enumerate(target):
        p=max(reps,key=lambda j:float(part[i,j,0]));first+=int(tokens[p,0]==tokens[t,0])
        siblings=np.flatnonzero(tokens[:,0]==tokens[t,0]).tolist()
        digit+=int(max(siblings,key=lambda j:float(part[i,j,1]))==t)
    return hits,margin,gold,first,digit,predictions

def interval(values,rows):
    values=np.asarray(values,dtype=float)
    tables=np.array([values[[i for i,r in enumerate(rows) if r['group']==g]].mean() for g in range(8)])
    draw=np.random.default_rng(95002).integers(0,8,(10000,8))
    return dict(mean=float(values.mean()),ci95=np.quantile(tables[draw].mean(1),[.025,.975]).tolist(),
        positive_fraction=float((values>0).mean()),table_means=tables.tolist(),seed=95002,draws=10000,descriptive_only=True)

def verify_summary(out,target,tokens,rows,claim):
    hit,margin,gold,first,digit,pred=independent_scores(out,target,tokens)
    assert claim['correct']==int(hit.sum()) and claim['accuracy']==interval(hit,rows)
    assert claim['mean_margin']==float(margin.mean()) and claim['median_margin']==float(np.median(margin))
    assert claim['margin_interval']==interval(margin,rows) and claim['nll']==float(-gold.mean())
    assert claim['first_prefix_correct']==first and claim['gold_digit_correct']==digit
    assert claim['predictions']==[IDS[p] for p in pred] and claim['target_ids']==[IDS[t] for t in target]
    assert claim['margins']==margin.tolist()
    assert claim['per_group']==[int(hit[[i for i,r in enumerate(rows) if r['group']==g]].sum()) for g in range(8)]
    assert claim['per_color']=={color:int(hit[[i for i,r in enumerate(rows) if r['color']==color]].sum()) for color in ['red','green','blue','yellow']}
    assert claim['per_id']=={x:int(hit[[i for i,t in enumerate(target) if IDS[t]==x]].sum()) for x in sorted({IDS[t] for t in target})}
    return hit,margin
