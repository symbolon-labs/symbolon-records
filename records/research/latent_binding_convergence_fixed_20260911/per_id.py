"""Post-outcome descriptive saved-score analysis; no selection or new gate."""
import argparse
import common as h
from common import ROOT,read,load,b


def analyze():
    h.check();x=read(ROOT/'inputs.json');parent=read(h.ALIGN/'inputs.json');tokens=x['candidate_tokens'];result={}
    for label in ['reference','trained']:
        result[label]={}
        for arm,target in [('correct','original'),('wrong_binding','changed')]:
            path=ROOT/'run'/f'{label}_novel'/f'{arm}.pt'
            if not path.exists():continue
            out=load(path)['first'];part=out['token_lp'].double().numpy();rows=[];prefixes={}
            for j,t in enumerate(tokens):prefixes.setdefault(t[0],[]).append(j)
            for i,t in enumerate(x['banks']['novel']['targets'][target]):
                prefix=max(prefixes,key=lambda k:float(part[i,prefixes[k][0],0]));group=prefixes[tokens[t][0]];digit=max(group,key=lambda j:float(part[i,j,1]))
                gold=float(part[i,t].sum());other=part[i].sum(-1);other=other.copy();other[t]=-float('inf')
                rows.append(dict(id=b.IDS[t],prediction=b.IDS[int(out['prediction'][i])],color=x['banks']['novel']['rows'][i]['color'],
                    first_prefix_correct=prefix==tokens[t][0],gold_digit_correct=digit==t,margin=gold-float(other.max())))
            result[label][arm]=dict(first_prefix_correct=sum(r['first_prefix_correct'] for r in rows),
                gold_digit_correct=sum(r['gold_digit_correct'] for r in rows),
                mean_margin=sum(r['margin'] for r in rows)/len(rows),
                per_id={name:[r['prediction'] for r in rows if r['id']==name] for name in sorted({r['id'] for r in rows})},
                per_id_margin={name:sum(r['margin'] for r in rows if r['id']==name)/max(1,sum(1 for r in rows if r['id']==name)) for name in sorted({r['id'] for r in rows})},
                rows=rows)
    result['continuation']=dict(start_step=h.STEPS,total_steps=h.TOTAL,history=load(ROOT/'run/fit.pt')['history'][-1],
        parent_history=load(h.ALIGN/'run/fit.pt')['history'][-1])
    result['candidate_sha']=h.sha(ROOT/'run/candidate.pt');result['parent_candidate_sha']=h.sha(h.ALIGN/'run/candidate.pt')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--save',action='store_true');args=parser.parse_args();result=analyze()
    if args.save:h.write(ROOT/'PER_ID.json',result)
    else:assert result==read(ROOT/'PER_ID.json')
    print({k:{a:v['mean_margin'] for a,v in arms.items()} for k,arms in result.items() if k in ['reference','trained']})
