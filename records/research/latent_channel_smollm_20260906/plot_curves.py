"""Render saved evaluations only; no smoothing, inference, selection or tuning."""
import argparse
import csv
import io
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser(); p.add_argument('mode',choices=['tiny','full']); args=p.parse_args()
    root=ROOT/f'run_{args.mode}'
    assert (root/'decision.json').exists(), 'Wait for completed training'
    metrics=[(int(p.parent.name.split('_')[1]),json.loads(p.read_text()))
             for p in sorted(root.glob('eval_*/metrics.json'))]
    series=['correct','shuffled','random_71001','random_71002','receiver_only','zero']
    colors=['#146b3a','#b94d12','#4258ad','#9462aa','#555555','#999999']
    right=max(step for step,_ in metrics)
    title='Tiny training-set accuracy (not held out)' if args.mode=='tiny' else 'Reusable DEV accuracy (not FINAL)'
    svg=['<svg xmlns="http://www.w3.org/2000/svg" width="800" height="450" viewBox="0 0 800 450">',
         '<rect width="800" height="450" fill="white"/>',
         f'<text x="60" y="28" font-family="sans-serif" font-size="18">{title}</text>']
    for percent in [0,25,50,75,100]:
        y=350-percent*2.8
        svg += [f'<line x1="60" x2="620" y1="{y}" y2="{y}" stroke="#dddddd"/>',
                f'<text x="15" y="{y+5}" font-family="sans-serif" font-size="12">{percent}%</text>']
    for index,(name,color) in enumerate(zip(series,colors)):
        points=' '.join(f'{60+step/right*560:.2f},{350-m["conditions"][name]["accuracy"]*280:.2f}' for step,m in metrics)
        dash=' stroke-dasharray="5 5"' if name in ['receiver_only','zero'] else ''
        svg += [f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"{dash}/>',
                f'<text x="635" y="{90+index*25}" fill="{color}" font-family="sans-serif" font-size="12">{name}</text>']
    svg += [f'<text x="60" y="375" font-family="sans-serif" font-size="12">0</text>',
            f'<text x="590" y="375" font-family="sans-serif" font-size="12">{right}</text>',
            '<text x="285" y="400" font-family="sans-serif" font-size="13">Optimizer step</text>',
            '<text x="60" y="430" font-family="sans-serif" font-size="12">Lines join recorded evaluations; no intermediate accuracies inferred.</text></svg>']
    with (root/'accuracy.svg').open('x') as f: f.write('\n'.join(svg)+'\n')
    output=io.StringIO(); writer=csv.writer(output); writer.writerow(['step',*series,'correct_full_vocab_ce'])
    for step,m in metrics: writer.writerow([step,*[m['conditions'][s]['accuracy'] for s in series],m['conditions']['correct']['full_vocab_ce']])
    with (root/'evaluation_curve.csv').open('x') as f: f.write(output.getvalue())


if __name__=='__main__': main()
