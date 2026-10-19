"""Generate an SVG from all completed tiny evaluations; no fitted smoothing."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    rows = []
    for path in sorted((ROOT / 'run_tiny').glob('eval_*/metrics.json')):
        rows.append((int(path.parent.name.split('_')[1]), json.loads(path.read_text())['conditions']))
    assert rows[0][0] == 0 and rows[-1][0] == 300
    pieces = ['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="680" viewBox="0 0 1000 680">',
              '<rect width="1000" height="680" fill="white"/>',
              '<g font-family="sans-serif" font-size="14" fill="#18212b">',
              '<text x="70" y="30" font-size="20">Binding V1: complete tiny-training trajectory (32 queries)</text>',
              '<text x="70" y="54">Training-set diagnostics only — no held-out data, no smoothed or omitted checkpoints</text>']

    def panel(top, height, maximum, title, ticks):
        pieces.append(f'<text x="70" y="{top-12}">{title}</text>')
        for value in ticks:
            y = top + height * (1-value/maximum)
            pieces.append(f'<path d="M70 {y:.3f} H950" stroke="#e1e5e9"/>')
            pieces.append(f'<text x="60" y="{y+5:.3f}" text-anchor="end">{value:g}</text>')
        for step in range(0, 301, 50):
            x = 70 + step/300*880
            pieces.append(f'<text x="{x:.3f}" y="{top+height+22}" text-anchor="middle">{step}</text>')
        pieces.append(f'<path d="M70 {top} V{top+height} H950" fill="none" stroke="#18212b"/>')

    def series(top, height, maximum, values, color, dashed=False):
        pts = [(70+step/300*880, top+height*(1-value/maximum)) for step, value in values]
        points = ' '.join(f'{x:.3f},{y:.3f}' for x, y in pts)
        dash = ' stroke-dasharray="6 4"' if dashed else ''
        pieces.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"{dash}/>')
        for x, y in pts:
            pieces.append(f'<circle cx="{x:.3f}" cy="{y:.3f}" r="3" fill="{color}"/>')

    panel(100, 220, 100, 'Exact accuracy / donor following (%)', [0, 25, 50, 75, 100])
    series(100, 220, 100, [(s, 100*c['correct']['accuracy']) for s, c in rows], '#0969da')
    series(100, 220, 100, [(s, 100*c['wrong_binding']['donor_following']) for s, c in rows], '#cf222e')
    series(100, 220, 100, [(s, 100*c['receiver_only']['accuracy']) for s, c in rows], '#6e7781', True)
    threshold_y = 100+220*.05
    pieces.append(f'<path d="M70 {threshold_y} H950" stroke="#1a7f37" stroke-dasharray="5 5"/>')
    pieces.append(f'<text x="940" y="{threshold_y-5}" text-anchor="end" fill="#1a7f37">95% tiny gate</text>')
    pieces.extend(['<text x="70" y="365" fill="#0969da">Correct / shuffled donor following</text>',
                   '<text x="390" y="365" fill="#cf222e">Same-ID counterfactual following</text>',
                   '<text x="720" y="365" fill="#6e7781">Receiver-only</text>'])
    assert all(c['correct']['accuracy'] == c['shuffled']['donor_following'] for _, c in rows)
    panel(410, 200, 18, 'Mean full-ID negative log likelihood (sum of both token losses)', [0, 3, 6, 9, 12, 15, 18])
    series(410, 200, 18, [(s, c['correct']['mean_id_nll']) for s, c in rows], '#0969da')
    pieces.extend(['<text x="510" y="661" text-anchor="middle">Optimizer step</text>', '</g></svg>'])
    with (ROOT / 'training_curves.svg').open('x') as stream:
        stream.write('\n'.join(pieces)+'\n')


if __name__ == '__main__':
    main()
