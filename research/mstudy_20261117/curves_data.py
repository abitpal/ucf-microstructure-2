"""Assemble training-curve data: fair-model validation evals from the VM training log (complete),
other curves from wandb histories (curves.json, pulled via the wandb API)."""
import json, re, os
S = os.path.dirname(os.path.abspath(__file__))
HZ = ['0.1s', '0.5s', '1s', '10s']

pat = re.compile(r'\[(\d\d):(\d\d):(\d\d)\]\s+(\w+) (?:step (\d+) ep (\d+) )?(?:resumed )?val R2 \[([^\]]+)\](?: (\d+)s)?')
evals = []
for line in open(f'{S}/fair_evals.txt'):
    m = pat.search(line)
    if not m:
        continue
    hh, mm, ss, kind, step, ep, vals, secs = m.groups()
    t = int(hh) * 60 + int(mm)
    v = [float(x) for x in vals.split()]
    evals.append(dict(t=t, kind=kind, step=int(step or 0), r2=v, secs=int(secs or 0)))

# which market: BTC-perp runs ran 10:47-11:34 UTC under the plain names
def market(e):
    return 'BTCPERP' if e['kind'] in ('transformer', 'gru') and 10 * 60 + 47 <= e['t'] <= 11 * 60 + 35 else 'BTC15M'

BS = {'transformer': 256, 'transformer_long': 256, 'gru': 256, 'gru_long': 256, 'transformer_cont': 512}
LABEL = {'transformer': 'Transformer · 40 min', 'transformer_long': 'Transformer · 75 min + continued',
         'transformer_cont': 'Transformer · 75 min + continued', 'gru': 'GRU · 30 min', 'gru_long': 'GRU · 60 min'}
series = {'BTC15M': {}, 'BTCPERP': {}}
cont_offset = 4237 * 256  # samples seen by transformer_long, where the continuation resumed
for e in evals:
    mk = market(e)
    name = LABEL[e['kind']] if mk == 'BTC15M' else ('Transformer · 25 min' if e['kind'] == 'transformer' else 'GRU · 20 min')
    x = e['step'] * BS[e['kind']]
    if e['kind'] == 'transformer_cont':
        x = cont_offset + e['step'] * 512
    pt = dict(x=round(x / 1e6, 4), r2=e['r2'], secs=e['secs'], resumed=(e['kind'] == 'transformer_cont' and e['step'] == 0))
    if e['kind'] == 'transformer_cont' and e['step'] == 0:
        continue  # same weights as the last transformer_long point; marked there instead
    series[mk].setdefault(name, []).append(pt)
for mk in series:
    for k in series[mk]:
        series[mk][k].sort(key=lambda p: p['x'])
# mark resume point
if any(e['kind'] == 'transformer_cont' for e in evals):
    s = series['BTC15M'].get('Transformer · 75 min + continued', [])
    for p in s:
        if abs(p['x'] - cont_offset / 1e6) < 1e-6:
            p['resume'] = True

# test R² endpoints for reference
tests = {}
for f in os.listdir(f'{S}/results/report/fair'):
    d = json.load(open(f'{S}/results/report/fair/{f}'))
    for k, v in d['r2'].items():
        tests[f"{d['market']}|{k}"] = v

C = json.load(open(f'{S}/curves.json'))
other = {}
ae = [r for r in C['seq-autoencoder']['rows'] if 'ae/val_mse' in r]
aes = [r for r in C['seq-autoencoder_strict']['rows'] if 'ae/val_mse' in r]
other['ae'] = {'Autoencoder': [dict(x=i + 1, y=r['ae/val_mse']) for i, r in enumerate(ae)],
               'Autoencoder · strict (no tick fields)': [dict(x=i + 1, y=r['ae/val_mse']) for i, r in enumerate(aes)]}
sc = [r for r in C['seq-autoencoder']['rows'] if 'seqclf/val_bal_acc' in r]
mlp = [r for r in C['market-type-classifier']['rows'] if 'mlp/val_bal_acc' in r]
other['clf'] = {'GRU on 64 raw events': [dict(x=i + 1, y=r['seqclf/val_bal_acc']) for i, r in enumerate(sc)],
                'MLP on N=256 window features': [dict(x=i + 1, y=r['mlp/val_bal_acc']) for i, r in enumerate(mlp)]}
tr = {}
for r in C['trade-toxicity-pnl']['rows']:
    for k, v in r.items():
        if k.endswith('/val_tox_auc_mean'):
            tr.setdefault(k.split('/')[0].replace('_', ' '), []).append(v)
other['trade'] = {k: [dict(x=i + 1, y=v) for i, v in enumerate(vals)] for k, vals in tr.items()}
out = dict(fair=series, tests=tests, other=other, horizons=HZ)
json.dump(out, open(f'{S}/curves_data.json', 'w'))
for mk in series:
    for k, v in series[mk].items():
        print(mk, k, [(p['x'], p['r2'][0]) for p in v])
print({k: {kk: len(vv) for kk, vv in v.items()} for k, v in other.items()})
