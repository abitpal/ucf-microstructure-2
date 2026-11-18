"""Aggregate report/fair_oos/*.json -> summary.json, per-day plot, wandb run (offline)."""
import sys, os, json, glob
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from common import *
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

OUT = f'{ROOT}/report/fair_oos'
HZ = ['0.1s', '0.5s', '1s', '10s']
COL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#888888']
rows = [json.load(open(f)) for f in sorted(glob.glob(f'{OUT}/*_20*.json'))]
summ = {}
for market in ('BTC15M', 'BTCPERP'):
    R = sorted([r for r in rows if r['market'] == market], key=lambda r: r['day'])
    if not R:
        continue
    oos = [r for r in R if not r['in_training']]
    models = [m for m in R[0]['res'] if m != 'zero']
    pooled = {}
    for m in models + ['zero']:
        # n-weighted mean of per-day R2 and directional acc on unseen days
        w = np.array([r['n'] for r in oos], float)
        if len(oos):
            pooled[m] = dict(r2=(np.array([r['res'][m]['r2'] for r in oos]) * w[:, None]).sum(0).__truediv__(w.sum()).tolist(),
                             dir_acc=(np.array([r['res'][m]['dir_acc'] for r in oos]) * w[:, None]).sum(0).__truediv__(w.sum()).tolist(),
                             days_beating_lgbm=int(sum(r['res'][m]['r2'][0] > r['res']['LightGBM (handcrafted)']['r2'][0] for r in oos)))
    summ[market] = dict(days=[r['day'] for r in R], n_oos_days=len(oos), pooled_oos=pooled,
                        per_day={r['day']: {m: r['res'][m]['r2'] for m in models} | {'n': r['n'], 'in_training': r['in_training']} for r in R})
    fig, axs = plt.subplots(1, 4, figsize=(20, 4.2), sharex=True)
    for k, ax in enumerate(axs):
        x = np.arange(len(R))
        for i, m in enumerate(models):
            ax.plot(x, [r['res'][m]['r2'][k] for r in R], '-o', ms=4, lw=1.6, color=COL[i], label=m)
        for i, r in enumerate(R):
            if r['in_training']:
                ax.axvspan(i - .5, i + .5, color='#dddddd', alpha=.6, lw=0)
        ax.axhline(0, color='#999', lw=.8)
        ax.set_title(f'{market} test R² · {HZ[k]}'); ax.set_xticks(x); ax.set_xticklabels([r['day'][5:] for r in R], rotation=70, fontsize=7)
        ax.grid(alpha=.25)
    axs[0].legend(fontsize=7)
    fig.suptitle(f'{market}: per-day R² on unseen days (grey = 09-22 val / 09-23 test, seen during model selection)')
    fig.tight_layout(); fig.savefig(f'{OUT}/per_day_{market}.png', dpi=110); plt.close(fig)
json.dump(summ, open(f'{OUT}/summary.json', 'w'), indent=1)

run = wandb_init('fair-out-of-sample', 'fair-oos', job_type='eval')
import wandb
for market, s in summ.items():
    cols = ['day', 'in_training', 'n'] + [f'{m}|{h}' for m in s['per_day'][s['days'][0]] if m not in ('n', 'in_training') for h in HZ]
    t = wandb.Table(columns=cols)
    for d in s['days']:
        pd_ = s['per_day'][d]
        t.add_data(d, pd_['in_training'], pd_['n'], *[v for m in pd_ if m not in ('n', 'in_training') for v in pd_[m]])
    run.log({f'{market}/per_day': t, f'{market}/per_day_plot': wandb.Image(f'{OUT}/per_day_{market}.png')})
    for m, v in s['pooled_oos'].items():
        run.summary.update({f'{market}/oos_r2/{m}/{h}': x for h, x in zip(HZ, v['r2'])})
run.finish()
for market, s in summ.items():
    print(market, s['n_oos_days'], 'oos days')
    for m, v in s['pooled_oos'].items():
        print(f'  {m:28s}', np.round(v['r2'], 4), 'dir', np.round(v['dir_acc'], 3), 'beats LGBM', v['days_beating_lgbm'])
