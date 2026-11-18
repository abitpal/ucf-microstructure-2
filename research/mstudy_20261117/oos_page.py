"""Out-of-sample page: trained fair models re-scored on days never seen in training (incl. today's tape)."""
import json, os, glob, datetime as dt
from curves_page import chart, CHART_CSS, CHART_JS

S = os.path.dirname(os.path.abspath(__file__))
R = f'{S}/results/report/fair_oos'
HZ = ['0.1s', '0.5s', '1s', '10s']
LAST_TRAIN = dt.date(2026, 9, 21)
NAMES = {'LightGBM (handcrafted)': 'LightGBM (handcrafted)', 'transformer_cont': 'Transformer · 75 + 90 min',
         'transformer_long': 'Transformer · 75 min', 'gru_long': 'GRU · 60 min', 'gru': 'GRU · 30 min',
         'transformer': 'Transformer · 25 min'}
ORDER = {'BTC15M': ['LightGBM (handcrafted)', 'transformer_cont', 'gru_long', 'gru', 'transformer_long'],
         'BTCPERP': ['LightGBM (handcrafted)', 'gru', 'transformer']}
LABEL = {'BTC15M': 'BTC 15-minute binaries', 'BTCPERP': 'BTC perp'}


def rows(market):
    out = [json.load(open(f)) for f in glob.glob(f'{R}/{market}_20*.json')]
    return sorted(out, key=lambda r: r['day'])


def rel(day):
    return (dt.date.fromisoformat(day) - LAST_TRAIN).days


def f4(v):
    return f'{v:.4f}'


def body(narr_html=''):
    summ = json.load(open(f'{R}/summary.json'))
    h = narr_html
    for market in ('BTC15M', 'BTCPERP'):
        rs = rows(market)
        if not rs:
            continue
        models = [m for m in ORDER[market] if m in rs[0]['res']]
        pooled = summ[market]['pooled_oos']
        h += f'<h2>{LABEL[market]}</h2>'
        # pooled table
        trs = ''.join(f'<tr><td>{NAMES[m]}</td>' + ''.join(f'<td>{f4(x)}</td>' for x in pooled[m]['r2'])
                      + ''.join(f'<td>{x:.3f}</td>' for x in pooled[m]['dir_acc'][:3])
                      + f'<td>{pooled[m]["days_beating_lgbm"] if m != "LightGBM (handcrafted)" else "–"}</td></tr>' for m in models)
        nd = summ[market]['n_oos_days']
        h += (f'<h3>Pooled over {nd} unseen days</h3><div class="tbl"><table><thead><tr><th>model</th>'
              + ''.join(f'<th>R² {x}</th>' for x in HZ) + ''.join(f'<th>dir. acc {x}</th>' for x in HZ[:3])
              + f'<th>days beating LightGBM at 0.1s (of {nd})</th></tr></thead><tbody>{trs}</tbody></table></div>')
        spec = dict(title=f'{LABEL[market]}: R² by day', xlabel='days after the last training day (Sep 21)', ylabel='R²', hz=True,
                    series=[dict(name=NAMES[m], pts=[dict(x=rel(r['day']), y=r['res'][m]['r2']) for r in rs]) for m in models])
        h += chart(f'oos-{market}', f'{LABEL[market]}: test R² on each day',
                   'x = days after the last training day. Days +1 and +2 (Sep 22, Sep 23) were used for early stopping and the original test; '
                   'every other day was never seen. The far-right point is today’s tape (Nov 17, 57 days later). Pick a horizon above the chart.', spec)
        # per-day table
        th = '<tr><th>day</th><th>status</th><th>samples</th><th>markets</th>' + ''.join(f'<th>{NAMES[m]} · 0.1s</th>' for m in models) + ''.join(f'<th>{NAMES[m]} · 1s</th>' for m in models) + '</tr>'
        tb = ''
        for r in rs:
            st = 'today' if r['day'] == '2026-11-17' else ('val' if r['day'] == '2026-09-22' else ('test' if r['day'] == '2026-09-23' else 'unseen'))
            tb += (f'<tr><td>{r["day"]}</td><td>{st}</td><td>{r["n"]:,}</td><td>{r["n_markets"]}</td>'
                   + ''.join(f'<td>{f4(r["res"][m]["r2"][0])}</td>' for m in models)
                   + ''.join(f'<td>{f4(r["res"][m]["r2"][2])}</td>' for m in models) + '</tr>')
        h += f'<details class="tv"><summary>Per-day table ({len(rs)} days)</summary><div class="tbl"><table><thead>{th}</thead><tbody>{tb}</tbody></table></div></details>'
    return h, CHART_CSS, CHART_JS, json.dumps({'horizons': HZ})
