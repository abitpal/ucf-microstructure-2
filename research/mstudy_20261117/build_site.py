"""Build the multi-page results site from pulled results (stdlib only)."""
import json, os, shutil, glob, html, datetime as dt

S = os.path.dirname(os.path.abspath(__file__))
R = f'{S}/results/report'
OUT = f'{S}/site'
NARR = json.load(open(f'{S}/narr.json')) if os.path.exists(f'{S}/narr.json') else {}
OPS = json.load(open(f'{S}/ops.json')) if os.path.exists(f'{S}/ops.json') else {}

PAGES = [
    ('index.html', 'Overview'),
    ('classical.html', 'Classical microstructure'),
    ('classify.html', 'Market-type classifiers'),
    ('autoencoder.html', 'Autoencoder embedding'),
    ('trades.html', 'Toxic vs retail flow'),
    ('fair.html', 'Raw-event fair models'),
    ('curves.html', 'Training curves'),
    ('oos.html', 'Out-of-sample days'),
    ('ops.html', 'Run log & data'),
]

CSS = r"""
/* layout: lab notebook. sticky section rail on the left, one reading column, figures on white plates */
:root{
  --paper:#f5f6f8; --surface:#ffffff; --ink:#18202b; --muted:#5b6676; --line:#dde2ea; --accent:#0f6e78;
  --crypto:#2a6fdb; --sports:#d06a2a; --perp:#2f9472; --good:#2f8a4e; --warn:#b7791f; --bad:#b83a3a;
  --plate:#ffffff; --code:#eef1f5;
  --f-display:'Space Grotesk', 'Helvetica Neue', Arial, sans-serif;
  --f-body:'IBM Plex Sans', 'Helvetica Neue', Arial, sans-serif;
  --f-mono:'IBM Plex Mono', ui-monospace, Menlo, monospace;
}
@media (prefers-color-scheme: dark){ :root:not([data-theme="light"]){
  --paper:#11161d; --surface:#18202a; --ink:#e6eaf0; --muted:#9aa6b5; --line:#2a3441; --accent:#4fb3bd;
  --crypto:#6d9ef0; --sports:#ec9a63; --perp:#5cc29d; --good:#5cb97a; --warn:#d9a441; --bad:#e06b6b;
  --plate:#f4f5f7; --code:#1f2833; color-scheme:dark; } }
:root[data-theme="dark"]{
  --paper:#11161d; --surface:#18202a; --ink:#e6eaf0; --muted:#9aa6b5; --line:#2a3441; --accent:#4fb3bd;
  --crypto:#6d9ef0; --sports:#ec9a63; --perp:#5cc29d; --good:#5cb97a; --warn:#d9a441; --bad:#e06b6b;
  --plate:#f4f5f7; --code:#1f2833; color-scheme:dark; }
*{box-sizing:border-box}
body{background:var(--paper);color:var(--ink);font:15px/1.6 var(--f-body);
  padding-left:max(16px, env(safe-area-inset-left, 0px));padding-right:max(16px, env(safe-area-inset-right, 0px));}
.wrap{max-width:1180px;margin:0 auto;display:grid;grid-template-columns:210px minmax(0,1fr);gap:40px;padding-block:28px 80px}
nav{position:sticky;top:env(safe-area-inset-top, 0px);align-self:start;padding-top:6px}
nav .brand{font:600 13px/1.3 var(--f-mono);letter-spacing:.06em;text-transform:uppercase;color:var(--muted);margin-bottom:14px}
nav a{display:block;color:var(--muted);text-decoration:none;padding:6px 10px;border-radius:6px;font-size:14px}
nav a:hover{color:var(--ink);background:var(--surface)}
nav a.on{color:var(--ink);background:var(--surface);box-shadow:inset 3px 0 0 var(--accent)}
nav a:focus-visible, a:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
main{min-width:0}
h1{font:600 32px/1.15 var(--f-display);margin:0 0 6px;text-wrap:balance;letter-spacing:-.01em}
h2{font:600 21px/1.25 var(--f-display);margin:44px 0 10px;text-wrap:balance}
h3{font:600 16px/1.3 var(--f-display);margin:26px 0 6px}
p,li{max-width:72ch}
.lede{color:var(--muted);font-size:16px;margin:0 0 22px;max-width:72ch}
.eyebrow{font:500 12px/1 var(--f-mono);letter-spacing:.08em;text-transform:uppercase;color:var(--accent);margin-bottom:10px}
.answer{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:16px 20px;margin:18px 0 8px}
.answer h3{margin-top:0}
.answer ul{margin:6px 0 0;padding-left:20px}
figure{margin:18px 0 26px}
figure .plate{background:var(--plate);border:1px solid var(--line);border-radius:8px;padding:10px;overflow-x:auto}
figure img{display:block;max-width:100%;height:auto;margin:0 auto}
figcaption{color:var(--muted);font-size:13px;margin-top:8px;max-width:80ch}
.tbl{overflow-x:auto;margin:12px 0 22px;border:1px solid var(--line);border-radius:8px;background:var(--surface)}
table{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}
th,td{padding:7px 10px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left}
th{font:600 12px/1.3 var(--f-mono);color:var(--muted);letter-spacing:.02em;background:var(--code)}
tr:last-child td{border-bottom:0}
code,.mono{font-family:var(--f-mono);font-size:13px;background:var(--code);padding:1px 5px;border-radius:4px}
.pill{display:inline-block;font:600 11px/1 var(--f-mono);padding:4px 8px;border-radius:999px;letter-spacing:.04em;text-transform:uppercase}
.pill.done{background:color-mix(in srgb,var(--good) 18%,transparent);color:var(--good)}
.pill.run{background:color-mix(in srgb,var(--warn) 20%,transparent);color:var(--warn)}
.pill.fail{background:color-mix(in srgb,var(--bad) 18%,transparent);color:var(--bad)}
.pill.wait{background:var(--code);color:var(--muted)}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:18px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin:18px 0}
.kpi{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.kpi .v{font:600 26px/1.1 var(--f-display);font-variant-numeric:tabular-nums}
.kpi .l{color:var(--muted);font-size:13px;margin-top:4px}
.log{font:13px/1.55 var(--f-mono);background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:12px 14px;overflow-x:auto;white-space:pre-wrap}
.c-crypto{color:var(--crypto)} .c-sports{color:var(--sports)} .c-perp{color:var(--perp)}
.note{color:var(--muted);font-size:13px}
.pending{border:1px dashed var(--line);border-radius:10px;padding:18px;color:var(--muted);background:var(--surface)}
footer{color:var(--muted);font-size:12px;margin-top:60px;border-top:1px solid var(--line);padding-top:14px}
@media (max-width:820px){ .wrap{grid-template-columns:1fr;gap:12px} nav{position:static;display:flex;flex-wrap:wrap;gap:4px}
  nav .brand{width:100%} h1{font-size:26px} }
@media (prefers-reduced-motion: reduce){ *{transition:none!important;animation:none!important} }
"""

FONTS = ('<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
         '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&family=Space+Grotesk:wght@500;600&display=swap">')


def esc(x):
    return html.escape(str(x))


def load(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def fmt(v, d=3):
    if v is None:
        return '–'
    try:
        if isinstance(v, str):
            return esc(v)
        if v != v:
            return '–'
        if isinstance(v, int):
            return f'{v:,}'
        return f'{v:.{d}f}'
    except Exception:
        return esc(v)


def table(rows, cols, heads=None, d=3):
    heads = heads or cols
    h = ''.join(f'<th>{esc(c)}</th>' for c in heads)
    body = ''
    for r in rows:
        body += '<tr>' + ''.join(f'<td>{fmt(r.get(c), d) if not isinstance(r.get(c), str) else esc(r.get(c))}</td>' for c in cols) + '</tr>'
    return f'<div class="tbl"><table><thead><tr>{h}</tr></thead><tbody>{body}</tbody></table></div>'


def fig(sub, name, cap):
    src = f'{R}/{sub}/{name}'
    if not os.path.exists(src):
        return f'<div class="pending">Figure <code>{esc(name)}</code> is not produced yet.</div>'
    os.makedirs(f'{OUT}/img/{sub}', exist_ok=True)
    shutil.copy(src, f'{OUT}/img/{sub}/{name}')
    return (f'<figure><div class="plate"><img src="img/{sub}/{name}" alt="{esc(cap)}" loading="lazy"></div>'
            f'<figcaption>{cap}</figcaption></figure>')


def narr(key, default=''):
    v = NARR.get(key)
    if not v:
        return default
    if isinstance(v, list):
        return '<ul>' + ''.join(f'<li>{x}</li>' for x in v) + '</ul>'
    return f'<p>{v}</p>'


def answer(key, title='What we found'):
    body = narr(key)
    if not body:
        return '<div class="pending">Results for this section are still being computed. This page refreshes as stages finish.</div>'
    return f'<div class="answer"><h3>{esc(title)}</h3>{body}</div>'


def page(fname, title, lede, body, eyebrow, extra_css='', extra_js=''):
    nav = ''.join(f'<a href="{f}" class="{"on" if f == fname else ""}">{esc(t)}</a>' for f, t in PAGES)
    upd = dt.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')
    doc = f"""<title>{'Microstructure Night Run' if fname == 'index.html' else esc(title) + ' · Microstructure Night Run'}</title>
<meta name="claude:viewport-fit-sides" content="cover">
{FONTS}
<style>{CSS}{extra_css}</style>
<div class="wrap">
<nav><div class="brand">Kalshi microstructure<br>study · Nov 17</div>{nav}</nav>
<main>
<div class="eyebrow">{esc(eyebrow)}</div>
<h1>{esc(title)}</h1>
<p class="lede">{lede}</p>
{body}
<footer>Built from the run on <span class="mono">kalshi-analysis</span>. Last rebuilt {upd}. wandb project: <span class="mono">kalshi-microstructure</span>.</footer>
</main></div>{extra_js}"""
    if fname != 'index.html':
        doc = '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">' \
              '<style>:root{padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)} body{margin:0} img{max-width:100%}</style></head><body>' + doc + '</body></html>'
    open(f'{OUT}/{fname}', 'w').write(doc)


def build():
    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT)
    C = load(f'{R}/classical/summary.json')
    K = load(f'{R}/classify/summary.json')
    A = load(f'{R}/ae/summary.json'); AS = load(f'{R}/ae_strict/summary.json')
    T = load(f'{R}/trade/summary.json')
    FA = {os.path.basename(p)[:-5]: load(p) for p in glob.glob(f'{R}/fair/*.json')}

    # ---------------- overview
    status = OPS.get('status', [])
    st_rows = ''.join(f'<tr><td>{esc(s["task"])}</td><td><span class="pill {esc(s["state"])}">{esc(s["label"])}</span></td><td style="text-align:left;white-space:normal">{s.get("note","")}</td></tr>' for s in status)
    kp = NARR.get('kpis', [])
    kpis = ''.join(f'<div class="kpi"><div class="v">{esc(k[0])}</div><div class="l">{esc(k[1])}</div></div>' for k in kp)
    body = (answer('overview', 'Headline answers')
            + (f'<div class="kpis">{kpis}</div>' if kpis else '')
            + '<h2>Status of every task</h2><div class="tbl"><table><thead><tr><th>Task</th><th>State</th><th style="text-align:left">Notes</th></tr></thead>'
            + f'<tbody>{st_rows}</tbody></table></div>'
            + narr('overview_next', '')
            + '<h2>How to read these pages</h2>'
            + '<p>Every number comes from three Kalshi market families recorded by the existing tape: <b class="c-crypto">crypto binaries</b> '
              '(BTC 15-minute, alt 15-minute, hourly and range strikes), <b class="c-sports">sports binaries</b> (NFL, NCAAF, MLB, soccer, WNBA, tennis) '
              'and <b class="c-perp">perpetual futures</b> (BTC, ETH and alt perps). The colours are used the same way in every figure. '
              'Tests are always on markets the models never saw in training.</p>')
    page('index.html', 'Microstructure Night Run', 'Does the order book look different from market to market, can the next taker order be judged as toxic or benign, and do raw-event sequence models beat handcrafted features at predicting the fair price?', body, 'Overview')

    # ---------------- classical
    b = answer('classical')
    if C:
        b += '<h2>Data in this comparison</h2>'
        rows = [{'class': k, 'windows': v, 'markets': C['n_markets'].get(k)} for k, v in C['by_fine'].items()]
        b += table(rows, ['class', 'windows', 'markets'], ['market class', 'sampled windows', 'distinct market-days'])
    b += '<h2>Fingerprint</h2>' + fig('classical', 'fingerprint_heatmap.png', 'Class medians of every scale-free window feature (last 256 events), z-scored across classes. Red is above the cross-class average, blue below.')
    b += fig('classical', 'feature_boxes.png', 'Distributions of key features per market (box = interquartile range).')
    b += fig('classical', 'effect_sizes.png', "Cohen's d between coarse classes. |d| above 0.8 is a large difference.")
    b += '<h2>Order flow and price dynamics</h2>' + fig('classical', 'stylized_facts.png', 'Left: persistence of trade signs. Middle: bounce in the mid-price. Right: shape of inter-event times after dividing by each market\'s own mean, which removes activity level.')
    b += fig('classical', 'depth_profile.png', 'Share of the top-5 book depth that sits at each level.')
    b += fig('classical', 'imbalance_response.png', 'Average next-second mid move against L1 order-book imbalance.')
    b += '<h2>What happens after a taker order</h2>' + fig('classical', 'markouts.png', 'Left: mid move after a taker order in units of half-spread. Middle: share of orders that end up past the fill price (toxic for the maker). Right: share of round trips that clear two taker fees and the exit spread.')
    b += fig('classical', 'impact_vs_size.png', 'Impact and toxicity against order size relative to displayed depth at the touch.')
    if C and C.get('trade'):
        tr = C['trade']
        rows = []
        for k in tr['p_toxic']:
            rows.append({'class': k, 'orders': tr['n_orders'].get(k), 'spread': tr['median_spread_ticks'].get(k),
                         'fee': tr['median_fee_ticks_per_side'].get(k),
                         'tox1': tr['p_toxic'][k].get('y_mk_1p0'), 'tox10': tr['p_toxic'][k].get('y_mk_10p0'),
                         'net1': tr['p_net_profitable'][k].get('y_net_1p0'), 'net10': tr['p_net_profitable'][k].get('y_net_10p0')})
        b += table(rows, ['class', 'orders', 'spread', 'fee', 'tox1', 'tox10', 'net1', 'net10'],
                   ['class', 'taker orders', 'median spread (ticks)', 'fee/side (ticks)', 'P(toxic) 1s', 'P(toxic) 10s', 'P(net>0) 1s', 'P(net>0) 10s'])
    b += '<h2>Method</h2>' + narr('classical_method', '')
    page('classical.html', 'Classical microstructure', 'Plain statistics of the order book and order flow, computed with features that do not depend on volume, price level or tick size.', b, 'Analysis · Part 1')

    # ---------------- classify
    b = answer('classify')
    if K:
        res = K['results']
        sweep = [r for r in res if r['tag'].startswith('window_') or r['tag'].startswith('strict_')]
        rows = {}
        for r in sweep:
            key = (r['window'])
            rows.setdefault(key, {'window': key, 'coverage': r.get('coverage')})
            rows[key][f"{r['features']}_{r['label']}"] = r['bal_acc']
        b += '<h2>More history, better identification</h2>'
        b += fig('classify', 'window_sweep.png', 'Balanced accuracy on held-out markets against history window. N = last N book events; W = last W seconds (coverage = share of samples with at least 8 events in that time window).')
        b += table(list(rows.values()), ['window', 'coverage', 'nontrivial_coarse', 'strict_coarse', 'nontrivial_fine', 'strict_fine'],
                   ['window', 'coverage', 'coarse · non-trivial', 'coarse · strict', 'fine · non-trivial', 'fine · strict'])
        b += '<h2>Confusion matrices</h2><div class="grid2">' + fig('classify', 'cm_multiwindow_coarse_strict.png', 'Coarse class with strict features.') + fig('classify', 'cm_multiwindow_fine_strict.png', 'Fine class with strict features.') + '</div>'
        b += '<div class="grid2">' + fig('classify', 'cm_multiwindow_coarse.png', 'Coarse class, non-trivial features.') + fig('classify', 'cm_multiwindow_fine.png', 'Fine class, non-trivial features.') + '</div>'
        b += '<h2>Which features carry the signal</h2>' + fig('classify', 'ablations.png', 'Fine-class ablations at N=256: red bars drop one group, blue bars keep only that group.')
        abl = [r for r in res if r['tag'].startswith('N256_') or r['tag'].startswith('N1024_')]
        b += table([{'test': r['tag'], 'bal_acc': r['bal_acc'], 'f1': r['macro_f1']} for r in abl], ['test', 'bal_acc', 'f1'], ['test', 'balanced accuracy', 'macro F1'])
        b += fig('classify', 'importance.png', 'LightGBM gain importance for the multi-window fine classifier.')
        b += '<h2>Within one family</h2><div class="grid2">' + fig('classify', 'cm_within_sports.png', 'Sports only.') + fig('classify', 'cm_within_crypto.png', 'Crypto binaries only.') + fig('classify', 'cm_within_perp.png', 'Perps only.') + '</div>'
        other = [r for r in res if r['tag'].startswith('btc15m_vs') or r['tag'].startswith('within') or 'trivial' in r['tag'] or r['tag'].startswith('mlp') or r['tag'].startswith('multiwindow')]
        b += table([{'test': r['tag'], 'label': r['label'], 'window': r.get('window'), 'features': r.get('features'), 'bal_acc': r['bal_acc'], 'chance': r['chance']} for r in other],
                   ['test', 'label', 'window', 'features', 'bal_acc', 'chance'])
        b += fig('classify', 'cm_mlp_N256_fine.png', 'MLP classifier (its hidden layer is reused by the trade model).')
    b += '<h2>Method</h2>' + narr('classify_method', '')
    page('classify.html', 'Market-type classifiers', 'Can a model tell which kind of market it is looking at from book behaviour alone, after we remove volume, price level and tick size?', b, 'Analysis · Part 1')

    # ---------------- AE
    b = answer('ae')
    if A:
        rows = []
        for k, v in A.items():
            if isinstance(v, dict) and 'bal_acc' in v:
                rows.append({'model': k, 'bal_acc': v['bal_acc'], 'chance': v['chance']})
        if AS:
            for k, v in AS.items():
                if isinstance(v, dict) and 'bal_acc' in v:
                    rows.append({'model': k + ' (strict: tick fields removed)', 'bal_acc': v['bal_acc'], 'chance': v['chance']})
        b += table(rows, ['model', 'bal_acc', 'chance'], ['model', 'balanced accuracy', 'chance'])
        b += fig('ae', 'embedding_pca.png', 'First two principal components of the 16-d embedding for held-out sequences.')
        b += fig('ae', 'latent_class_means.png', 'Average value of each latent dimension per market; each dimension is labelled with its most-correlated handcrafted feature.')
        b += '<div class="grid2">' + fig('ae', 'cm_linear_probe_coarse.png', 'Linear probe, coarse.') + fig('ae', 'cm_gbm_probe_fine.png', 'GBM probe, fine.') + '</div>'
        if AS:
            b += '<div class="grid2">' + fig('ae_strict', 'cm_linear_probe_coarse.png', 'Strict AE, linear probe, coarse.') + fig('ae_strict', 'cm_gbm_probe_fine.png', 'Strict AE, GBM probe, fine.') + '</div>'
        b += fig('ae', 'cm_seq_gru_classifier.png', 'Supervised GRU on the same 64-event sequences (upper bound for this input).')
        it = A.get('latent_interpretation', {})
        b += '<h2>What the latent dimensions encode</h2>' + table([{'dim': k, 'top correlates': ', '.join(f'{a} ({c:+.2f})' for a, c in v[:3])} for k, v in it.items()], ['dim', 'top correlates'])
    b += '<h2>Method</h2>' + narr('ae_method', '')
    page('autoencoder.html', 'Autoencoder embedding', 'An unsupervised 16-number summary of the last 64 book events, and how much market identity it carries.', b, 'Analysis · Part 1')

    # ---------------- trades
    b = answer('trades')
    if T:
        res = T['results']
        rows = []
        for name, r in res.items():
            row = {'model': name}
            for h in ('0p1', '0p5', '1p0', '10p0'):
                row[f'tox {h}'] = r.get(f'tox_auc_{h}'); row[f'net {h}'] = r.get(f'net_auc_{h}')
            rows.append(row)
        hs = ['0p1', '0p5', '1p0', '10p0']
        b += '<h2>How predictable is a taker order?</h2>' + table(rows, ['model'] + [f'tox {h}' for h in hs] + [f'net {h}' for h in hs],
                                                                   ['model / inputs'] + [f'toxic AUC {h.replace("p", ".")}s' for h in hs] + [f'net>0 AUC {h.replace("p", ".")}s' for h in hs])
        b += fig('trade', 'auc_by_featureset.png', 'Test ROC-AUC by horizon for each input set.')
        b += '<h2>Confusion matrices</h2><div class="grid2">' + ''.join(fig('trade', f'cm_toxic_{h}.png', f'Toxic vs benign at {h.replace("p", ".")}s; threshold chosen on validation markets.') for h in hs) + '</div>'
        b += '<div class="grid2">' + ''.join(fig('trade', f'cm_net_{h}.png', f'Profitable after two taker fees at {h.replace("p", ".")}s.') for h in hs) + '</div>'
        full = res.get('MLP | trade+window+AE+clf', {})
        econ = [{'horizon': h.replace('p', '.') + 's', 'base': full.get(f'net_base_{h}'), 'top10': full.get(f'top10_net_hit_{h}'),
                 'top1': full.get(f'top1_net_hit_{h}'), 'all_net': full.get(f'all_mean_net_ticks_{h}'),
                 'top10_net': full.get(f'top10_mean_net_ticks_{h}'), 'top1_net': full.get(f'top1_mean_net_ticks_{h}')} for h in hs]
        b += '<h2>Does it pay to follow the model?</h2>' + table(econ, ['horizon', 'base', 'top10', 'top1', 'all_net', 'top10_net', 'top1_net'],
                                                                 ['horizon', 'P(net>0) all', 'P(net>0) top 10%', 'P(net>0) top 1%', 'mean net (ticks) all', 'top 10%', 'top 1%'])
        pc = T.get('per_coarse', {})
        b += '<h2>By market family</h2>' + table([{'family': k, 'n': v.get('n'), **{f'tox {h}': v.get(f'tox_auc_{h}') for h in hs}, **{f'base {h}': v.get(f'tox_base_{h}') for h in hs}} for k, v in pc.items()],
                                                  ['family', 'n'] + [f'tox {h}' for h in hs] + [f'base {h}' for h in hs])
        pf = T.get('per_fine', {})
        b += table([{'market': k, 'n': v.get('n'), **{f'tox {h}': v.get(f'tox_auc_{h}') for h in hs}} for k, v in pf.items()], ['market', 'n'] + [f'tox {h}' for h in hs])
        b += '<h2>What makes an order toxic</h2>' + fig('trade', 'importance_toxic_1s.png', 'LightGBM gain for toxic-at-1s, by feature and by input block.')
        b += fig('trade', 'size_x_momentum.png', 'Empirical toxicity at 1s by order size relative to touch depth and by the prior 10s move in the order\'s direction.')
    b += '<h2>Method</h2>' + narr('trades_method', '')
    page('trades.html', 'Toxic vs retail flow', 'Given the book state and the incoming taker order, predict whether the order is informed (the price runs past the fill) and whether the taker\'s round trip clears fees.', b, 'Analysis · Part 1')

    # ---------------- fair
    b = answer('fair')
    rows = []
    for key, v in sorted(FA.items()):
        if not v:
            continue
        for name, r2 in v['r2'].items():
            rows.append({'market': v['market'], 'model': name, **{h: x for h, x in zip(['0.1s', '0.5s', '1s', '10s'], r2)}})
    if rows:
        b += '<h2>Out-of-sample R² on a held-out day</h2>' + table(rows, ['market', 'model', '0.1s', '0.5s', '1s', '10s'], d=4)
        extra = []
        for key, v in sorted(FA.items()):
            if v and v.get('stats', {}).get('directional_acc_nonzero'):
                extra.append({'model': key, **v['stats']['directional_acc_nonzero'], 'params': v['stats'].get('nparam'), 'steps': v['stats'].get('steps')})
        if extra:
            b += '<h3>Direction accuracy when the mid actually moves</h3>' + table(extra, ['model', '0.1s', '0.5s', '1s', '10s', 'params', 'steps'])
    b += narr('fair_arch', '')
    b += '<h2>Method</h2>' + narr('fair_method', '')
    page('fair.html', 'Raw-event fair models', 'A Transformer and a GRU read the last 128 raw book events, with time encoded by each event\'s age in seconds, and predict the mid move at four horizons. Baselines use the handcrafted features.', b, 'Predict · Part 2')

    # ---------------- curves
    import curves_page
    cb, ccss, cjs, cdata = curves_page.body()
    page('curves.html', 'Training curves', 'How each model learned: validation scores over the course of training, for the raw-event fair models and the analysis models.', cb, 'Predict · Part 2',
         extra_css=ccss, extra_js=f'<script type="application/json" id="curve-data">{cdata}</script><script>{cjs}</script>')
    b_link = ''

    # ---------------- out-of-sample
    if os.path.exists(f'{R}/fair_oos/summary.json'):
        import oos_page
        ob, ocss, ojs, odata = oos_page.body(answer('oos', 'What held up'))
        ob += '<h2>Method</h2>' + narr('oos_method', '')
        page('oos.html', 'Out-of-sample days', 'The trained fair models, frozen, re-scored on every other day on the tape, including today\'s collection.', ob, 'Predict · Validation',
             extra_css=ocss, extra_js=f'<script type="application/json" id="curve-data">{odata}</script><script>{ojs}</script>')

    # ---------------- ops
    b = narr('ops_intro', '')
    for sec in OPS.get('sections', []):
        b += f'<h2>{esc(sec["title"])}</h2>' + sec['html']
    if OPS.get('timeline'):
        b += '<h2>Timeline</h2><div class="log">' + '\n'.join(esc(x) for x in OPS['timeline']) + '</div>'
    pl = f'{S}/results/logs/pipeline.log'
    if os.path.exists(pl):
        b += '<h2>Pipeline log (VM)</h2><div class="log">' + esc(open(pl).read()[-6000:]) + '</div>'
    page('ops.html', 'Run log & data', 'What ran overnight, what broke, what was changed on the Azure resources, and where everything lives.', b, 'Operations')
    print('built', os.listdir(OUT))


if __name__ == '__main__':
    build()
