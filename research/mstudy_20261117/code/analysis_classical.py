"""Part 1a: classical microstructure comparison across market classes.
Outputs: REPORT/classical/*.png, summary.json ; wandb run (group=classical)."""
import sys, os, json, glob, time
sys.path.insert(0, '/mnt/scratch/mstudy/pylib'); sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import *
from features import FEATS, GROUPS, prep

REP = f'{ROOT}/report/classical'
os.makedirs(REP, exist_ok=True)
ORDER = ['BTC15M', 'alt15M', 'cryptoHourly', 'cryptoRange', 'NFL', 'NCAAF', 'MLB', 'SOCCER', 'WNBA', 'TENNIS',
         'otherSport', 'BTCPERP', 'ETHPERP', 'altPERP']
CCOL = {'crypto': '#2a6fdb', 'sports': '#e07b39', 'perp': '#3aa17e'}
plt.rcParams.update({'figure.dpi': 110, 'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 9,
                     'axes.grid': True, 'grid.alpha': .25})


def load_win():
    fs = sorted(glob.glob(f'{DATA}/feat/win/*.parquet'))
    return pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)


def load_trades():
    fs = sorted(glob.glob(f'{DATA}/feat/trade/*.parquet'))
    return pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)


def cohen_d(a, b):
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if len(a) < 10 or len(b) < 10:
        return np.nan
    s = np.sqrt((a.var() + b.var()) / 2) + 1e-12
    return (a.mean() - b.mean()) / s


def main():
    run = wandb_init('classical-microstructure', 'classical', job_type='analysis')
    import wandb
    W = load_win()
    W = W[W.fine.isin(ORDER)]
    fines = [f for f in ORDER if f in set(W.fine)]
    summary = {'n_anchors': int(len(W)), 'by_fine': W.fine.value_counts().to_dict(),
               'by_coarse': W.coarse.value_counts().to_dict(),
               'n_markets': W.groupby('fine').ticker.nunique().to_dict()}
    blk = 'N256'
    # ---------------- per-class medians table (event window 256)
    med = W.groupby('fine')[[f'{blk}__{f}' for f in FEATS]].median()
    med.columns = FEATS
    med = med.loc[fines]
    med.to_csv(f'{REP}/medians_{blk}.csv')
    summary['medians_N256'] = med.round(4).to_dict(orient='index')
    # ---------------- effect sizes between coarse classes
    eff = {}
    for a, b in (('crypto', 'sports'), ('crypto', 'perp'), ('sports', 'perp')):
        eff[f'{a}_vs_{b}'] = {f: float(cohen_d(W.loc[W.coarse == a, f'{blk}__{f}'].values,
                                               W.loc[W.coarse == b, f'{blk}__{f}'].values)) for f in FEATS}
    E = pd.DataFrame(eff)
    E.to_csv(f'{REP}/effect_sizes_{blk}.csv')
    summary['effect_sizes_N256'] = E.round(3).to_dict(orient='index')
    # BTC15M vs sports specifically
    eff2 = {f: float(cohen_d(W.loc[W.fine == 'BTC15M', f'{blk}__{f}'].values,
                             W.loc[W.coarse == 'sports', f'{blk}__{f}'].values)) for f in FEATS}
    summary['effect_BTC15M_vs_sports'] = eff2
    # heatmap of standardized medians
    Z = (med - med.mean()) / (med.std() + 1e-12)
    fig, ax = plt.subplots(figsize=(13, 5))
    im = ax.imshow(Z.values, aspect='auto', cmap='RdBu_r', vmin=-2.5, vmax=2.5)
    ax.set_yticks(range(len(fines))); ax.set_yticklabels(fines)
    ax.set_xticks(range(len(FEATS))); ax.set_xticklabels(FEATS, rotation=90, fontsize=7)
    ax.grid(False)
    plt.colorbar(im, ax=ax, fraction=.02, label='z-score of class median')
    ax.set_title('Microstructure fingerprint: class medians of scale-free window features (last 256 events)')
    fig.tight_layout(); fig.savefig(f'{REP}/fingerprint_heatmap.png'); plt.close(fig)
    # effect-size bar chart
    fig, axs = plt.subplots(1, 3, figsize=(14, 8), sharey=True)
    for ax, c in zip(axs, E.columns):
        v = E[c].reindex(FEATS)
        ax.barh(range(len(FEATS)), v.values, color=['#c44' if x > 0 else '#36c' for x in v.values])
        ax.set_title(c.replace('_', ' ')); ax.axvline(0, color='k', lw=.6)
        ax.set_yticks(range(len(FEATS))); ax.set_yticklabels(FEATS, fontsize=7)
    fig.suptitle("Cohen's d between coarse classes (N=256 event window). |d|>0.8 = large")
    fig.tight_layout(); fig.savefig(f'{REP}/effect_sizes.png'); plt.close(fig)
    # violin/box plots of key features
    key = ['tm_cv', 'tm_batch', 'ev_trade', 'ev_cxl_add', 'ev_touch', 'sp_mean', 'sp_volnorm', 'im1_abs', 'im_pred',
           'dsh1', 'd_l1_vs_trade', 'pm_change', 'pm_ac1', 'pm_eff', 'tr_sign_ac1', 'tr_size_vs_l1', 'tr_impact',
           'tr_walk']
    fig, axs = plt.subplots(3, 6, figsize=(18, 9))
    for ax, f in zip(axs.flat, key):
        data = [W.loc[W.fine == c, f'{blk}__{f}'].dropna().values for c in fines]
        bp = ax.boxplot(data, showfliers=False, patch_artist=True, widths=.6)
        for patch, c in zip(bp['boxes'], fines):
            co = W.loc[W.fine == c, 'coarse'].iloc[0]
            patch.set_facecolor(CCOL[co]); patch.set_alpha(.6)
        ax.set_xticks(range(1, len(fines) + 1)); ax.set_xticklabels(fines, rotation=90, fontsize=6)
        ax.set_title(f, fontsize=9)
    fig.suptitle('Distribution of key scale-free features by market (blue=crypto binaries, orange=sports, green=perps)')
    fig.tight_layout(); fig.savefig(f'{REP}/feature_boxes.png'); plt.close(fig)
    # how separability grows with window: per-feature median spread (between-class var / total var) by block
    blocks = [f'N{n}' for n in (16, 64, 256, 1024)] + [f'W{w}' for w in (1, 10, 60, 600)]
    eta = {}
    for b in blocks:
        r = {}
        for f in FEATS:
            x = W[f'{b}__{f}']
            ok = x.notna()
            if ok.sum() < 100:
                r[f] = np.nan; continue
            g = W.loc[ok].groupby('fine')[f'{b}__{f}']
            m = x[ok].mean()
            ssb = (g.count() * (g.mean() - m) ** 2).sum(); sst = ((x[ok] - m) ** 2).sum()
            r[f] = ssb / sst if sst > 0 else np.nan
        eta[b] = r
    ETA = pd.DataFrame(eta)
    ETA.to_csv(f'{REP}/eta2_by_window.csv')
    summary['eta2_by_window'] = ETA.round(3).to_dict(orient='index')
    summary['coverage_time_windows'] = {b: float(W[f'{b}__tm_cv'].notna().mean()) for b in blocks}
    fig, ax = plt.subplots(figsize=(13, 5))
    im = ax.imshow(ETA.T.values, aspect='auto', cmap='viridis', vmin=0, vmax=.8)
    ax.set_xticks(range(len(FEATS))); ax.set_xticklabels(FEATS, rotation=90, fontsize=7)
    ax.set_yticks(range(len(blocks))); ax.set_yticklabels(blocks); ax.grid(False)
    plt.colorbar(im, ax=ax, fraction=.02, label='η² (share of variance explained by market class)')
    ax.set_title('How much each feature separates market classes, by history window (N=events, W=seconds)')
    fig.tight_layout(); fig.savefig(f'{REP}/eta2_by_window.png'); plt.close(fig)

    # ---------------- trade-level stylized facts
    T = load_trades()
    T = T[T.fine.isin(ORDER)]
    hz = ['0p1', '0p5', '1p0', '10p0']
    mk = T.groupby('fine')[[f'y_mk_{h}' for h in hz]].mean().loc[[f for f in fines if f in set(T.fine)]]
    tox = T.groupby('fine')[[f'y_mk_{h}' for h in hz]].agg(lambda x: (x > 0).mean()).loc[mk.index]
    net = T.groupby('fine')[[f'y_net_{h}' for h in hz]].agg(lambda x: (x > 0).mean()).loc[mk.index]
    sprd = T.groupby('fine').trf_spread.median().loc[mk.index]
    fee = T.groupby('fine').trf_fee_ticks.median().loc[mk.index]
    summary['trade'] = {'n_orders': T.fine.value_counts().to_dict(), 'mean_markout_ticks': mk.round(3).to_dict(orient='index'),
                        'p_toxic': tox.round(3).to_dict(orient='index'), 'p_net_profitable': net.round(4).to_dict(orient='index'),
                        'median_spread_ticks': sprd.round(2).to_dict(), 'median_fee_ticks_per_side': fee.round(3).to_dict()}
    # markout normalised by half spread
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.5))
    xs = [0.1, 0.5, 1, 10]
    for c in mk.index:
        co = T.loc[T.fine == c, 'coarse'].iloc[0]
        hs = max(sprd[c] / 2, 0.5)
        axs[0].plot(xs, (mk.loc[c].values + hs) / hs, marker='o', label=c, color=CCOL[co], alpha=.85,
                    ls='-' if c in ('BTC15M', 'BTCPERP', 'NFL', 'MLB') else '--')
        axs[1].plot(xs, tox.loc[c].values, marker='o', label=c, color=CCOL[co], alpha=.85,
                    ls='-' if c in ('BTC15M', 'BTCPERP', 'NFL', 'MLB') else '--')
        axs[2].plot(xs, net.loc[c].values, marker='o', label=c, color=CCOL[co], alpha=.85,
                    ls='-' if c in ('BTC15M', 'BTCPERP', 'NFL', 'MLB') else '--')
    for ax in axs:
        ax.set_xscale('log'); ax.set_xlabel('horizon (s)')
    axs[0].set_ylabel('mid move in taker direction / half-spread'); axs[0].axhline(1, color='k', lw=.6)
    axs[0].set_title('Price impact after taker orders (1 = price moved a half spread)')
    axs[1].set_title('P(toxic): mid moves past the fill price'); axs[2].set_title('P(taker round-trip profitable after 2 fees)')
    axs[2].legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(f'{REP}/markouts.png'); plt.close(fig)
    # impact vs relative size
    T['szb'] = pd.cut(T.trf_logsize_vs_hit, [-20, -3, -2, -1, -0.5, 0, 0.5, 20])
    imp = T.groupby(['coarse', 'szb'], observed=True).y_mk_1p0.mean().unstack(0)
    tox_sz = T.groupby(['coarse', 'szb'], observed=True).y_mk_1p0.apply(lambda x: (x > 0).mean()).unstack(0)
    summary['impact_vs_size_1s'] = {str(k): v for k, v in imp.round(3).to_dict(orient='index').items()}
    summary['toxic_vs_size_1s'] = {str(k): v for k, v in tox_sz.round(3).to_dict(orient='index').items()}
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    for co in imp.columns:
        axs[0].plot(range(len(imp)), imp[co].values, marker='o', color=CCOL[co], label=co)
        axs[1].plot(range(len(tox_sz)), tox_sz[co].values, marker='o', color=CCOL[co], label=co)
    for ax in axs:
        ax.set_xticks(range(len(imp))); ax.set_xticklabels([str(i) for i in imp.index], rotation=30, fontsize=7)
        ax.set_xlabel('log(order size / displayed depth at the touch)')
        ax.legend()
    axs[0].set_title('1s markout (ticks, taker view) vs relative order size')
    axs[1].set_title('P(toxic at 1s) vs relative order size')
    fig.tight_layout(); fig.savefig(f'{REP}/impact_vs_size.png'); plt.close(fig)
    # momentum: toxicity vs prior 10s move in trade direction
    T['momb'] = pd.cut(T.trf_mom10, [-1e9, -3, -1, -0.01, 0.01, 1, 3, 1e9])
    mom = T.groupby(['coarse', 'momb'], observed=True).y_mk_1p0.apply(lambda x: (x > 0).mean()).unstack(0)
    summary['toxic_vs_prior10s_move'] = {str(k): v for k, v in mom.round(3).to_dict(orient='index').items()}
    # imbalance response from anchors
    W['imb'] = pd.cut(W['N16__im1_mean'], np.linspace(-1, 1, 11))
    resp = W.groupby(['coarse', 'imb'], observed=True)['fw_1.0'].mean().unstack(0)
    summary['imbalance_response_1s'] = {str(k): v for k, v in resp.round(3).to_dict(orient='index').items()}
    fig, ax = plt.subplots(figsize=(6, 4))
    for co in resp.columns:
        ax.plot(np.linspace(-.9, .9, 10)[:len(resp)], resp[co].values, marker='o', color=CCOL[co], label=co)
    ax.set_xlabel('L1 imbalance (mean of last 16 events)'); ax.set_ylabel('E[mid change next 1s] (ticks)')
    ax.set_title('Order-book imbalance → next-second mid move'); ax.legend()
    fig.tight_layout(); fig.savefig(f'{REP}/imbalance_response.png'); plt.close(fig)

    # ---------------- sequence-level stylized facts from raw events (sample of markets)
    acf_rows = {}
    rng = np.random.default_rng(0)
    for fine in fines:
        ms = W.loc[W.fine == fine, ['ticker', 'day', 'regime']].drop_duplicates()
        ms = ms.sample(min(len(ms), 8), random_state=0)
        ts_acf = []; dt_hist = []; m_acf = []
        for _, r in ms.iterrows():
            path = f"{DATA}/events/{r.regime}/{r.day}/{r.ticker.replace('/', '_')}.parquet"
            if not os.path.exists(path):
                continue
            df = pd.read_parquet(path, columns=['rn', 'ty', 'sd', 'valid', 'bp0', 'ap0'])
            tr = df[df.ty == 1]
            s = tr.sd.values.astype(float)
            if len(s) > 200:
                s = s - s.mean()
                v = (s * s).mean()
                ts_acf.append([(s[:-k] * s[k:]).mean() / v for k in range(1, 31)])
            mid = ((df.bp0 + df.ap0) / 2).where(df.valid == 1).ffill().values
            dm = np.diff(mid); dm = np.sign(dm[np.abs(dm) > 0])
            if len(dm) > 100:
                dm = dm - dm.mean(); v = (dm * dm).mean()
                m_acf.append([(dm[:-k] * dm[k:]).mean() / v for k in range(1, 11)])
            dt = np.diff(df.rn.values) / 1e9
            dt = dt[dt > 0]
            if len(dt) > 100:
                x = dt / dt.mean()
                dt_hist.append(np.histogram(np.log10(x), bins=np.linspace(-6, 3, 37), density=True)[0])
        acf_rows[fine] = dict(trade_sign_acf=np.nanmean(ts_acf, 0).tolist() if ts_acf else None,
                              mid_sign_acf=np.nanmean(m_acf, 0).tolist() if m_acf else None,
                              dt_hist=np.nanmean(dt_hist, 0).tolist() if dt_hist else None)
    summary['stylized'] = acf_rows
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.5))
    for fine, r in acf_rows.items():
        co = W.loc[W.fine == fine, 'coarse'].iloc[0]
        ls = '-' if fine in ('BTC15M', 'BTCPERP', 'NFL', 'MLB') else '--'
        if r['trade_sign_acf']:
            axs[0].plot(range(1, 31), r['trade_sign_acf'], color=CCOL[co], ls=ls, label=fine)
        if r['mid_sign_acf']:
            axs[1].plot(range(1, 11), r['mid_sign_acf'], color=CCOL[co], ls=ls, marker='.', label=fine)
        if r['dt_hist']:
            axs[2].plot(np.linspace(-6, 3, 37)[:-1] + .125, r['dt_hist'], color=CCOL[co], ls=ls, label=fine)
    axs[0].set_xscale('log'); axs[0].set_title('Trade-sign autocorrelation (order-flow persistence)'); axs[0].set_xlabel('lag (trades)')
    axs[1].set_title('Mid-change sign autocorrelation (<0 = bounce)'); axs[1].set_xlabel('lag (mid changes)'); axs[1].axhline(0, color='k', lw=.6)
    axs[2].set_title('Inter-event time / its mean (log10) — burstiness shape'); axs[2].set_xlabel('log10(Δt / mean Δt)')
    axs[2].legend(fontsize=6, ncol=2)
    fig.tight_layout(); fig.savefig(f'{REP}/stylized_facts.png'); plt.close(fig)
    # depth profile
    prof = W.groupby('fine')[[f'N256__dsh{k}' for k in range(1, 6)]].mean().loc[fines]
    summary['depth_profile'] = prof.round(4).to_dict(orient='index')
    fig, ax = plt.subplots(figsize=(7, 4))
    for fine in fines:
        co = W.loc[W.fine == fine, 'coarse'].iloc[0]
        ax.plot(range(1, 6), prof.loc[fine].values, marker='o', color=CCOL[co], alpha=.8,
                ls='-' if fine in ('BTC15M', 'BTCPERP', 'NFL', 'MLB') else '--', label=fine)
    ax.set_xlabel('book level'); ax.set_ylabel('share of top-5 depth'); ax.set_title('Depth profile (where liquidity sits)')
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout(); fig.savefig(f'{REP}/depth_profile.png'); plt.close(fig)

    json.dump(summary, open(f'{REP}/summary.json', 'w'), indent=1, default=float)
    for p in glob.glob(f'{REP}/*.png'):
        wandb.log({os.path.basename(p)[:-4]: wandb.Image(p)})
    wandb.log({'medians_N256': wandb.Table(dataframe=med.reset_index()),
               'effect_sizes': wandb.Table(dataframe=E.reset_index())})
    run.summary.update({'n_anchors': len(W), 'n_trades': len(T)})
    run.finish()
    log('classical analysis done', 'analysis.log')


if __name__ == '__main__':
    main()
