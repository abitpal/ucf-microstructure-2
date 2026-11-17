"""Shared helpers for the microstructure study (mstudy)."""
import os, sys, json, time, datetime as dt
ROOT = '/mnt/scratch/mstudy'
DATA = f'{ROOT}/data'
RUNS = f'{ROOT}/runs'
LOGS = f'{ROOT}/logs'
ACCOUNT = 'kalshiobfdfaed53'
CONTAINER = 'tape'
BLOB_OUT = 'research/mstudy_20261117'
WANDB_PROJECT = 'kalshi-microstructure'

_cli = None


def container():
    global _cli
    if _cli is None:
        from azure.identity import DefaultAzureCredential
        from azure.storage.blob import BlobServiceClient
        svc = BlobServiceClient(f'https://{ACCOUNT}.blob.core.windows.net', credential=DefaultAzureCredential())
        _cli = svc.get_container_client(CONTAINER)
    return _cli


def list_blobs(prefix):
    return [(b.name, b.size) for b in container().list_blobs(name_starts_with=prefix)]


def get_blob(name):
    for i in range(5):
        try:
            return container().download_blob(name, max_concurrency=2).readall()
        except Exception as e:  # transient
            if i == 4:
                raise
            time.sleep(2 * (i + 1))


def put_file(local, name):
    with open(local, 'rb') as f:
        container().upload_blob(f'{BLOB_OUT}/{name}', f, overwrite=True)


def log(msg, fn='main.log'):
    line = f'[{dt.datetime.utcnow().strftime("%H:%M:%S")}] {msg}'
    print(line, flush=True)
    os.makedirs(LOGS, exist_ok=True)
    with open(f'{LOGS}/{fn}', 'a') as f:
        f.write(line + '\n')


# ---------------------------------------------------------------- market taxonomy
SPORT_MAP = [
    ('KXNFL', 'NFL'), ('KXNCAAF', 'NCAAF'), ('KXMLB', 'MLB'), ('KXWNBA', 'WNBA'), ('KXNBA', 'NBA'),
    ('KXNHL', 'NHL'), ('KXATP', 'TENNIS'), ('KXWTA', 'TENNIS'), ('KXPGA', 'GOLF'), ('KXUFC', 'MMA'),
]
SOCCER_HINTS = ('MLS', 'EPL', 'LALIGA', 'SERIEA', 'BUNDESLIGA', 'LIGUE1', 'UCL', 'USL', 'ALLSVENSKAN', 'ARGPREM',
                'ARGNACB', 'LIGAMX', 'EREDIVISIE', 'BRASIL', 'AFCON', 'ECULP', 'VENFUTVE', 'SOCCER', 'UEL', 'CHAMP', 'FUT')


def classify(ticker):
    """-> (coarse, fine) labels."""
    s = ticker.split('-')[0]
    if s.endswith('PERP'):
        return 'perp', ('BTCPERP' if s == 'KXBTCPERP' else 'ETHPERP' if s == 'KXETHPERP' else 'altPERP')
    if s in ('KXBTC15M',):
        return 'crypto', 'BTC15M'
    if s.endswith('15M'):
        return 'crypto', 'alt15M'
    if s in ('KXBTCD', 'KXETHD'):
        return 'crypto', 'cryptoHourly'
    if s in ('KXBTC', 'KXETH'):
        return 'crypto', 'cryptoRange'
    for p, n in SPORT_MAP:
        if s.startswith(p):
            return 'sports', n
    if any(h in s for h in SOCCER_HINTS):
        return 'sports', 'SOCCER'
    return 'sports', 'otherSport'


CRYPTO_SERIES = {'KXBTC15M', 'KXETH15M', 'KXSOL15M', 'KXXRP15M', 'KXBTCD', 'KXETHD', 'KXBTC', 'KXETH'}
NON_SPORT_PREFIX = ('KXBTC', 'KXETH', 'KXSOL', 'KXXRP', 'KXNEAR', 'KXBNB', 'KXDOGE', 'KXHYPE', 'KXZEC', 'KXGOLD',
                    'KXSILVER', 'KXWTI', 'KXINX', 'KXNASDAQ', 'KXHIGH', 'KXRAIN', 'KXLOW')


def keep_series(regime, s):
    if regime == 'perps':
        return s.endswith('PERP')
    if regime == 'crypto':
        return s in CRYPTO_SERIES
    if regime == 'football':
        return not s.startswith(NON_SPORT_PREFIX)
    return False


def wandb_init(name, group, config=None, job_type='train'):
    sys.path.insert(0, f'{ROOT}/pylib')
    os.environ.setdefault('WANDB_MODE', 'offline')
    os.environ.setdefault('WANDB_DIR', RUNS)
    os.environ.setdefault('WANDB_SILENT', 'true')
    import wandb
    return wandb.init(project=WANDB_PROJECT, name=name, group=group, job_type=job_type, config=config or {},
                      dir=RUNS)


def trade_keep(fine_col, cap=25000, seed=0):
    """deterministic per-file subsample of taker orders, capped per fine class (shared by ae.py / trade_model.py)."""
    import numpy as np, pandas as pd
    rng = np.random.default_rng(seed)
    fine_col = pd.Series(np.asarray(fine_col))
    keep = np.zeros(len(fine_col), bool)
    for f, idx in fine_col.groupby(fine_col).indices.items():
        k = min(len(idx), cap)
        keep[rng.choice(idx, k, replace=False)] = True
    return keep
