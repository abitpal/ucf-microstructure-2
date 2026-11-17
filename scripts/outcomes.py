"""Fetch the TRUE settled outcome for every panel market.

The settlement PnL so far used each market's last observed mid as a stand-in
for the 0/1 result, which is a proxy and was labelled as one. `GET /markets/
{ticker}` returns `result` ("yes"/"no") once `status` is finalized/settled, so
the real label is available and there is no reason to keep the proxy.
"""
import os, glob, json, time, re
import urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor

OUT = "/var/lib/kalshi-study/SETTLE.json"
BASE = "https://api.elections.kalshi.com/trade-api/v2/markets/"

tick = set()
for p in glob.glob("/mnt/data/panels/*/panel_*.parquet"):
    b = os.path.basename(p)
    m = re.match(r"panel_(.+)\.parquet$", b)
    if m: tick.add(m.group(1))
tick = sorted(tick)

done = {}
if os.path.exists(OUT):
    try: done = json.load(open(OUT))
    except Exception: done = {}
todo = [t for t in tick if t not in done]
print(f"{len(tick)} markets, {len(done)} cached, {len(todo)} to fetch", flush=True)

def one(t):
    for attempt in range(4):
        try:
            rq = urllib.request.Request(BASE + t, headers={"User-Agent": "kalshi-study"})
            with urllib.request.urlopen(rq, timeout=20) as r:
                m = json.load(r).get("market", {})
            return t, {"result": m.get("result"), "status": m.get("status"),
                       "close": m.get("close_time"), "strike": m.get("floor_strike")}
        except urllib.error.HTTPError as e:
            if e.code == 429: time.sleep(1.0 + attempt); continue
            return t, {"err": f"HTTP{e.code}"}
        except Exception as e:
            time.sleep(0.5 + attempt)
            if attempt == 3: return t, {"err": type(e).__name__}
    return t, {"err": "retries"}

t0 = time.time(); n = 0
with ThreadPoolExecutor(max_workers=8) as ex:
    for t, v in ex.map(one, todo):
        done[t] = v; n += 1
        if n % 250 == 0:
            json.dump(done, open(OUT + ".tmp", "w")); os.replace(OUT + ".tmp", OUT)
            el = time.time() - t0
            print(f"  {n}/{len(todo)}  {el/60:.1f} min  eta {(el/n*(len(todo)-n))/60:.1f} min", flush=True)
json.dump(done, open(OUT + ".tmp", "w")); os.replace(OUT + ".tmp", OUT)

ok = [v for v in done.values() if v.get("result") in ("yes", "no")]
print(f"\nwrote {OUT}: {len(done)} markets, {len(ok)} with a settled result")
from collections import Counter
print("  result :", Counter(v.get("result") for v in done.values()).most_common())
print("  status :", Counter(v.get("status") for v in done.values()).most_common()[:6])
