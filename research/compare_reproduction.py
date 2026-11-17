"""Summarize BTC reproduction against the previously reported tradeability result."""
import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('reproduced', type=Path, help='tradecheck_reproduction.json')
p.add_argument('--original', type=Path, default=Path('TRADECHECK.json'))
a = p.parse_args()
original = json.loads(a.original.read_text())
new = json.loads(a.reproduced.read_text())
for h in ('0.1', '1.0', '2.0', '5.0', '30.0', 'settle'):
    key = f'BTC|{h}'
    if key not in original or key not in new:
        print(f'{key}: missing from one result')
        continue
    x, y = original[key], new[key]
    print(key)
    for field in ('r2_0p1s', 'trades', 'rate', 'mark_cents_per_trade', 'round_cents_per_trade', 'nfeat', 'test_markets'):
        old, got = x.get(field), y.get(field)
        if isinstance(old, (float, int)) and isinstance(got, (float, int)):
            print(f'  {field:24s} original={old:.8g}  reproduced={got:.8g}  difference={got-old:+.8g}')
        else:
            print(f'  {field:24s} original={old}  reproduced={got}')
print('Counts and reported metrics are necessary but do not prove identical input rows or an independent test.')
