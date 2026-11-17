import sys, pandas as pd, numpy as np
T = pd.read_parquet(sys.argv[1])
print(len(T), T.fine.value_counts().to_dict())
g = T.groupby('fine')
print(g.trf_price_vs_mid.apply(lambda x: (x > 0).mean()).rename('frac_pay_above_mid').to_string())
cols = ['y_mk_0p1', 'y_mk_1p0', 'y_mk_10p0', 'y_net_1p0', 'y_net_10p0', 'trf_spread', 'trf_fee_ticks']
print(g[cols].median().round(3).to_string())
print('P(toxic 1s)', g.y_mk_1p0.apply(lambda x: (x > 0).mean()).round(3).to_dict())
print('P(net>0 10s)', g.y_net_10p0.apply(lambda x: (x > 0).mean()).round(3).to_dict())
