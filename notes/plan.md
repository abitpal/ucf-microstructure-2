Correlated markets, market data, predict x secs into the future (0.1s), neural networks w/ deep, wide, and weight decay. 

What I want to cover: 
1. Let’s train a neural network on a bunch of features, what happens? How does it trade - is it just momentum? What features seem important (gradient tracking on features to see) 
1. Could we market make? 
1. How about correlated markets. Could we take in these? 
1. When could we take? Let's try different k, which markets are good? 
1. 


A new strategy: 

Which trades should we take? How should we train the model --> I feel like


Questions: 

1. on which data is the model trained? --> which model did the best, training on all the market data or per basis market data? 

Also let's try scaling this model more and see what's the best. 

Internal research dossier · v1

Kalshi Crypto Microstructure
Twenty-five days of 15-minute crypto binaries, four coins, 952M book events. A directional signal that survives every control we threw at it — and cannot pay for its own exit. This is the verified record, the reasons, and what is still open.

Window 2026-09-15 → 10-09
Coins BTC ETH XRP NEAR
Test markets ~350/coin
Features 219
Tape 138 GB, zero gaps
The signal is real. It earns 0.4–0.9¢ per contract, and a round trip costs 0.3¢ in the extreme tails and 2.2–4.5¢ mid-book.

Every number below charges cost explicitly, because the distinction between one cost and two is the entire result. Figures quoting a single entry cost are upper bounds, not realisable PnL.

R² out-of-sample

+0.0331
BTC, 0.1s target, vs zero

Mark — one cost

+0.658¢
per contract, p<0.0001

Round trip

−0.535¢
negative in all 11 bands

Edge half-life

<50ms
gone entirely by 50ms

Capacity ceiling

~$6k/day
BTC is 93% of it

01
The signal is real
An MLP, 640 wide × 5 deep, 219 scale-free features, two seeds averaged. Target is (mid(t+h) − mid(t)) / mid(t). Market-disjoint, time-ordered splits; every interval bootstrapped over markets, never rows.

Fake PnL by horizon — one entry cost: half spread + 0.07·m·(1−m), charged once. No exit.
Coin	Horizon	Trades	Take	PnL / trade	95% CI	p	
BTC	0.1s	44,284	0.78%	+0.6582¢	[+0.533, +0.803]	0.0000	sig
BTC	1.0s	44,128	0.77%	+0.8093¢	[+0.678, +0.965]	0.0000	sig
BTC	5.0s	43,779	0.77%	+0.8707¢	[+0.672, +1.091]	0.0000	sig
BTC	30s	39,080	0.68%	+1.1188¢	[+0.657, +1.647]	0.0000	sig
BTC	settle	44,284	0.78%	+0.3579¢	[−0.781, +1.705]	0.5640	ns
ETH	0.1s	9,621	0.64%	+1.1400¢	[+0.957, +1.324]	0.0000	sig
XRP	0.1s	4,297	0.41%	+0.9132¢	[+0.697, +1.164]	0.0000	sig
NEAR	0.1s	2,205	0.46%	+0.4904¢	[+0.200, +0.859]	0.0000	sig
Four controls it survived
Validation — each of these was designed to kill the result.
Control	What it tests	Result
Staleness audit	Target mixed a panel base mid with a tape forward mid — if the panel lagged, the model would be predicting catch-up	exact match on 100.0000% of 407,010 rows
Fill-rate separation	Resting orders on the model's side vs flipped vs random, measured through a different data path	90.5% / 29.0% / 59.8%
Calibration slope	Actual on predicted through the origin; <1 means over-confident	1.06–1.24 — mildly under-confident
Forced-side controls	Same gate, same costs, direction held constant — catches static bias masquerading as skill	caught a fake +1.70¢ on settlement
Why the fill-rate control matters most

A 3× separation between the model's side and the anti-signal's side, measured off the touch/trade tape rather than the mid tape the model trained on. A spurious signal does not produce that through independent data.

02
It cannot pay for its exit
A timed hold has to be closed, and Kalshi charges the taker fee on every trade. Since entry cost ≈ exit cost, round trip ≈ mark − cost — so the whole mark is consumed.

Round trip by price band — 0.5s model, entry at t+10ms with the no-chase guard, 5s hold, exit charged. BTC.
Band	Trades	Cost in	Cost out	Mark	Round trip	95% CI		Depth
0–2¢ *	431	0.166¢	0.166¢	+0.327¢	+0.161¢	[−0.016, +0.324]	ns	10
2–5¢ *	4,735	0.301¢	0.299¢	+0.430¢	+0.131¢	[−0.060, +0.339]	ns	17
5–10¢ *	6,434	0.519¢	0.566¢	+0.380¢	−0.186¢	[−0.571, +0.138]	ns	17
10–20¢	1,064	1.393¢	1.229¢	+0.928¢	−0.301¢	[−1.212, +0.656]	ns	70
20–35¢	2,173	1.910¢	1.878¢	+0.289¢	−1.589¢	[−2.562, −0.657]	sig	56
35–65¢	8,547	2.250¢	2.162¢	+0.038¢	−2.123¢	[−2.643, −1.623]	sig	57
65–80¢	6,522	1.901¢	1.784¢	+0.551¢	−1.233¢	[−1.606, −0.792]	sig	52
80–90¢	6,756	1.414¢	1.308¢	+0.342¢	−0.966¢	[−1.399, −0.417]	sig	84
90–95¢ *	19,346	0.522¢	0.586¢	+0.132¢	−0.454¢	[−0.670, −0.252]	sig	15
95–98¢ *	23,278	0.290¢	0.309¢	−0.009¢	−0.318¢	[−0.426, −0.218]	sig	17
98–100¢ *	10,022	0.104¢	0.119¢	+0.056¢	−0.063¢	[−0.138, +0.041]	ns	126
* tapered tick region — 0.1¢ tick below 10¢ and above 90¢, 1.0¢ between

Not one band is significantly positive. The two best are the extreme tails, both with intervals through zero on 10–17 contracts of depth. Mid-book is −1.2¢ to −2.1¢, all significant.

The three exits — all of them.
Exit	Result	Status
Cross out at 1–30s	negative everywhere; significant in 7 of 11 bands	closed
Hold to settlement	no signal at all — R² negative in 12 of 12 coin×model cells, gross negative before any cost in 10 of 12	closed
Rest out as maker	+0.003¢, CI [−0.48, +0.46] — fill 87.4% at +3.09¢ against 12.6% at −21.40¢	≈ zero
The cleanest number in the study

Maker exit, 6 ticks, 180s, cost-filtered: 0.8739 × 3.0911 = +2.701 and 0.1261 × (−21.401) = −2.699. The gain when you fill and the loss when you don't cancel to within a third of a basis point. That is not a tuning artifact — it is what an efficiently priced market looks like from the inside.

03
Four structural findings
Why it fails, in mechanisms rather than numbers. These are the transferable parts.

3.1
Latency is brutal, and the clock starts at packet receipt
Guarded arm, BTC, 1s hold. The no-chase guard executes at t+delay only if the touch hasn't moved against us.
Delay	PnL / contract		Guard pass
0ms	+0.810¢	sig	100.0%
5ms	+0.380¢	sig	68.4%
10ms	+0.282¢	sig	60.1%
20ms	+0.077¢	ns	49.3%
50ms	−0.177¢	sig	38.1%
500ms	−0.833¢	sig	26.5%
Monotone across all four coins × four horizons × both arms, with a consistent sign flip. That coherence, not any single cell, is the evidence. Measured warm-socket round trip to Kalshi is ~11.0ms, so execution lands around 6–10ms — inside the positive region. The sweep's clock starts at recv_ns, so the 20–28ms feed latency is already inside t and does not stack.

Also: do not hold the target horizon. At 10ms, BTC at h=0.1s is +0.039¢ (ns) while h=5s is +0.360¢ (sig). Train short, hold long.

3.2
The edge is a depletion detector
Touch depth falls monotonically with predicted edge. BTC, deciles over gated-eligible rows.
Decile	Predicted edge	Touch depth	Spread
0	0.0028¢	2,950 ct	0.503¢
3	0.0327¢	2,381 ct	0.684¢
6	0.1512¢	1,058 ct	0.838¢
9	1.0638¢	208 ct	1.251¢
And the side asymmetry is the whole story.
Rows	Median depth, side we must hit
All test rows	1,251 ct
Gated trades — side we hit	50 ct
Gated trades — opposite side	2,080 ct
40× thinner on exactly the side we need. Book imbalance is the signal: the model predicts "up" largely because the ask is nearly empty, and then we try to buy that nearly-empty ask. The edge is inseparable from the thinness that creates it.

Verified against the alternative explanation — Kalshi's known zero-quantity-level artifact is not present in our tape (0.000% of finite asks carry zero size over 4.98M events), so mid and spread are sound.

Capacity ceiling. BTC, 10ms guarded, 7,281 trades/day, single-level fills only.
Clip	Avg filled	$ / day
100	42 ct	1,096
500	113 ct	2,963
5,000	225 ct	5,891 — the ceiling
Unlimited clip tops out near $5.9k/day on BTC because that is all the depth there is. ETH ~$350, XRP ~$39, NEAR ~$18. Hard ceiling ≈ $6.3k/day, BTC 93% of it — and taking the whole touch every time would move the price.

3.3
Target horizon is the scaling lever, and R² picks it wrong
Total dollars over the test set (trades × PnL/trade), zero latency. The quantity that matters is not PnL per trade.
Coin	Target	h=0.1s	h=1s	h=5s	h=10s	h=30s	Take
BTC	0.1s	$295	$342	$344	$307	$396	0.68%
BTC	0.5s	$145	$521	$628	$635	$667	2.20%
BTC	1.0s	−$233	$414	$613	$740	$736	3.86%
ETH	1.0s	−$134	$130	$240	$275	$253	4.66%
XRP	1.0s	−$86	$67	$120	$140	$159	2.60%
NEAR	1.0s	−$33	$3	$49	$76	$127	3.55%
Roughly 2× more dollars on every coin from the longer target, driven by 5.7× more trades at half the edge each — and those extra trades sit lower in the edge deciles where the book is thicker, so capacity improves too.

R² would have chosen wrong

BTC R² goes 0.0323 → 0.0409 → 0.0249 across 0.1s / 0.5s / 1.0s, peaking at 0.5s, while dollars keep rising to 1.0s. ETH's R² falls monotonically while its dollars double. Never select the target on R².

Latency-adjusted, the optimum is interior: significance of positive PnL ends at the 1.0s target, and targets ≥5s are significantly negative. The 30s target takes 1.67M trades at 95% guard pass and −0.69¢/contract — its forecasts are so large they clear edge > cost everywhere and it fires on noise.

3.4
The gate is only calibrated for the mid-range
In the tapered bands cost collapses to ~0.14¢, so edge > cost is nearly free to clear. A long-horizon model predicts correspondingly large moves and clears that tiny bar almost always — 17.3% take rate at 98–100¢ against 1.18% for the 0.1s model — firing on mostly noise.

Latency-adjusted PnL per contract at 10ms, tapered regions only. BTC.
Band	0.1s model	0.5s model	1.0s model
0–2¢	no trades	+0.327 sig	+0.229 sig
5–10¢	+1.794 sig	+0.380 sig	+0.169 ns
95–98¢	+0.134 ns	−0.009	−0.049
98–100¢	+0.240 sig	+0.056 ns	−0.024
The 0.1s model is 2–10× better per contract in the tails and the only one that works above 95¢. Raising k in edge > k·cost improves per-contract quality 13× (0.117¢ → 1.556¢) but total dollars fall monotonically — so the fix is which model trades the band, not a higher threshold.

04
Deep-learning results
What we swept, and what it taught. Most of this is negative, which is the useful part.

Architecture — width helps, depth does not.
Study	Finding	Numbers
Width	Monotone in width; strong weight decay essential	w64 +0.0523 > w16 +0.0423 > w4 +0.0305; weak wd → −0.0912
Iso-param depth	At fixed parameter count depth hurts; residual stops the decay but adds nothing	P30k plain L1 +0.011142 → L7 +0.008434; paired L1→L2 −0.000602 sig
Init	ReZero + warm start beats zero-init and stays stable at depth	0.0399 vs 0.0312; zero-init collapsed to −0.0222 at d1
Optimizer	Large batch is catastrophic	bs1024 +0.0273; bs8192 −0.0032; bs65536 −0.0173
What did not work — all paired, same rows and seeds.
Idea	Result	
Symmetry augmentation	Doubling rows made it worse on all three model families — ridge −0.00364, lgbm −0.00418, mlp −0.00317 (win rates 29% / 7% / 25%)	sig
Seed bagging > 2	1 → 0.024145, 2 → 0.024856, 16 → 0.024981. Everything past two seeds is noise	flat
Cross-model blending	Simplex weights came out literally [0, 0, 1]; NNLS gave [0, 0, 1.321] — just rescaling the MLP. blend_nnls == scale_only exactly. Equal-weight was −0.010456 worse	sig
Cross-coin transfer	BTC-only is best (0.017566). Pooling significantly hurts (−0.002914); pretrain+finetune neutral (+0.000036)	sig
What did work
Idea	Result
Rank vol-normalisation
the one clean win	Same pooled R², 6.5× lower per-market sd, 8.6× better worst decile. BTC raw 0.008933 (sd 0.595, worst −0.646) → 0.008962 (sd 0.091, worst −0.075). On XRP it raised pooled R² too: 0.0167 → 0.0205, sd 12.12 → 0.051
Multi-head aux targets	Single head 0.002755 → 8 heads 0.003772 at identical params — +37% purely from auxiliary objectives shaping the shared trunk
Expanded features	106 → 334 gave ridge +0.003159 sig but the MLP only +0.00016 — the net was already extracting the structure
Unresolved, and it matters

We never re-ran LGBM vs MLP with the final 219-feature set. At the 106-feature stage, lgbm_bag3 beat the MLP on 95.7% of 141 markets (pooled 0.0216 vs 0.0155, t = 6.88, p = 2e−10). Later runs had the MLP ahead, but with a different config and after the LGBM early-stopping collapse was fixed. For a scalar tabular target, boosted trees winning is the base-rate expectation. "MLP is the best model" rests on less evidence than it should.

05
Traps and corrections
Every one of these produced a wrong number that was believed for a while. Read this section before running anything.

575 configs → 7 winners → 0 survived the holdout
Nobody cheated. Selection bias does not require bad faith, and with six people iterating on one test set it compounds. Seal a holdout and keep it sealed.
Row-level statistics inflate t by ~80×
Rows inside a market are heavily dependent. Every interval must bootstrap over markets. Every false finding in this project traces to row-level stats.
The same target gave R² +0.0323 and +0.0884
2.7× apart, purely from which rows were included. Absolute levels are not comparable across universes; paired deltas are. Ship the row mask.
scale_free is an allow-list that fails open
It silently dropped all 110 perp features three separate times. Ship the ordered feature-name list so this class of bug is impossible.
LGBM early-stopped at iteration 1
num_leaves=31 with min_child_samples=2000 quit after one tree, which read as "trees cannot learn this." They can.
A fixed step cap made more data look harmful
More rows at a fixed 3,000 steps means fewer epochs. Bracket fixed-steps and fixed-epochs separately.
Multiplying an unconditional "if filled" mean by a fill rate
Gave +1.088¢ where the honest conditional value is ~0, because fills concentrate exactly where the payoff is smallest. Never multiply those two.
A promising +0.226¢ that was noise
The 60s cross-out looked like the best cell in the study until it got an error bar: not significant on BTC or XRP, significantly negative on ETH.
Software self-inflictions
pkill -f matching its own shell and killing the launch; a module-level for L in DELAYS rebinding the global layer count to 0.020; 200MB of pandas warning spam slowing runs; a touch rule testing the wrong side of the book, which inverted the intended bracket.
06
Closed and open
Closed

Crossing out — negative in every band, target and latency
Hold to settlement — no signal; negative gross at zero cost
Mid-book maker exit — nets ~0 because tick < fee
"Latency is the constraint" — it isn't; the exit is
Cross-coin pooling — significantly negative
Open

Maker at 0–2¢ and 98–100¢ — the only bands where tick > fee
Perp-implied digital fair value — a different mechanism entirely
Distributional objectives — representation effects
Book-VWAP execution cost — corrects every $/day figure
LGBM vs MLP at 219 features — never settled
11 unused days of tape, never touched
Where the maker arithmetic finally works

A filled maker exit earns the tick and pays no fee (all four series are plain quadratic — maker fee zero, API-verified), so it needs tick > fee. That holds in exactly two bands on the entire grid: 0–2¢ (0.1¢ vs 0.072¢, depth 10) and 98–100¢ (0.1¢ vs 0.073¢, depth 126). Mid-book it is a 1.0¢ tick against a 1.7¢ fee, which is precisely why the maker study netted zero.

98–100¢ is the only cell with both the right inequality and real depth, and it is untested.

07
Proposed directions
Ranked by expected value per unit effort. Costs are spot-instance compute at $0.34/hr, measured from this session's actual run times.

01
Book-VWAP execution cost
~2h · $0.70 · no new model
Every dollar figure in this dossier assumes you fill min(clip, touch_size) at the touch price. That is wrong for size: if you want 500 contracts and 50 sit at the touch, you either get 50 or you walk the ladder and pay worse.

Do this first because it corrects existing results rather than adding new ones, and it decides whether the strategy is worth modelling at all. It is the largest unfixed weakness in the whole study.

02
Perp-implied digital fair value
~1 day · new signal
A 15-minute binary is a digital option on the perp, and we already pull floor_strike from the API. Fit the conditional distribution of the perp price at expiry, price the digital as P(S_T ≥ K), and compare to the Kalshi mid.

This is the only proposal that escapes the capacity ceiling. It does not come from book imbalance, so it does not vanish exactly when depth vanishes — and a mispricing against fair value persists far longer than a microstructure imbalance, so it does not need 10ms execution. v1 needs no flow: a realized-vol estimate and a normal/Student-t digital price, differenced against the mid. If that residual predicts anything, a learned density is the obvious upgrade.

03
Lattice-aware distributional head
~1h to implement · $0.50/config
The target is a discrete variable on Kalshi's tapered grid with a 69% atom at zero. MSE spends most of its gradient learning to predict that atom. A plain softmax over tick buckets is also wrong — cross-entropy treats the buckets as exchangeable, so predicting −1 tick when the truth is +1 costs the same as predicting −20.

Three fixes, cheapest first: EMD / CDF-L1 loss (Σ|CDF_pred − CDF_true|, a one-line change that encodes distance exactly); an ordinal cumulative-link head; or — preferred — a zero-inflated discretized logistic mixture, integrated over real bin edges.

The mixture handles all three problems at once: continuity is structural, the atom is representable, and it respects the tapered tick. That last point is not cosmetic — the tick is 0.1¢ below 10¢ and above 90¢ but 1.0¢ between, so a softmax over tick index would pool bins of physically different widths across exactly the bands where the maker arithmetic works.

04
Flow matching as an auxiliary objective
~$0.56 for the full t-sweep
With the conditional-OT path, the learned field is v(x,t) = E[x₁ − x₀ | x_t]. At t=0 and with x₀ ⊥ x₁, a single Euler step gives x̂₁ = x₀ + v(x₀,0) = E[Y|X] — one-step flow matching is identically MSE regression, not approximately. (Which is also why naive one-step distillation of diffusion models produces blurry samples.)

So all the distributional information lives in t > 0, and the value — if any — is that training the field at every noise level is a multi-scale denoising auxiliary task that may shape the trunk.

Two consequences make this unusually clean: you need no separate mean head, because v(·,0) is the mean predictor; and the ablation is exact and free — training only at t=0 recovers today's model bit-for-bit, so the experiment is just sweeping the t sampling distribution from a point mass at 0 toward uniform. If R² is flat, the distributional signal is inert here.

05
Representation probe — does distributional training help?
~$1.50 · falsifiable
Train trunk A with MSE and trunk B with a zero-inflated mixture on the same target and params. Freeze both. Linear-probe on tasks neither was trained for: p(up) at 1s, touch-L probability, forward VWAP, settlement 0/1, spread at t+1s.

This tests the representation hypothesis directly rather than by argument, and R² on the primary target is the wrong instrument for it. There is precedent — distributional RL improves performance on tasks that only use the mean — and our own multi-head result (+37% from aux objectives) is local evidence that the mechanism operates on this data. It also reframes the failed cross-coin transfer: we transferred a narrow discriminative head, and a generative trunk might transfer where that did not.

06
Settle LGBM vs MLP at 219 features
~$1 · unglamorous
Possibly the highest-value item on this list if R² is the objective. Boosted trees beat the net on 95.7% of markets at the 106-feature stage and the comparison was never re-run with the final feature set.

If LGBM wins, most of the distributional machinery above becomes moot — you would want quantile or distributional boosting instead.

07
Maker strategy, properly
a different build
Everything measured so far has us paying to enter. The untested alternative is earning the spread on both sides — continuous two-sided quoting, using the forecast defensively to skew and pull before being picked off — which makes the fee a non-event rather than the binding cost.

Needs a fill model and adverse-selection measurement, not another R² sweep. This is the structurally viable version of the whole project, and nothing here measures it.

08
The 11 unused days, and a rolling holdout
~3h · $1.70
The study used 14 of 25 collected days. Sep 29 – Oct 9 has never been touched: 61 GB, ~79% more data, zero gaps.

Test markets per coin go ~350 → ~600, narrowing every interval by ~24% — which would resolve several findings that currently sit right at the significance boundary. More importantly it is the only way to get a clean test set, since we have now looked at all 14 days. Collection has been off since Oct 9 and should be restarted if this becomes an ongoing benchmark.

08
Reproducing this
Data and artifacts. All derived data was deleted; everything below rebuilds from the tape.
Artifact	Where	Size	Rebuild
Raw tape, 25 days	tape/raw/regime=crypto/	138 GB	source of truth
Study results — 145 JSONs + 34 scripts	deploy/kalshi-study-results.tgz	430 KB	md5 verified
Panels	deleted	65 GB	~25 min/coin
Feature matrices	deleted	24 GB	finalgrid merge
Touch / size / mid tapes	deleted	7.4 GB	~55 / 30 / 26 min
Prediction caches	deleted	2.3 GB	~40 min/coin
Proposed split — seal before anyone looks

train Sep 16–30  ·  val Oct 1–4  ·  test Oct 5–9, held back. Sep 15 and Oct 9 are partial days and should be dropped. Splits ship as fixed market ticker lists, not as a rule to re-derive.

Model: MLP 640w × 5L, 219 scale-free features, 2 seeds, AdamW lr 1e−3 / wd 0.3, OneCycle, early stop on a 120-market holdout.
Splits market-disjoint and time-ordered; all intervals bootstrapped over markets (2,000 resamples). R² measured against zero, not the mean.
Fees: Kalshi taker 0.07·C·P·(1−P); maker zero on all four series (fee_type: quadratic, API-verified).

