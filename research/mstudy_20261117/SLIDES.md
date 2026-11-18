# Microstructure study (2026-11-17): slide material

Tape: Kalshi crypto binaries, sports, and perps, 2026-09-18 → 09-23. Splits are disjoint by market-day (fair models: train 09-18..21, val 09-22, test 09-23).
- Full write-up: `site/index.html` (open locally) or https://claude.ai/artifact/C3CLpPzPdzm6rwuXiaUjnB
- Figures: `report/<section>/*.png`
- Numbers: `report/<section>/summary.json` and `report/fair/*.json`
- Curve data: `data/curves_data.json`

## Suggested deck

1. **Question.** Does microstructure differ across market types? Does it predict trade PnL or fair value after fees?

2. **Classical stylized facts.**
   - Figures: `report/classical/stylized_facts.png`, `fingerprint_heatmap.png`, `effect_sizes.png`, `markouts.png`, `impact_vs_size.png`.

3. **Market types are separable without trivial features.**
   - Volume and tick-grid features are excluded.
   - Coarse balanced accuracy: 0.906 (16 events) → 0.986 (1024 events).
   - Fine (13 classes): 0.68 → 0.86.
   - Within-family accuracy: sports 0.78, crypto 0.99, perps 0.97.
   - Figures: `report/classify/window_sweep.png`, `ablations.png`, `importance.png`, `cm_multiwindow_fine_strict.png`.

4. **The autoencoder embedding also separates them, but less well.**
   - Probe accuracy: coarse 0.83, fine 0.54. Strict AE: 0.77 / 0.40.
   - Supervised GRU on 64 raw events: 0.85.
   - Figures: `report/ae/embedding_pca.png`, `latent_class_means.png`, `cm_gbm_probe_fine.png`.

5. **Toxic vs retail flow is predictable, but not profitable after fees.**
   - Toxicity AUC: 0.927 / 0.892 / 0.870 / 0.785 at 0.1 / 0.5 / 1 / 10 s.
   - At 1 s: sports 0.92, crypto 0.82, perps 0.77.
   - Even the model's top 1% of trades loses money after two taker fees.
   - AE and classifier embeddings add nothing over trade + window features.
   - Figures: `report/trade/*.png`.

6. **Fair-value prediction from raw events ("bitter lesson").** Test R² on BTC 15-minute binaries:

   | Model | 0.1s | 0.5s | 1s | 10s |
   |---|---|---|---|---|
   | LightGBM, handcrafted | 0.232 | 0.152 | 0.114 | 0.015 |
   | Transformer, 75 min | 0.233 | 0.154 | 0.116 | 0.018 |
   | Transformer, 75 + 90 min | 0.244 | 0.158 | 0.119 | 0.017 |
   | GRU, 60 min | **0.262** | **0.164** | **0.123** | **0.018** |

   On BTC perps, the GRU (0.187 at 0.1s) beats LightGBM (0.128) and the Transformer (0.123).

7. **Out-of-sample days.** Frozen checkpoints were run on 20 unseen BTC15M days and 19 unseen perp days, including today (Nov 17).
   - Pooled 0.1s R², BTC15M: GRU 60 min 0.262, Transformer continued 0.247, LightGBM 0.235. The GRU beats LightGBM on 17 of 20 days.
   - Perp: GRU 0.173 vs LightGBM 0.092. The GRU wins on 19 of 19 days.
   - No decay over 18 days. Today every model drops to about 0.13 (larger, more frequent mid moves), but the raw-event models still lead.
   - Source: `site/oos.html`, `report/fair_oos/per_day_*.png`, `report/fair_oos/summary.json`.

8. **Training curves.**
   - Source: `site/curves.html` and `data/curves_data.json`.
   - Transformer validation R² was still rising when stopped: 0.224 → 0.252.

9. **Takeaways and next steps.** The structure is real and learnable. Edge after fees is not shown. Raw-event models match or beat handcrafted features, and scale with compute.

Ops and issues log: `site/ops.html`, `logs/`. Code: `code/`. Rebuild the site with `build_site.py`.
