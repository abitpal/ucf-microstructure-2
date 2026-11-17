# Microstructure deck: slide plan and design guide

For the Texas UCF microstructure study (2026-11-17). Built in Overleaf (Beamer). Content comes from `research/mstudy_20261117/SLIDES.md`. The reference deck is "BTC Dealer Flow" (44 slides, Google Slides). The goal is the same plain, research-first feel, done a bit more carefully.

Team on this pitch: Abitpal Gyawali, Andrew Li, Ishan Guthikonda, Kaylee Zheng, Karol Tang, Vishal Sundaram.

---

## Part 1: Slide overview (36 slides)

Figure paths are relative to `research/mstudy_20261117/report/`.

### Opening (1–4)
1. **Title.** "Does market microstructure tell you anything after fees?" Subtitle with the team and date. A row of six team portraits along the bottom.
2. **The team.** Six portraits, each with a name and a one-line role on the project (who built the tape, who trained the GRU, and so on). No bios.
3. **The question.** One sentence: does microstructure differ across market types, and does it predict trade PnL or fair value after fees? Nothing else on the slide.
4. **Answer up front.** Three lines: the structure is real, it is learnable, and we found no edge after fees. The audience should know where this ends before they sit through 30 slides.

### Data (5–8)
5. **The tape.** Kalshi crypto binaries, sports and perps, 2026-09-18 → 09-23. A simple timeline graphic.
6. **Splits.** Train 09-18..21, val 09-22, test 09-23. Disjoint by market-day. Draw it as three colored bars on the timeline, not as bullets.
7. **What an "event" is.** One annotated example: a few rows of raw events with arrows naming each field.
8. **What we left out and why.** Volume and tick-grid features are excluded so that classification cannot win on trivial features.

### Stylized facts (9–12)
9. **Section divider: "Do markets look different?"**
10. **Stylized facts.** `classical/stylized_facts.png`, full width. Title states what it shows, e.g. "Sports spreads are wider and quieter than crypto."
11. **Fingerprints.** `classical/fingerprint_heatmap.png`. Circle or box the 2–3 cells you talk about.
12. **Markouts and impact.** `classical/markouts.png` on top, `classical/impact_vs_size.png` below, or split across two slides if the labels get too small.

### Separability (13–17)
13. **Section divider: "Can a model tell them apart?"**
14. **Headline number.** A big "0.986" with a one-line caption: coarse balanced accuracy at 1024 events (0.906 at 16).
15. **Window sweep.** `classify/window_sweep.png`. Coarse 0.906 → 0.986, fine 13-class 0.68 → 0.86.
16. **Within each family.** Sports 0.78, crypto 0.99, perps 0.97, as a small bar chart or three big numbers. Point out that sports is hardest.
17. **What the model uses.** `classify/importance.png` and `classify/ablations.png`, with the confusion matrix `classify/cm_multiwindow_fine_strict.png` as backup.

### Autoencoder (18–20)
18. **The embedding separates them too, but less well.** `ae/embedding_pca.png`.
19. **Probe accuracy table.** Coarse 0.83 / fine 0.54. Strict AE 0.77 / 0.40. Supervised GRU on 64 raw events 0.85. Bold the GRU row.
20. **So what.** Unsupervised structure exists, but supervision on raw events beats it. One sentence plus `ae/latent_class_means.png`.

### Toxic flow (21–25)
21. **Section divider: "Can we spot informed traders?"**
22. **Yes.** Toxicity AUC 0.927 / 0.892 / 0.870 / 0.785 at 0.1 / 0.5 / 1 / 10 s, as a bar chart with a dashed chance line at 0.5.
23. **By family at 1 s.** Sports 0.92, crypto 0.82, perps 0.77.
24. **But it doesn't pay.** Even the model's top 1% of trades loses money after two taker fees. Make this the visually loudest slide in the section.
25. **Embeddings add nothing.** AE and classifier embeddings add nothing over trade + window features. `trade/auc_by_featureset.png`.

### Fair value (26–31)
26. **Section divider: "The bitter lesson, on BTC binaries."**
27. **Setup.** Predict fair value from raw events on BTC 15-minute binaries. Four models, four horizons.
28. **The results table.** The full R² table with the GRU row highlighted (bold or burnt orange) and best-per-column marked.
29. **Same data, as a chart.** R² by horizon, with the GRU in orange and the other models in gray, labelled directly on the lines.
30. **Perps too.** BTC perps at 0.1 s: GRU 0.187, LightGBM 0.128, Transformer 0.123.
31. **Training curves.** The Transformer's validation R² was still rising (0.224 → 0.252) when we stopped, so it is under-trained, not beaten. Source: `site/curves.html`.

### Close (32–36)
32. **Takeaways.** At most three lines: the structure is real and learnable, there is no edge after fees, and raw-event models match or beat handcrafted features and scale with compute.
33. **What we got wrong / what broke.** Pull 2–3 real incidents from `site/ops.html` and `logs/`. Audiences trust a deck more when it admits mistakes.
34. **Next steps.** A bigger window (more days of tape), longer Transformer training, and a fee-aware objective.
35. **Thanks / questions.** Team portraits again, plus a QR code or link to the write-up.
36. **Appendix marker.** Backup slides go after this: confusion matrices, the per-class table and the ops log.

---

## Part 2: Design guide

### 2.1 What "looks AI-made" and how to avoid it

These are the patterns people now read as "someone prompted this". Avoid them.

| AI tell | Do this instead |
|---|---|
| Emoji or icon on every bullet (🚀 📊 ✅) | No icons. A plain dash or no bullet at all. |
| Titles like "Unlocking Alpha: A Deep Dive into Microstructure" | Say the finding: "Toxic flow is easy to spot and still loses money." |
| Rows of three identical rounded cards with an icon, a heading and two lines | Uneven layouts. One big figure and a short caption beats three cards. |
| Purple/blue gradients, glowing blobs, abstract "AI" backgrounds | White background. Color only where it carries meaning. |
| Every slide has the same density and the same number of bullets | Vary it: some slides are one number, some are a dense table. |
| Generic summary words: "robust", "leverage", "seamless", "key insights", "comprehensive" | Specific numbers and plain verbs. "0.986 at 1024 events." |
| Em-dashes and colons everywhere in bullet text | Short sentences. |
| Perfectly balanced pros/cons, everything sounds positive | Lead with the negative result where it is the honest one (slide 24). |
| Stock photos or AI art unrelated to the content | Real figures from the study. The team portraits are the only illustration. |
| A "Conclusion" slide that repeats every section | Three lines, max. |

Things that make a deck read as made by people:
- **Real figures from your own pipeline.** Ugly but real beats pretty but generic. Re-export them so they're readable (2.5 below).
- **Hand annotations.** A red circle around the cell that matters, or an arrow with "this is the GRU". The reference deck does this well on its perp slides; keep that.
- **One or two jokes or memes, no more.** The reference deck's Homer slide works. Six of them stops working.
- **Inside details.** Dates, the actual VM size, "we stopped training at hour 9 because the budget ran out".
- **Uneven layouts.** Text on the left with a figure bleeding off the right edge, or a single giant number. Not everything centered.
- **Your voice in the titles.** Write them the way you'd say them out loud.

### 2.2 Overleaf / Beamer setup

Avoid the stock Beamer themes (Madrid, Warsaw, Berlin). Their blue bars and navigation bullets are the LaTeX equivalent of a template look. Start from the default theme and strip it down:

```latex
\documentclass[aspectratio=169,11pt]{beamer}
\usepackage[sfdefault,lining]{FiraSans}
\usepackage{newtxsf}            % math that matches Fira
\usepackage{booktabs,graphicx,xcolor,tikz}

\definecolor{ink}{HTML}{1B1F24}
\definecolor{orange}{HTML}{BF5700}   % UT burnt orange
\definecolor{slate}{HTML}{333F48}
\definecolor{mute}{HTML}{6B7178}
\definecolor{rule}{HTML}{E6E8EB}

\setbeamertemplate{navigation symbols}{}
\setbeamercolor{normal text}{fg=ink,bg=white}
\setbeamercolor{frametitle}{fg=ink}
\setbeamercolor{structure}{fg=orange}
\setbeamerfont{frametitle}{size=\Large,series=\bfseries}
\setbeamertemplate{frametitle}{\vspace{0.6em}\insertframetitle\par\vspace{0.2em}}
\setbeamertemplate{itemize item}{\color{orange}\textbullet}
\setbeamertemplate{itemize subitem}{\color{mute}--}
\setbeamertemplate{footline}{\hfill\color{mute}\scriptsize\insertframenumber\hspace{1.2em}\vspace{0.8em}}
```

- **16:9** to match the reference deck and projectors.
- **Fira Sans** ships with Overleaf. It's clean without being Helvetica or Calibri, which is part of what makes the reference deck look like a default Google Slides deck.
- **Slide numbers** bottom right, small, gray. Nothing else in the footer: no logo on every slide, no section name.
- **Logo** only on the title and closing slides.

### 2.3 Color

| Role | Color | Hex |
|---|---|---|
| Text | Ink | `#1B1F24` |
| Highlight (one thing per slide) | Burnt orange | `#BF5700` |
| Secondary series | Slate | `#333F48` |
| Everything else in charts | Gray | `#9CADB7` / `#D6D2C4` |
| Captions and axis labels | Muted | `#6B7178` |

Rules:
- **Orange means "look here".** At most one orange thing per slide: the winning model, the key number, the circled cell.
- Everything you aren't talking about is gray.
- No colored slide backgrounds except section dividers. Those may be full burnt orange with white text, and that's the only place the deck goes dark.

### 2.4 Type and layout

- **Title:** one line, bold, left-aligned, about 24 pt. If it wraps, shorten it.
- **Body:** at least 16 pt. If text won't fit at 16 pt, it belongs in the speaker notes.
- **At most 4 bullets** per slide, ideally one line each. The reference deck's "Measuring the Flow" and "Key Takeaways" slides are too dense; split them.
- **Margins:** keep about 0.6 in clear on every side. Don't let figures touch the edge unless you mean them to bleed off it.
- **Left-align** text. Center only the single big number on a "headline number" slide.
- **Same title position on every slide.** The reference deck's titles drift (compare "Backtesting:" with "Backtesting"). Beamer fixes this automatically if you don't override it.

### 2.5 Figures

- **Re-export from Python with the deck style:** `plt.style.use("tools/ucf.mplstyle")` in this folder, then save as PDF (vector) so Overleaf scales it without blur.
- **Remove the chart's own title** if the slide title says the same thing. The reference deck has double titles on most chart slides ("Backtesting:" above "BTC: position sized by...").
- **Label lines directly** instead of using a legend box.
- **One highlight:** the series you're talking about in orange, the rest gray. `tools/make_figures.py` shows a before/after pair (`figures/guide/bad_chart.pdf` vs `good_chart.pdf`) using the R² numbers.
- **Readable at the back of the room:** axis labels of at least 9 pt at final size. Many of the study's existing `report/*.png` files have small labels; re-plot the ones you show full-screen.
- **Never screenshot a terminal table.** The reference deck's black terminal-screenshot tables are the biggest single upgrade available. Rebuild them with `booktabs`:

```latex
\begin{tabular}{lrrrr}
\toprule
Model & 0.1s & 0.5s & 1s & 10s \\
\midrule
LightGBM, handcrafted  & 0.232 & 0.152 & 0.114 & 0.015 \\
Transformer, 75 min    & 0.233 & 0.154 & 0.116 & 0.018 \\
Transformer, 75+90 min & 0.244 & 0.158 & 0.119 & 0.017 \\
\textcolor{orange}{\textbf{GRU, 60 min}} & \textbf{0.262} & \textbf{0.164} & \textbf{0.123} & \textbf{0.018} \\
\bottomrule
\end{tabular}
```

  No vertical lines, three horizontal rules, numbers right-aligned to the same decimal places.

### 2.6 Equations

Keep the reference deck's habit of putting one big equation on its own when it matters (its Δ and Γ slides look good). Use `\[ \]` with `\large`, nothing else on the slide but a one-line plain-English gloss underneath.

### 2.7 Team portraits

- **Where:** the title slide (a small row), the team slide (2), and the closing slide (35). Nowhere else.
- **Current files:** `team/<name>.png`. These are filtered versions of the club-site headshots (a burnt-orange duotone with a circular crop, made by `tools/make_portraits.py`), not AI images. They're a consistent placeholder.
- **If you make AI portraits:**
  - Generate all six with **one prompt and one style**, so they look like a set. Mismatched styles are the biggest AI tell for portraits.
  - Use an illustrated style (flat vector, linocut, or ink-and-wash in burnt orange and ink), not photoreal. Photoreal AI faces of real people look uncanny and can read as fake photos.
  - **Check with each person before using their AI likeness.**
  - Save them as `team/ai/<name>.png`, square, then swap the path in the `\includegraphics` call.
  - Suggested prompt: *"Flat illustrated portrait of [person], head and shoulders, front facing, linocut style, two-tone burnt orange (#BF5700) and near-black ink on cream paper, simple background, no text"*, giving the tool their headshot as the reference image.
- **Circle-crop** all of them at the same size, with names in 12 pt underneath. Roles go in gray below the names.

### 2.8 Final checklist

- [ ] Every title is a sentence that states the finding (except section dividers).
- [ ] At most one orange highlight per slide.
- [ ] No terminal screenshots. All tables use booktabs.
- [ ] No chart has a duplicate title.
- [ ] No body text below 16 pt or chart text below 9 pt.
- [ ] No emoji, no icon rows, no gradients.
- [ ] Slide 24 ("it doesn't pay") is the loudest slide in its section.
- [ ] Each number on a slide matches `SLIDES.md` / `report/*/summary.json`.
- [ ] Every team member has approved their portrait.
- [ ] Backup slides come after the appendix marker.
- [ ] One full run-through, timed, done out loud.
