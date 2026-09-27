# Literature review: V-JEPA physics take-home (Parts 1 + 2)

27 Sep 2026. Seven parallel search passes: circular codes; causal geometry; manifold steering + held-out evaluation;
video world-model interpretability + Sonia's group; probing method; steering failure modes; representational geometry.
Sources: orx/alphaXiv (keyword, embedding, full text), Semantic Scholar citations of 2602.07050 (15 unique citers) and of
2605.05115 (21), OpenAlex, OpenReview, the goodfire-ai/causalab `manifold_steering` branch, soniajoseph.ai. About 60 papers
were read in the relevant sections. Numbers below come from fetched text; "(inferred)" marks anything else. Full list:
`papers.tsv`.

**Research question.** V-JEPA 2 carries direction, speed and acceleration. Which *geometry* do they live in, and does an
edit that follows that geometry change what the model computes downstream? Judge the edit on held-out values and by the
model's own later computation, not by a probe at the edited layer.

**Three facts from this search that change the plan.**
- **Sonia's group already published spline steering on V-JEPA 2-L.** Musa, …, **Joseph, Kowal**, Derpanis, arXiv 2609.01551
  (ICML'26 MechInterp workshop). It steers camera motion within a single clip. Evaluation is only the distance to that clip's
  own features, the same points the spline passes through. There are no held-out values, no downstream readout and no
  controls. The paper also lists an open question: per-clip V-JEPA 2 trajectories oscillate in PCA, and the authors say
  "We do not have a definitive explanation".
- **Sonia's "tens of dimensions" count has been attacked in print.** Jin et al., arXiv 2608.10566, cite her C.11 by name. They
  prove that the iterative-erasure count changes under invertible reparameterisations. On her exact protocol with a
  synthetic circular target, identity mixing gives K = 1 in 20/20 runs and a shear of a = 1 gives a mean K of 5.75 (range 3–8).
  A whitened metric returns rank 2 every time.
- **Her own stated open questions** (blog, 23 Feb 2026) are "whether more efficient steering methods can recover the same
  control with fewer probes" and that "the representation may be organized around a harmonic basis". Her June essay adds:
  "If we can steer that velocity representation and observe corresponding counterfactual changes in future predictions,
  then the model starts looking less like a statistical predictor and more like a simulator".

---

## 1. Ranked slam-dunk additions

Score = impact on Sonia (1–5) × feasibility with stored pooled activations plus one cheap GPU session (1–5). All items use
the Part 1 split; none reads `test` before the final pass. P = precedent, F = figure, C = control that makes it a result.

**1. "How many dimensions is direction?", as four separate estimands. [5 × 5 = 25]**
- P: Jin 2608.10566 (count is metric-dependent); RLACE 2201.12091 Prop. 4.1 (INLP removes the filter Σ⁻¹Σxz, the signal
  rides the pattern Σxz); LEACE 2306.03819 (edit rank = rank Σxz; INLP removed rank 360 vs LEACE 17 in their amnesic
  replication); Karkada 2602.15029 (symmetric similarity → Fourier PCs, amplitude falling in k); the rebuttal's own
  "copies" toy (60 sin/cos copies need 40 probes; OpenReview aijGVmEG9Y).
- F (per layer, direction vs speed): (a) literal K vs random-removal band; (b) whitened, residualised K and rank(I−T);
  (c) fresh ridge + MLP R² after rank-2 LEACE; (d) split-half DFT spectrum of the 64 centroids (k = 0…8) + participation
  ratio. Whitened K ≈ 1 with LEACE-2 at chance → the literal K is conditioning; mass at k ≥ 2 → curved ring with harmonics
  (her "harmonic basis" remark); LEACE-2 failing → her copies reading. Note cos θ ⟂ cos kθ on an even grid, so harmonics
  alone cannot make a long (sin, cos) sawtooth: the figure decides copies vs metric.
- C: a planted ring of known harmonic content and copy count through all four estimators (reviewer zUMQ asked for a
  synthetic sawtooth control); shuffled centroids; random-init ViT. CPU, minutes.

**2. "Fewer probes": low-rank steering bake-off at matched edit norm. [5 × 5 = 25]**
- P: Marks & Tegmark 2310.06824 (mass-mean more causal, 7/8); Im & Li 2502.02716 (mean-of-differences MSE-optimal);
  ITI 2306.03341 (probe 34.8% vs mass-mean 42.3%); AxBench 2501.17148 (probe AUROC 0.940 but steering 0.098 vs DiffMean
  0.239); Angular Steering 2510.26243. Answers her blog question verbatim.
- Arms: probe-QR least squares (N = 1…K); centroid transport x + μ(θ*) − μ(θ); rotation of the k = 1 plane by Δ (rank 2,
  norm-preserving); periodic spline in PCA-k; nearest-centroid snap.
- F: held-out-probe angle error vs edit rank at matched ‖Δ‖. The claim under test: rank 2–4 matches her 40-dim edit.
- C: random subspace of equal rank with its own solve; learned coefficients on a random basis; sham edit; an MLP evaluator
  on disjoint clips (a linear evaluator shares filter geometry with probe-QR, 2608.22985). Pooled, minutes.

**3. Held-out circular steering: error vs |Δθ|, long way round, chord collapse. [5 × 5 = 25]**
- P: Kantamneni & Tegmark App. C.2 (80/20 value split + shuffled helix); Gurnee & Tegmark 2310.02207 (block hold-out);
  Engels Fig. 7 ((r, θ) grid: output depends on angle, r only near 0); Heinzerling 2403.10381 (held-out entities;
  longitude Spearman ρ = 0.55 from wrap). Absence: Goodfire, 2609.01551 and GAGA are in-sample; causalab calls its own
  test "in-sample".
- Design: three disjoint clip sets (fit, readout, carriers); holdouts = every 4th angle and one contiguous 45° wedge (the
  spline-vs-line claim rests on the wedge).
- F: (a) wrapped error vs |Δθ| (15° bins) per item-2 arm; (b) 0°→180° via 90° vs via 270° (only a manifold expresses
  both); (c) readout radius ‖(ŝ, ĉ)‖ along the chord, predicted ≈ 0 at 180°.
- C: nearest-centroid snap (floor = half the value gap); spline on permuted centroid order; circular–circular correlation.

**4. The "simulator dial": V-JEPA 2 predictor as the behavioural readout. [5 × 4 = 20]**
- P: her essay (quoted above); Garrido 2502.11831 (L1 surprise, context-only encoding); 2609.15980 (recovery
  R = (w_edit − w_C)/(w_A − w_C) on the decoded future); 2606.29522 (one-layer transplant < 2% of all-layer patching;
  steering vectors 77× / 26× natural change; score on the discriminating subset); Othello-GPT 2210.13382.
- Method: render counterfactual twins (same start and speed, θ*); validate the renderer on 20 supplied clips. Encode
  frames 1–8 only: HF `VJEPA2Model` applies `context_mask` after encoding (`modeling_vjepa2.py` L695), so context tokens
  otherwise see the future. Broadcast pooled Δ to all tokens at L, propagate, predict future tubelets.
- F: probe-free R = ⟨ẑ_edit − ẑ_src, ẑ_tgt − ẑ_src⟩/‖ẑ_tgt − ẑ_src‖² and future-direction error (probe fit on real future
  encodings), vs arm and vs injection layer.
- C: unsteered predictor preserves direction first; random matched rank + norm; U⊥ edit; shuffled target; natural-scale
  norm. ~200 carriers × 6 arms × 4 layers, one session (inferred < 1 h on a 4090).

**5. Velocity-plane test: ring, cone, or linear (vx, vy)? [4 × 5 = 20]**
- P: Engels EVR (regress out nuisance, PCA the residual); Goodfire cylinder TPS with ghost points; Clock/Pizza
  2306.17844; Williams 2110.14739 shape metrics.
- F: ring radius vs speed per layer; Procrustes of centroids to (cos θ, sin θ) vs (v cos θ, v sin θ); speed readout along
  a chord steer (linear velocity plane predicts a cos(Δθ/2) drop); onset of (vx, vy) vs (sin θ, cos θ) (her Table 1 claim).
- C: held-out speeds; constant-speed vs accelerating subsets. Pooled.

**6. Propagation and closure depth in the encoder. [4 × 4 = 16]**
- P: Othello washout (L4 + all later layers: errors 0.12 vs 2.68 null); Hydra 2307.15771; Rushing & Nanda 2402.15390;
  2609.15980 closure depth; 2609.21787 (edits rotate out of the entry subspace).
- F: heatmap steer layer × read layer, probes refit per read layer; cosine to the rendered twin's activation at L + k.
- C: matched random; untrained ViT-L. Encoder-only passes, same session.

**7. Answer 2609.01551's open question: straightening and oscillation. [4 × 4 = 16]**
- P: Hénaff 2019 (curvature relative to pixels); ReStraV 2507.00583 (angle + step length); 2603.12231 (Balestriero,
  LeCun: JEPA straightens; not measured on V-JEPA 2); Antognini & Sohl-Dickstein 1806.08805 (Toeplitz similarity →
  sinusoidal PCs, so the oscillation may be a PCA artefact).
- F: curvature (V-JEPA − pixels) and d_{t+1}/d_t per layer on the 8-step `timepool`, constant vs accelerating vs static;
  per-clip PCA beside a random-walk null. C: static-disk renders (RoPE floor); shuffled time order.

**8. Table stakes: selectivity baselines.** Random-init ViT-L, pixels + frame differences, last-frame-only, time-reversed
clips (same occupancy, opposite direction), label permutation. CALIPER 2609.08250: raw pixels 0.996 vs V-JEPA 2 0.995 in a
clean fixed-camera scene; a random ViT within 0.02 R². [4 × 5, expected by any reviewer]

Order: items 1–3 and 5 on stored activations; items 4, 6, 7 batched into the GPU session.

---

## 2. Prior experiments to reuse vs genuine gaps

**Reuse (method, paper, what to copy)**
| Method | Paper | Copy |
|---|---|---|
| Held-out-value fit + shuffled-label control | Kantamneni & Tegmark 2502.00873 App. C.1–C.2 | 80/20 value split; permuted-order refit; held-out values projected between neighbours |
| Off-manifold (r, θ) grid | Engels 2405.14860 Fig. 7, App. K (EVR) | Radius × angle sweep in the k = 1 plane; regress out nuisance variables before PCA |
| Centroid substitution in PCA-k + random-k ablation | Gurnee et al. 2601.04480 | a − μ_orig + μ_c inside PCA-k; ablation split by readout type (specificity) |
| Spline, periodic coordinate, metrics | Goodfire 2605.05115 + causalab | `detect_periodic_dims`, periodic cubic, isometry/energy definitions; **not** its injection asymmetry |
| Matched-injection control | Oozeer et al. 2605.24942 Table 2 | Linear and spline both injected into PCA-64: weekday energy 0.1014 vs 0.1011 (the anchor's gap disappears) |
| Whitened/residualised erasure count | Jin 2608.10566 Alg. 1 | Metric-QR erasure, rank(I−T), rather than 2K |
| LEACE | Belrose 2306.03819 | Rank-2 closed-form erasure; random rank-matched projection as control |
| Recovery R, layer × gain grid | 2609.15980 | R on predictor outputs; closure depth; overshoot at high gain |
| Counterfactual-update selectivity, natural-scale norm | 2606.29522 | Score only where source and target futures differ; report ‖Δ‖/median natural change |
| Rowspace/nullspace split of an edit | Makelov 2311.17030 | Effect of the downstream-sensitive part vs the ignored part (46.7% → 13.5% in their MLP case) |
| Per-example steerability | Tan 2407.12404; Braun 2505.22637 | Histogram and anti-steerable fraction; cosine agreement of centroid differences predicts failure |
| Topology and label-free coordinate | Gardner 2022 Nature; Chaudhuri 2019 Nat Neurosci | Ripser H1 bar vs shuffle nulls; unsupervised ring coordinate vs θ |
| Predictor surprise | Garrido 2502.11831 | Context-only encoding; L1 surprise of a steered context against real target-direction futures |

**Differentiation map (mechanism × objective × evaluation)**
| Cell | Status | Occupant |
|---|---|---|
| Spline steering, V-JEPA 2, within-clip camera motion, NN-distance eval | OCCUPIED | 2609.01551 (Sonia's group) |
| Manifold steering of a *population* variable (direction/speed) in a video SSL encoder | EMPTY | — |
| Any manifold steering evaluated on **held-out values** | EMPTY | anchor, 2609.01551 and GAGA are all in-sample |
| Steering a JEPA encoder read through its **predictor** | EMPTY | Garrido uses the predictor only unedited; Alam 2605.24322 and Joseph read at the edited layer |
| Steering in a video model judged against a **rendered counterfactual input** | EMPTY | LLM work can't do this (2604.09839: steered states have no preimage) |
| Critique of the INLP count | PARTIAL | 2608.10566 (theory + synthetic + contact stress tests); nothing on real direction activations with 64 angles |
| Harmonic (k ≥ 2) analysis of motion direction in video transformers | EMPTY | only CNN/MT: 2605.11718 (axial component), theory: Karkada |
| Pixel-relative straightening of V-JEPA 2 | EMPTY (imminent) | 2603.12231 measured DINOv2 and its own models |
| Closure depth in a pretrained SSL video encoder | EMPTY | 2609.15980 covers generative DiTs only |
| Decoded-behaviour steering in video generators | PARTIAL | 2609.15980; Alam (circular same-layer readout) |

**Top three genuine gaps**
1. **Predictor-read steering against rendered counterfactual twins.** No paper judges an activation edit in V-JEPA (or any
   JEPA) by the predictor's forecast, and none compares a steered state with the real activation of the re-rendered target
   clip. The data is synthetic, so the counterfactual input can simply be rendered. Doable in one GPU session.
2. **Held-out-value manifold steering, stratified by |Δθ|, with a long-way-round test.** Goodfire, 2609.01551 and GAGA all
   fit and evaluate on the same values. With 64 angles, a wedge hold-out is possible here, and the chord-through-centre
   prediction is parametric in |Δθ|.
3. **A constructive replacement for the "tens of dimensions" count.** Jin shows the count is not a dimension but offers no
   estimate on real circular-motion activations. Four estimators (literal, whitened, LEACE-2, centroid spectrum) on stored
   activations settle whether direction is copies, a metric artefact, or a curved ring with harmonics.

---

## 3. Closest prior art, ranked by threat
1. **2609.01551 Musa, …, Joseph, Kowal, Derpanis.** V-JEPA 2-L/G + VideoMAE-v2, 25 probe points; camera-motion AUC 91.0
   (V-JEPA 2-L), IntPhys 2 near chance; per-clip cubic spline vs chord scored by NN distance to the clip's own features.
   Limits: in-sample, no downstream readout, no controls. Position: its population-level, held-out, causal follow-up.
2. **2608.10566 Jin et al.** Erasure count not affine-invariant; her literal protocol on synthetic circles; says steering
   and tuning evidence are *not* invalidated. Position: credit it, then be constructive (item 1).
3. **2605.05115 Goodfire.** Energy 0.34 vs 0.93, pullback R² 0.77 vs 0.42 (weekdays). Linear replaces the whole residual,
   manifold only PCA-64; Mountain Car "behaviour" is a softmax over the spline's own points (isometry r = 0.996).
4. **2605.24942 GAGA.** Matched PCA-64 injection: linear vs spline energy indistinguishable; spline undefined off centroids.
5. **2609.15980 (Ziming Liu).** Video DiT, decoded-future recovery; top-4 writes 87.8% vs 92.4% full edit; closure depth.
   Generative model with a trained shortcut, not an SSL encoder.
6. **2605.24322 Alam.** CAV steering in VideoMAE read by the same layer-5 probe (circular); flip rate saturates at 0.25.
7. **2602.07050 C.12.** Eval probe fit on the clips then steered; 8 angles, one target; 82.9° → 11.9° at 20 probes (the
   main text's "< 0.5°" is steering across all probe directions); never propagated.

---

## 4. Per-topic notes (id — finding → what it changes)

**(1) Circular and periodic codes**
- 2405.14860 Engels — label-free circles; x* = mean + WᵀP⁺(circle(α′) − mean); no held-out α → EVR before ring PCA; copy the (r, θ) grid.
- 2502.00873 Kantamneni & Tegmark — FFT over values finds T = 2, 5, 10, 100; helix < PCA on 3/6 tasks → FFT the 64 centroids; copy the C-appendix controls.
- 2602.15029 Karkada — circulant Gram → Fourier PCs; PC1 vs PC3 Lissajous → a PCA ring is expected for *any* smooth symmetric code; the harmonic decay (tuning width) is the informative number; atan2(PC2, PC1) sees k = 1 only.
- 2601.04480 Gurnee et al. — count = rippled 1-D manifold in 6-D; ripples = optimal low-dim projection of a curved curve → many dimensions for a 1-D variable is expected, not "distributed variables".
- 2605.01148 Feucht, …, Fel (Goodfire) — month circles exist but Llama computes via base-10 Fourier addition → structure ≠ computation; ask which band the predictor uses.
- 2301.05217 / 2306.17844 / 2406.03445 — frequency-restricted losses; Clock vs Pizza; low vs high band → Fourier-band ablation of the ring, read by the predictor.
- 2605.11718 — video CNN MT maps with axial (180°) tuning ≈ 0.3 of the main peak → fit k = 1…3 per unit (her Eq. 4 is k = 1 only).
- 2510.26243 / 2608.30986 / 2609.10658 — norm-preserving rotations, binary behaviours only → k = 1 plane rotation as the minimal nonlinear arm.
- Chaudhuri 2019; Gardner 2022 — persistent homology + label-free ring/torus recovery vs shuffle nulls → certify the ring without labelled atan2.
- 1411.5908 Lenc & Vedaldi — equivariance φ(gx) ≈ M_g φ(x) → steer by Δ should commute with the rendered rotated twin.

**(2) Multi-dimensional features, LRH, causal evidence**
- 2311.03658 / 2406.01506 Park — binary/categorical LRH → a ring is outside it; frame as "the LRH describes the subspace, not the moves".
- 2408.10920 Csordás — linear DAS IIA 0.00–0.01 vs 0.83–0.89 for a magnitude edit → if speed steering fails, try a norm edit first.
- 2311.17030 Makelov — dormant-pathway illusion for supervised subspaces (46.7% → 13.5%) → split her probe-QR edit by downstream sensitivity.
- 2606.29522 Shih — "used" requires a later computation to consume it → selectivity on the discriminating subset; natural-scale norm.
- 2604.09839 Mishra — additive steering leaves the input-reachable set → compare with the rendered twin's activation.
- 2606.00926 — probe row-space ablation leaves the variable refittable (1.000 → 1.000) → refit + matched random for every ablation.
- 2609.18080 / 2604.22128 — probe-aligned vs behaviour-driving features overlap ≈ 12%; decodable subspaces can be inert → rank dims by predictor sensitivity too.
- 2301.04709 / 2303.02536 / 2305.08809 — interchange-intervention accuracy → swapping spline coordinates between clips, read by the predictor, is the formal item 4.

**(3) Held-out steering evaluation**
- 2501.17148 AxBench — pick on one half, evaluate on the other; harmonic mean of scores → tune α on calibration carriers.
- 2605.17231 FishBack — off-target KL at *matched* concept change → compare at matched effect, not matched α.
- 2602.02315 Sarfati — probe transfer decays with distance; linear steering shifts σ → off-target = steer direction, read speed (and reverse).
- 2410.17245 Pres — informative baselines, in-distribution contexts → no-op and random edits on the same carriers.
- 2411.02385 Kang — video generators generalise case-based → interior vs extrapolated speeds as separate rows.

**(4) Manifold and nonlinear steering beyond Goodfire**
- 2603.09313 Curveball — kernel-PCA path; distortion ratio d_geo/d_Euc on a kNN graph → free per-layer curvature diagnostic.
- 2410.23054 Linear-AcT — per-neuron 1-D transport → cheap value-to-value arm. (2410.04962 is "Activation Scaling", not AcT.)
- 2402.09631 / 2411.09003 — affine steering/editing → a stronger linear baseline than a pure shift.
- 2505.18230 Béthune — density-based metric → kNN-geodesic path as a label-free arm.
- 2307.12868 Park — pullback-Jacobian singular vectors as edit directions (the only metric → intervention precedent) → optional; costly on ViT-L.

**(5) Video world-model interpretability**
- 2502.11831 Garrido — predictor surprise, context-only encoding; random-init at chance → item 4 readout.
- 2506.09985 V-JEPA 2 — predictor ViT 22M, width 384, depth 12; anticipation probe on predictor outputs → precedent for probing predictions.
- 2606.09646 Punzo — attentive probes + temporal-shuffle control → add frame-shuffled clips.
- 2603.14482 V-JEPA 2.1 — context tokens act "similarly to register tokens"; ADE20K 22.2 mIoU → test background-token pools too.
- 2609.22788 — Physion: V-JEPA 2 73.2% vs humans 74.2%, 26.4% disagreement. 2605.15618 — latent prediction encodes the arrow of time → time reversal is a real control.

**(6) Probing method**
- 1909.03368 Hewitt & Liang — control tasks; degenerate for direction (class identity *is* direction) → say so; use random-init and pixel baselines.
- 2207.04153 Kumar — erasure damages correlated features → collateral curve (speed, position, start) per erasure round.
- 2506.10178 — linear probes understate MIM models → single-query attentive probe as the mean-pool upper bound.
- 2310.06824 / 2502.02716 / 2606.13720 — mass-mean beats probe direction; nullspace projection collapses between clusters, flipping moves into the target → report "erase" and "transport" separately.

**(7) Sonia and coauthors: what they value**
- Reviews 4/4/5/4 asked for operational definitions, a sawtooth control, practical use. The rebuttal added 14 models, a
  sigmoid emergence criterion (inflection ≤ 50% depth; encoders 28 ± 4%) and surgical fine-tuning; the AC asked for
  narrower claims. Prisma 2504.19475; Into the Rabbit Hull 2510.08638 (Fel, Kowal, Balestriero, Joseph); Kowal VTCD
  2401.10831; Kowal and Fel are on the Goodfire paper. → Neuroscience framing, formal criteria, controlled stimuli,
  causal tests beyond decodability.

**(11) Representational geometry → intervention**
- Gao & Ganguli PR; Jazayeri & Ostojic 2107.04084 — intrinsic vs embedding dimension → PR of centroids vs INLP K.
- Stringer 2019 — a smooth d-dim code needs α > 1 + 2/d (α > 3 here); α-ReQ 2202.05808 (Richards) → descriptive only (sparse 64-angle grid).
- Chung, Lee & Sompolinsky 1710.06487; Cohen 2020 — manifold capacity → optional appendix explaining the probe peak.
- Ansuini 1905.12784; Recanatesi 1906.00443 — hunchback ID profile → ID ≈ 3 at the peak layer would be a clean figure.
- Hénaff 2019/2021; Harrington ICLR'23; Toosi & Issa 2308.13870; Niu 2411.01777 — task-dependent straightening → curvature relative to pixels; low curvature predicts a small spline gap.
- Arvanitidis 1710.11379; Shao 1711.08014 — pullback geodesics → V-JEPA has no decoder; the predictor is the analogue.
- Sadtler 2014 — within-manifold BCI perturbations learnable, outside not → framing for spline vs chord-in-PCA-k vs complement edits.
- CEBRA 2204.00673; MARBLE 2304.03376 — label-guided embeddings impose topology → never ring evidence.

---

## 5. What works and what does not in steering (point 8)
| Approach | Strongest support | Strongest contradiction | Lesson here |
|---|---|---|---|
| Mean-difference (ActAdd 2308.10248, CAA 2312.06681) | AxBench: DiffMean best *detector* (AUROC 0.942); ITI mass-mean 42.3% vs 30.5% baseline | AxBench steering 0.239 vs prompting 0.894; Tan: up to ~50% anti-steerable | The right linear baseline; always show the per-clip distribution |
| Probe-direction | Joseph C.12: 82.9° → 11.9° (same-layer readout) | ITI 34.8% (probe) vs 42.3% (mass-mean); AxBench probe steering 0.098 | Probe weights are separating filters, not causal patterns |
| INLP/LEACE erasure | LEACE provably removes all linear information at minimal rank | Hydra: downstream compensation; 2606.00926: refit recovers; 2608.10566: count not invariant | Erasure measures linear guardedness, not use; refit downstream |
| Multi-probe subspace (Joseph) | Needs about 20 probes; 1–5 give MAE > 50° | Makelov illusion; a 40-dim span covers most linear readouts, so success is partly built in | Needs a non-linear or downstream evaluator |
| SAE features | Golden Gate demo (blog) | AxBench SAE steering 0.165 < DiffMean; SAEs tile manifolds (2509.02565) | Skip for a 3-variable ring |
| Spline/manifold | Goodfire energy 2.8× lower | Matched injection erases the gap (2605.24942); in-sample only; Mountain Car behaviour tautological | Match the injection, hold out values, read downstream |
| Geodesic/pullback | GAGA, FishBack (1.8–3.6× lower off-target KL at matched change) | Needs a decoder or output metric; costly | The predictor is the only output metric V-JEPA has |
| Random-direction null | ITI: random ≈ no-op (31.2% vs 30.5%) | Real-activation directions move outputs faster than random (2409.15019, 2409.17113) | Use random edits matched in norm *and* "toward another real clip" |

---

## 6. Where Part 2 should go (point 9)
- **Most informative experiment for the cyclic variable.** Held-out-wedge steering stratified by |Δθ|, run with matched
  injection, read at a later layer or the predictor, with the radius dip plotted beside it (items 3 + 4). The prediction is
  parametric: no gap below about 45°, and the largest gap at 180°. A flat gap after matched injection is itself the
  GAGA-style finding.
- **Most informative experiment for the scalar.** Leave-one-value-out reconstruction by line vs cubic on speed centroids,
  plus a check of knot spacing. If the speed curve is straight and evenly spaced, the answer is "no spline advantage", and
  the interesting object becomes the (θ, speed) cone, fitted with a TPS over a cylinder (item 5). Acceleration equals mean
  speed here, so run it as a replication of speed, not as a separate variable.
- **Known dead ends:** SAE analysis of the ring (SAEs tile manifolds, 2509.02565); CEBRA-style label-guided embeddings as
  ring evidence; NN distance to the spline's own knots (2609.01551); "behaviour" built from the steered layer's centroids
  (Mountain Car); subspace spline vs full-replacement linear (GAGA); same-layer probe flips (Alam); Physion/natural video as
  a fix (her rebuttal: spurious cues); pullback-Jacobian directions on ViT-L (cost); DAS-optimised 1-D subspaces without
  a rowspace check (Makelov).
---

## 7. Pitfalls and what not to do (points d and 10)
1. **Same-readout evaluation** (Alam; C.12; 2608.22985, where 15.4% of a steering vector's squared norm retains 96.3% of its
   effect). Fix: a readout disjoint in clips *and* in space: later layer, predictor, rendered twin.
2. **Evaluation probe trained on the steered clips** (C.12). Fix: three disjoint clip sets; report the evaluation probe's
   out-of-fold R².
3. **Leaky future prediction:** the HF `context_mask` is applied after encoding. Fix: encode the context frames only.
4. **Pooling artefacts:** high-norm register-like tokens (Darcet 2309.16588: about 10× norm, about 2% of tokens, most often
   over uniform background) and register-like context tokens in V-JEPA 2 (2603.14482). Fix: a token-norm histogram;
   mean-pool vs norm-clipped vs disk-pool.
5. **Position/retinotopy confound:** end position ∝ direction × displacement, and 3D RoPE. Fix: a time-reversal control, a
   start-position split (train x < 0, test x > 0), and a last-frame-only baseline.
6. **Pixel-decodable stimuli** (CALIPER). Fix: item 8 on every curve.
7. **INLP K reported as dimensionality** (2608.10566; RLACE). Fix: item 1; say "erasure count".
8. **Unmatched support or norm** (GAGA; 2606.29522: 77× natural scale). Fix: identical injection subspace; report
   ‖Δ‖/median natural change.
9. **Mean MAE only** (Tan). Fix: per-clip histograms, anti-steerable fraction, bootstrap CIs over clips, all targets rather
   than just 90°.
10. **Geometry read as computation** (2605.01148; Clock/Pizza). Fix: a ring claim needs a downstream consumer.
11. **Precision:** no published interpretability study found. In jepa_steering, BF16 flipped a curvature result. Fix: FP32
    with TF32 off, plus one BF16 repeat.
12. **Small n:** her steering used 103 clips, 8 angles and 1 target. Fix: 64 angles × all targets, with CIs.
13. **Rank/Spearman on angles** (longitude ρ = 0.55). Fix: wrapped error and circular–circular correlation.
14. **Overclaiming "physics understanding"** (the AC's explicit request). Fix: name the stimulus, the model and the
    16-frame horizon.

---

## 8. Papers table (core; full list with overlap/source/full_read in `papers.tsv`)
| id | title (short) | yr | why |
|---|---|---|---|
| 2602.07050 | Interpreting Physics in Video World Models (Joseph) | 26 | Part 1 anchor; C.11/C.12 protocol |
| 2605.05115 | Manifold Steering (Goodfire) | 26 | Part 2 anchor; metrics; injection asymmetry |
| 2609.01551 | What, Where, and How (Musa, …, Joseph, Kowal) | 26 | In-group spline steering on V-JEPA 2; open oscillation question |
| 2608.10566 | Iterative Erasure Count Is Not an Affine-Invariant Concept Dimension | 26 | Direct critique of the C.11 count |
| 2605.24942 | Riemannian-Manifold Steering (GAGA) | 26 | Matched injection erases the linear-vs-spline gap |
| 2609.15980 | Causal Writability in Video Models | 26 | Decoded-future recovery R; closure depth |
| 2606.29522 | When Does Activation Steering Change What a Model Computes From? | 26 | "Used" criterion; natural-scale norm |
| 2605.24322 | Causal Physics Steering via CAVs (Alam) | 26 | Video steering with a circular readout |
| 2609.08250 | CALIPER | 26 | Pixel/random-init tie in clean scenes |
| 2502.11831 | Intuitive physics emerges in V-JEPA (Garrido) | 25 | Predictor surprise protocol |
| 2506.09985 | V-JEPA 2 | 25 | Model, predictor, anticipation probe |
| 2603.14482 | V-JEPA 2.1 | 26 | Register-like context tokens |
| 2405.14860 | Not All LM Features Are One-Dimensionally Linear (Engels) | 24 | Circle discovery, (r, θ) grid, EVR |
| 2502.00873 | LMs Use Trigonometry to Do Addition (Kantamneni & Tegmark) | 25 | Held-out values; shuffled control |
| 2602.15029 | Symmetry shapes representation geometry (Karkada) | 26 | Fourier PCs; harmonics; Lissajous |
| 2601.04480 | When Models Manipulate Manifolds (Gurnee et al.) | 25 | Rippled manifold; centroid substitution; random-k |
| 2605.01148 | Arithmetic in the Wild (Feucht, …, Fel) | 26 | Structure ≠ computation |
| 2311.17030 | Subspace patching illusion (Makelov) | 23 | Dormant pathway; rowspace split |
| 2407.12404 | Reliability of steering vectors (Tan) | 24 | Per-example; anti-steerable |
| 2501.17148 | AxBench | 25 | Detection ≠ steering; split protocol |
| 2306.03341 | Inference-Time Intervention | 23 | Probe vs mass-mean; random null |
| 2310.06824 | Geometry of Truth (Marks & Tegmark) | 24 | Mass-mean more causal |
| 2502.02716 | Unified steering (Im & Li) | 25 | Mean-of-differences optimal |
| 2604.09839 | Steered activations are non-surjective | 26 | Off-reachable states; rendered twin |
| 2608.22985 | What Does Activation Steering Control? | 26 | Readout-shared success |
| 2306.03819 | LEACE | 23 | Rank-2 erasure |
| 2606.00926 | Refit the Probe | 26 | Ablation needs refit + matched random |
| 2307.15771 | Hydra effect | 23 | Downstream compensation |
| 2210.13382 | Othello-GPT | 22 | Downstream-judged edits; washout; random-init probe |
| 2408.10920 | Onion representations (Csordás) | 24 | Linear failure ≠ absence |
| 2406.03689 | World model implicit in a generative model (Vafa) | 24 | Probe success ≠ coherent model |
| 2309.16588 | ViTs Need Registers | 23 | High-norm tokens in pooling |
| 2605.11718 | Self-organized MT Direction Maps | 26 | Axial tuning component |
| 2603.12231 | Temporal Straightening for Latent Planning | 26 | JEPA straightens (not measured on V-JEPA 2) |
| 2507.00583 | ReStraV | 25 | Curvature + step metric |
| 1806.08805 | PCA of high-dimensional random walks | 18 | Oscillatory PCs are a null |
| 2510.08638 | Into the Rabbit Hull (Fel, …, Joseph) | 25 | Group's geometry lens |
| DOI 10.1038/s41593-019-0460-x | Head-direction attractor manifold (Chaudhuri) | 19 | Label-free ring recovery |
| DOI 10.1038/s41586-021-04268-7 | Toroidal grid-cell topology (Gardner) | 22 | Persistent-homology certification |
| DOI 10.1038/s41593-019-0377-4 | Perceptual straightening (Hénaff) | 19 | Pixel-relative curvature |

---

## 9. Open search directions and conclusion
- **Open searches:**
  - Venue versions (OpenReview) of 2608.10566 and 2605.24942.
  - Whether Sonia or Kowal have an unposted follow-up to 2609.01551. Check the MechInterp workshop poster and X.
  - Any V-JEPA 2 SAE or transcoder release through Prisma.
  - Precision-sensitivity studies in interpretability (none found).
- **Coverage:** high for LLM circular codes, steering evaluation and video-probing citers (all 15 S2 citers of 2602.07050
  triaged). Medium for neuroscience geometry. X/Bluesky were not searched directly; the blog and essay were read.
- **Conclusion:** the reproduction is standard. What makes this submission stand out:
  1. It replaces the erasure count with estimands that mean something (item 1).
  2. It answers Sonia's own "fewer probes" question (item 2).
  3. It evaluates on held-out angles against |Δθ| (item 3).
  4. It judges edits by the predictor against rendered counterfactual twins (item 4). Nobody has done this, and her essay
     asks for it.

  Present it as the causal, held-out follow-up to 2609.01551, and credit 2608.10566.
