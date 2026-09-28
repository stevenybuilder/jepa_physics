# Papers to mine for Part 2 (2026-09-28 15:05 ET)

Every arXiv id below was resolved through the arXiv API on 2026-09-28 (title + first author checked). "Used" = already in
REPORT.md; "planned" = cheap enough for tonight and not yet run; "idea" = worth a sentence in the talk, not a run.
Sources: lit_review.md §1–§9 (27 Sep), the sibling folder `~/Documents/jepa_steering` (its paper bib, README and
JEPA_STEERING_LESSONS.md), and one 10-minute pass on alphaXiv + arXiv for the neighbourhoods the 27 Sep review lacked
(sublayer self-repair, object binding, conserved-quantity steering).

## Slide citations (one slide, one line each)

1. Joseph et al. 2026, arXiv 2602.07050 — the Part 1 protocol (layer-wise probes, iterative nullspace count, probe-QR steering).
2. Wurgaft et al. (Goodfire) 2026, arXiv 2605.05115 — spline / manifold steering, energy and isometry metrics (Part 2 anchor).
3. Musa, …, Joseph, Kowal, Derpanis 2026, arXiv 2609.01551 — spline steering already run on V-JEPA 2-L (camera motion, in-sample); we add held-out values, controls and a downstream judge.
4. Engels et al. 2024, arXiv 2405.14860; Kantamneni & Tegmark 2025, arXiv 2502.00873 — circular features, the (r, θ) grid, held-out-value and shuffled-label controls.
5. Jin et al. 2026, arXiv 2608.10566 — the iterative-erasure count is not affine-invariant (why we report an erasure count, not a dimension).
6. McGrath et al. 2023, arXiv 2307.15771; Rushing & Nanda 2024, arXiv 2402.15390; Patrawala et al. 2026, arXiv 2609.07876 — self-repair / adjacent-layer correction: why edits at points 2–12 wash out and point-22 edits survive.
7. Wang et al. 2026, arXiv 2609.15980; Shih et al. 2026, arXiv 2606.29522 — recovery R on the model's own forecast, closure depth, natural-scale edit norm.
8. Mishra et al. 2026, arXiv 2604.09839 — steered activations have no input preimage, hence the rendered pixel twin as the yardstick.
(Optional ninth if there is room: Jaeger 2014, arXiv 1403.3369 + Miao et al. COAST 2026, arXiv 2605.17144 — multiplicative conceptor gates as the untested alternative to additive shifts.)

## Table

| Paper | arXiv id | Idea (one line) | How we use it / could still use it tonight | Status |
|---|---|---|---|---|
| Musa, …, Joseph, Kowal, Derpanis, *What, Where, and How* | 2609.01551 | Spline steering of camera motion inside V-JEPA 2-L/G and VideoMAE-v2; evaluation = distance to the same clip's features; open question: per-clip trajectories oscillate in PCA | Framed as the closest prior art: we add held-out values, matched-support chord, random-norm null, predictor readout, rendered twins. Talk: one sentence crediting it, one saying what it did not test | used |
| Engels et al., *Not All LM Features Are One-Dimensionally Linear* | 2405.14860 | Label-free circular features; edit x* = mean + WᵀP⁺(circle(α′) − mean); (r, θ) grid shows the output depends on angle, radius only near 0; EVR to regress out nuisance before ring PCA | Ring PCA and the radius/hollow analysis in §4.1; the radius-matched arm in §4.5. Tonight: none left | used |
| Kantamneni & Tegmark, *LMs Use Trigonometry to Do Addition* | 2502.00873 | Helix/circle fits with 80/20 value hold-out and a shuffled-order refit control; FFT over values finds periods | Held-out-value design (§4.2) and shuffled-target control (§4.5). Tonight: FFT of the 64 centroids to report harmonic k ≥ 2 power (15 min CPU, `p2_*_centroids`) | used / planned |
| Jaeger, *Controlling RNNs by Conceptors* | 1403.3369 | Conceptor C = R(R + α⁻²I)⁻¹ is a soft projector onto a state cloud's principal subspace; Boolean algebra (¬C = I − C, AND, OR) on subspaces; aperture α sets rank | A multiplicative "keep the ring subspace" gate is the one arm we never ran (all ours are additive). Caution from jepa_steering (JEPA_STEERING_LESSONS L150): conceptor edits stayed on-manifold and did nothing at that box's power. Only as a talk sentence tonight | idea |
| Miao et al., *COAST: Contrastive Conceptor Activation Steering* | 2605.17144 | Fit success and failure conceptors from rollouts, take C⁺ AND ¬C⁻, gate the residual stream multiplicatively in a frozen VLA; +20% sim / +40% real success (their abstract) | Recipe for a contrastive conceptor between θ_a clips and θ_b clips at point 22, gated per token, judged by the predictor. Same caution as above; ~45 min GPU if the box is up, otherwise a slide line "untested gate family" | idea |
| Postmus & Abreu, *Steering LLMs using Conceptors* | 2410.16314 | Conceptor steering beats additive mean-difference on LLM function-vector tasks | Second citation for the conceptor line; nothing to run | idea |
| McGrath et al., *The Hydra Effect* | 2307.15771 | Ablating one attention layer is compensated by later layers (self-repair); erasure at one site understates what the site does | Names the wash-out: point 2–12 edits read 82–88° four blocks later, point-25 MAE back to 4–7° (§4.5). Talk: "the Hydra effect on a video encoder" | used (interpretation) |
| Rushing & Nanda, *Explorations of Self-Repair* | 2402.15390 | Self-repair is imperfect and noisy; two mechanisms: final-LayerNorm scaling and sparse "anti-erasure" neurons | Tonight, cheap if per-block outputs are cached: per-block direct effect of Δ on the point-25 direction probe (dot of each block's output change with the probe weight) to see which blocks cancel the edit | planned |
| Patrawala et al., *LLM Layers Immediately Correct Each Other* | 2609.07876 | Adjacent transformer layers systematically counteract each other's contributions (TLCM), selectively per subspace, measured with the layer Jacobian; 5/7 model families | Predicts the point-12 failure: the next block rejects the edited subspace. Test: cosine between block-(L+1) output change and −Δ for edits at 12 vs 22. New (Sept 2026), not on any video model | idea / planned |
| Wang et al., *Causal Writability in Video Models* | 2609.15980 | A low-dimensional physics edit restores correct motion in a video DiT; recovery R on the decoded future; a sharp depth boundary ("closure") after which a fixed-strength edit no longer changes the video | Recovery R (dir 0.78–1.00 at point 22) and the closure-depth reading of points 2–12 vs 22 (§4.5) | used |
| Shih et al., *When Does Activation Steering Change What a Model Computes From?* | 2606.29522 | "Used" requires a later computation to consume the edit; report ‖Δ‖ relative to natural change; score only where futures differ | ‖Δ‖ ÷ twin change column, discriminating-target scoring (§4.5) | used |
| Mishra et al., *Steered LLM Activations are Non-Surjective* | 2604.09839 | Additive steering leaves the set of activations reachable from any input, almost surely | Justifies the rendered pixel twin as the only on-manifold reference; the projection of the edit on the twin's change (0.16–0.23 at point 25) is the number to quote | used |
| Chen et al., *Compact but Moving* | 2609.21787 | A rank-4 velocity correction leaves its entry subspace during rollout but is tracked by transporting the entry directions through the Jacobian chain | Explains why the edit's projection on the twin's point-25 change is small even when the forecast moves; tonight: quote as interpretation only | idea |
| Liu et al., *Low-Rank Dynamics-Effective Latent Carriers* | 2608.15156 | Learn low-rank carriers from counterfactual-minus-factual hidden differences; an affine map predicts carrier coordinates from the factual state and the requested edit | A "twin-difference" arm: Δ = mean(twin − carrier activation) at point 22 over probe clips, applied to test carriers. Mean-difference on real counterfactual pairs, the strongest linear baseline we have not run; ~30 min GPU | planned |
| Pham et al., *LEAP: Latent Energy Action Planning* | 2609.03294 | Optimise a whole action horizon through a frozen LeWM with an energy = terminal latent match + decoded-descriptor match; quasi-Newton refinement, projection to admissible range | Our pullback arm minimised 1 − cos on the forecast angle and overshot position (R 1.3–1.8). LEAP's second term says add a position-descriptor penalty; talk sentence, not a rerun | idea |
| Hong et al., *Steering Robustness into World Action Models* | 2607.14943 | Contrastive activation directions plus local linearity enable feedback steering (WA-LQR) in Cosmos-Policy / DiT4DiT; separability predicts steerability | Same reading as ours: probe R² does not predict steerability, forecast reach does. Cite as the world-action-model parallel | idea |
| Bao et al., *Correcting a learned physical invariant* | 2608.23526 | A frozen world model learns an energy-like invariant that drifts in rollout; projecting back to the level set reduces error, random constraints do not | Off-target check we lack: direction clips have fixed speed 1.0 m/s, so a direction edit should leave the speed readout on its level set. Speed probe exists; 15 min CPU on stored edited activations | planned |
| Michalkiewicz et al., *Foveated Probes Recover Localized Binding Information* | 2608.00726 | Global pooled readouts of frozen vision encoders lose localised information that an attention-pooled (foveated) readout recovers | Our edit and readout are mean-pooled over 2,048 tokens; a query-pooled readout of the forecast is the natural upgrade to the predictor probe. Limitation sentence tonight | idea |
| Huang et al., *Formalizing the Binding Problem* | 2606.03976 | Information-theoretic definition and probe for feature binding in ViTs | The single-disk stimulus has no binding problem; a two-disk clip would test whether a direction edit binds to the right object. Name as the open gap | idea (gap) |
| Gurnee et al., *When Models Manipulate Manifolds* | 2601.04480 | A 1-D variable lives on a rippled curve in a 6-D subspace; centroid substitution in PCA-k plus random-k ablation | Centroid-substitution arm and random-k control (§4.4) | used |
| Jin et al., *Iterative Erasure Count Is Not Affine-Invariant* | 2608.10566 | The INLP-style count changes under invertible reparameterisation; whitened metric returns rank 2 on a synthetic circle | Reported as an erasure count at fixed α, not a dimension (§3.2) | used |
| Oozeer et al., *Riemannian-Manifold Steering (GAGA)* | 2605.24942 | Matched injection into PCA-64 erases the linear-vs-spline energy gap | Matched-support chord arm (§4.4) | used |
| Makelov et al., *Interpretability illusion for subspace patching* | 2311.17030 | Supervised subspaces can act through dormant pathways; split an edit into downstream-sensitive rowspace and ignored nullspace | Split the probe-QR edit by predictor sensitivity (rank the 2K directions by forecast change). ~20 min if edited forecasts are cached | planned |

## Genuine gap still unaddressed (after this pass)

- **Sublayer attribution of the wash-out.** Nobody has shown which blocks of a video encoder cancel an activation edit
  (TLCM 2609.07876 and self-repair 2402.15390 are LLM-only). We have the phenomenon (points 2–12 vanish, 22 survives)
  but not the per-block ledger. EMPTY.
- **Binding of a steering edit in a multi-object scene.** All video steering work (2609.01551, 2605.24322, ours) uses
  a single moving thing. EMPTY.
- **Multiplicative (conceptor) gates vs additive shifts on a video encoder, read by the predictor.** COAST is VLA-only;
  jepa_steering's null result was under-powered. EMPTY.
- **Conserved-quantity off-target under a direction edit** (speed level set, 2608.23526's test). Cheap; PARTIAL only in
  the sense that Sarfati-style off-target reads are in lit_review §4, not run.
