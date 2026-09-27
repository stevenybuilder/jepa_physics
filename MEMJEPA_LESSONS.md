# Lessons from MemJEPA (ICLR 2027) for the V-JEPA physics take-home

Take-home: https://anonymous.4open.science/r/vjepa-physics-takehome-4E00 (README + DATA.md read 26 Sep 2026).
Task recap: frozen V-JEPA 2 ViT-L/16 (`facebook/vjepa2-vitl-fpc64-256`); three synthetic datasets (one blue disk, 16 frames,
24 fps, 256 px; direction 1,500 clips, speed 1,536, acceleration 1,536; 64 label values each). Part 1 reproduces
Joseph et al. (layer-wise probes, iterative nullspace probing, multi-probe subspace steering on held-out data). Part 2 applies
Goodfire's manifold/spline steering to speed, acceleration and direction and compares it with Part 1. Deliverable: ~15 min
talk, then open discussion. AI tools allowed; you must own every choice.

**Bottom line.** MemJEPA worked because of its controls (fixed-in-advance bars, a matched twin, a state swap, reported
negatives) and was hurt by sprawl, a late story and one late confound. The take-home is a small version of the question
MemJEPA answered: *the latent holds a variable, but does the model use it, and in what geometry?* Answer that at three rungs
(decodable, causally used, used by the world model's predictor) and you beat a standard reproduction.

---

## 1. The core insight that transfers: decodable is not used

MemJEPA's strongest evidence was a three-rung argument. Every rung maps onto the take-home.

| Rung | MemJEPA | Take-home equivalent |
|---|---|---|
| Decodable | Ridge probe reads the hidden fingertip from the belief, R² ≥ 0.92, vs ~0 from the frame latent | Layer-wise probes for direction, speed, acceleration |
| Present but unused | The standard RSSM and a clean-clip belief *could* remember, yet planning stayed at the memoryless floor (3.7 / 5.2 / 26.3%) | Linear info can survive in a subspace that later layers or the predictor never read; INLP removal may not change behaviour |
| Causally used | Swapping the carried state between paired episodes sent success from 93% to 7.7%; zeroing it raised the masked-window loss | Transplant one clip's speed coordinates into another; steer along probe subspace or spline |
| Used by the world model | The planner, not a probe, decided success | Steer the encoder latent, run V-JEPA 2's **predictor**, decode the predicted future: does the disk move at the steered speed? |

Also from MemJEPA: **a representation that fails a precondition closes the question.** On Memory Gym's Searing Spotlights a
probe on the frozen latent erred by 23.9 px against a 6 px bar, so no memory test was run. Same here: a variable that
does not beat a pixel baseline at any layer is not a V-JEPA finding.

## 2. Mech-interp methodology for this take-home

**Probing (Part 1a)**
- Baselines on every layer curve: raw pixels / frame differences (a disk on black makes speed nearly trivial), a
  randomly initialised ViT-L, and a shuffled-label control (Hewitt & Liang "selectivity").
- Linear and MLP probes both. We had linear and MLP probe tables in the appendix; the gap tells you whether the variable
  is linearly available or only nonlinearly present.
- Pooling is a hypothesis, not a detail. V-JEPA makes 2-frame tubelets: 8 × 16 × 16 tokens. Mean-pooling all tokens
  dilutes a small disk. Compare mean-pool vs disk-token pool (our read-site result, Fig. 5a, says this can decide the
  answer). Document hook placement (residual stream after block L).
- Metrics: MAE + R² for speed and acceleration; direction as (sin θ, cos θ), reported as circular MAE (R² on sin/cos
  can look fine while angle error is bad).

**Iterative nullspace probing (Part 1b)**
- Compare every INLP curve with removing *random* subspaces of the same rank. "Redundant" means accuracy survives far
  more removals than random removal would predict.
- INLP removes linear info only. After k rounds, fit an MLP probe: if it still decodes, the info is nonlinear, not gone.
- Fit each round's probe on train, score on validation; never refit on test.
- Behavioural amnesic test (Elazar et al.): project out the subspace and check whether the predictor's future changes.
  Probe accuracy dropping is rung 1; the predictor changing is rung 3.

**Subspace steering (Part 1c) and spline steering (Part 2)**
- **Never score steering with the probe you steered with.** This is our shared-scorer bug (a head fit on one arm scored
  every arm; it cost ~$400 and 3.5 h). Build the steering subspace on train clips; score with probes fit on a disjoint
  split, at a later layer or the final layer, and with the predictor readout.
- Held-out steering means: steering built on train clips, applied to test clips, toward **label values never used to fit
  the subspace or the spline knots** (hold out e.g. every 4th of the 64 values). That tests interpolation along the manifold.
- Report dose–response (steer magnitude vs decoded change), off-target effects (a 3 × 3 selectivity matrix: steer speed,
  measure speed / acceleration / direction), and activation-norm drift. Linear steering that "works" only by leaving the
  data manifold is a failure case, and that is exactly where spline steering should win.
- Geometry hypotheses to register *before* fitting:
  - direction lies on a closed curve (circular features, cf. Engels et al., "Not All Language Model Features Are Linear");
    the spline must be periodic, and linear steering through the centre should break;
  - speed may be compressed (log-like) spacing along its curve; test by comparing knot spacing against a linear fit;
  - acceleration is weakest: 16 frames = 0.67 s, so small accelerations barely move the disk. Check its correlation with
    mean speed and final displacement before believing any acceleration probe or steering result.
- Transplant/interchange (our state swap): take a source clip's coordinates in the speed subspace (or its position on the
  speed spline), write them into a target clip, and check the predictor's future follows the source's speed and the
  target's direction.

**World-model readout (the part most candidates will skip)**
- V-JEPA 2 is a world model: encoder + predictor. The HF `VJEPA2Model` returns `predictor_output` (verified, Section 8).
- Fit probes on *real* encoder tokens of later frames; apply them to the predictor's predicted tokens. Unsteered first
  (does the predictor preserve speed?), then steered. A steered speed that produces the right future displacement over
  time is behavioural evidence, and it answers the ICML reviewers' "practical implications" critique.

## 3. What we did well (keep)
- Protocols and bars fixed before looking; test sets read once; every number recounted from a results file.
- Matched controls (the same-recipe memoryless twin) and causal tests (swap, ablation ladder) carried the paper.
- Negatives reported alongside positives.
- Results archived with no agent attached, so two session outages lost nothing.
- Dead ends closed fast once a precondition failed (MIKASA, Memory Gym, Maze).
- A four-way final audit (citations, numbers, equations/figures, supplement) found 16 real errors; 0 hallucinated references.
- Prose modelled on one good paper (TD-JEPA); figures on one reference (LEAP Fig. 4).

## 4. What to improve
- **Sprawl.** Six-plus benchmark families in 10 days; the named designs existed 36 h before the deadline; the story was
  written in a crunch and read as "LLM slop". Pick the question on day 1.
- **Late confound.** The shared scorer head should have been caught by a pre-read audit of every learned component.
- **Underpowered checks.** An 8-episode gate at a 42% base rate passed by luck. Here: ~24 clips per label value; compute
  scene-clustered bootstrap CIs before claiming any difference.
- **Process overhead.** 1,786 ledger lines, 62 correction rows, internal words leaking into prose, shell chains that kept
  going after a failed assert (one false ledger row). For a take-home: one page of decisions, `set -euo pipefail`.
- **Agent overload.** ~13 live agents killed every lane for 95 min.
- **Stats and scope** (reviewer W1, W3, W5): one encoder, three seeds, intervals conditional on the model.
- **Redundant figures** until late. One claim per figure.

## 5. Execution: compute, agents, processing
- **One GPU is enough.** Extract once. Full tokens for 24 layers ≈ 460 GB; pooled variants ≈ 250 MB each. Store two
  pooled variants for all layers plus full tokens only at the one or two steering layers. Everything else runs on CPU.
- Check the model config for 16-frame input (the checkpoint is fpc64) and the exact preprocessing (resize, normalisation).
- Commit the split file (by label value and by scene) before the first probe number.
- One script produces every figure and every number on the slides from results files. Nothing typed by hand.
- Agents: 4–6 live, Opus for engineering, audits staggered. Agents read, build and audit; you own every decision and can
  explain every line.
- Freeze results at T−24 h; after that only slides.

## 6. Experiment design: hypothesis first, highest value first
- For each experiment write: the claim, the result that would falsify it, and the slide you would show either way. If no
  outcome changes the talk, skip it.
- Rank by value to the talk, not by ease. MemJEPA's most valuable runs were the causal ones (swap, ablation ladder), not
  more benchmarks.
- Suggested single Part 2 question: *Is each physical variable a straight direction or a curved manifold in V-JEPA's
  representation, and does steering along the manifold produce physically correct futures in the predictor where linear
  steering does not?*
- Before any read: list every learned component in the evaluation path (probes, splines, PCA, normalisers, decoders) and
  what data each was fit on.
- Timebox: Part 1 ≈ 25%, the Part 2 question ≈ 50%, talk + audit ≈ 25%. One sharp extension beats three shallow ones.

## 7. Agents and prompts to reuse
- **`neel-review`, `mats-eval`**: this take-home is mech interp; Neel Nanda's bar (does it look inside the model and answer
  a real "why") is the right one. Run on the plan and on the final deck.
- **`/iclr_agent` pattern → interviewer panel**: calibrate on the ICML reviews of Joseph et al. (synthetic data, few model
  families, limited evaluation, unclear practical use). Their predicted questions become backup slides.
- **`/pat`**: section-by-section review plus claim fidelity on the write-up or slide script.
- **`research-anti-sync`**: brutal critique of the Part 2 plan against real prior art.
- **`lit-review` / `arxiv-research-agent`**: first pass done (Section 8). Next: open the Goodfire code
  (goodfire-ai/causalab, branch manifold_steering) and check whether Joseph et al. released theirs.
- **`data-qa-power`**: confounds (acceleration vs speed, start position), leakage, power.
- **`provenance`**: every slide number traced to a file.
- **`compute-manager` / `gpu-experiment-ops`**: one box, qualified once, results archived automatically.
- **Final-audit pattern**: four parallel checks (numbers, figures, method claims, clean-clone reproduction).
- **Figures**: LEAP style (sans-serif, no bar outlines, larger text, legend on top), one claim per figure title.

## 8. Literature grounding (5-minute review, 26 Sep 2026)

VERIFIED = opened or seen in search results with a matching title. UNVERIFIED = seen but details not confirmed.

**The paper to reproduce: Joseph et al., "Interpreting Physics in Video World Models" (arXiv 2602.07050, ICML 2026).** VERIFIED.
- Models: V-JEPA 2 L/H/G and VideoMAE-v2-G. Data: Kubric rolling ball. Velocity set 392 videos (8 directions × 7 speeds ×
  7 starts); acceleration set 280. 16 frames at 24 fps; labels in px/frame.
- **The take-home differs:** 64 directions instead of 8, and labels in m/s. With 64 values the circle can be shown
  explicitly rather than inferred.
- Probes: linear probes on mean-pooled space-time patches, plus patch-preserving attentive-MLP probes; learning-rate and
  weight-decay sweep; 5-fold *grouped* CV, mean ± SD.
- Nullspace probing: fit, orthogonalise, refit on the residual **until chance**. Direction needs ~40–50 features (up to ~80
  near the output) with a "sawtooth" curve consistent with sin/cos pairs; speed is lower-dimensional.
- Geometry: direction forms a unit-circle population code in a "Physics Emergence Zone" at ~1/3 depth; speed and
  acceleration are readable from early layers. Direction and physics subspaces are nearly orthogonal (69–83°).
- Multi-probe steering: stack probe weights, QR-orthogonalise, steer. Evaluated on a 30% held-out split with a **separate
  evaluation probe**. 20 probes: ~12° error to the target angle vs 82.9° baseline.
- Attention ablation in the emergence zone: direction R² 0.97 → 0.14; IntPhys 78.3 → 61.7%; ImageNet unchanged.
- Its ICML reviewers (from the OpenReview printout read on 26 Sep): final scores 4/4/5/4; weaknesses were synthetic
  "toy ball" data, few model families, limited evaluation and unclear practical implications. Three reviewers raised
  their scores after a rebuttal that added experiments.
- Follow-up: Alam, "Causal Physics Steering in Video World Models via CAVs" (arXiv 2605.24322, CVPRW 2026). Steers
  VideoMAE with probe weights and evaluates by **behaviour** (IntPhys plausibility flips). VERIFIED.

**The method to extend: Wurgaft et al., "Manifold Steering Reveals the Shared Geometry of Neural Network Representation
and Behavior" (arXiv 2605.05115, Goodfire, 2026).** VERIFIED.
- Construction: PCA to 64 dims; one centroid per attribute value; a cubic spline through the centroids (thin-plate splines
  for 2-D sheets). For Mountain Car, a visual world model, they bin position and fit the spline through bin means.
- Steering: linear baseline interpolates straight between two activations. Manifold steering maps both ends to the
  spline's intrinsic coordinate, interpolates there, and maps back, so every step stays on the manifold.
  K = 50 waypoints for the LM (Llama 3.1 8B, layer 28), K = 20 for Mountain Car.
- Evaluation, three metrics to reuse:
  1. **Isometry:** do distances along the activation manifold match distances between output distributions?
     (r = 0.99 weekdays, 0.89 months.)
  2. **Energy:** how far the steered output trajectory strays from the behaviour manifold (integrated Bhattacharyya
     distance); ~2.8× lower than linear.
  3. **Pullback R²:** 0.77 manifold vs 0.42 linear.
- Code: github.com/goodfire-ai/causalab/tree/manifold_steering (URL seen in the paper, not opened).

**Probing and concept removal.** All VERIFIED.
- Hewitt & Liang 2019, control tasks and selectivity (arXiv 1909.03368). A probe is only meaningful relative to what it
  can learn on a control task.
- Belinkov 2022, "Probing Classifiers: Promises, Shortcomings, and Advances" (arXiv 2102.12452). Decodability is not use.
- Elazar et al. 2021, amnesic probing (arXiv 2006.00995). Remove a property, then measure the effect on behaviour.
- Ravfogel et al. 2022, RLACE (arXiv 2201.12091) and Kumar, Tan & Sharma 2022 (arXiv 2207.04153). Probe-based removal can
  leave the concept recoverable or destroy unrelated features. INLP guarantees only *linear* unrecoverability.

**Causal and steering evaluation.**
- Geiger et al., DAS / interchange interventions (arXiv 2303.02536). The formal version of our state swap. VERIFIED.
- Zhang & Nanda 2023 (arXiv 2309.16042) and Heimersheim & Nanda 2024 (arXiv 2404.15255). The choice of metric and
  corruption changes patching conclusions. VERIFIED.
- Tan et al. 2024 (arXiv 2407.12404, NeurIPS). Steering vectors are unreliable in and out of distribution. VERIFIED.
- "Steered LLM Activations are Non-Surjective" (arXiv 2604.09839). Steering produces activations the model never
  produces naturally. UNVERIFIED details.

**Geometry.**
- Engels et al. 2024, "Not All Language Model Features Are Linear" (arXiv 2405.14860). Circular weekday and month
  features, confirmed by intervention. VERIFIED.
- Kantamneni & Tegmark 2025 (numbers on a helix); Fourier features for numbers. UNVERIFIED ids.

**Video and world-model interpretability.**
- Garrido et al. 2025 (arXiv 2502.11831). Intuitive-physics understanding emerges in V-JEPA. VERIFIED.
- Li et al. 2023, Othello-GPT (arXiv 2210.13382), and Nanda et al. 2023 (arXiv 2309.00941). Linear world
  representations, steered by vector arithmetic, with the effect checked on the model's moves. VERIFIED.
- "How Do Video Foundation Models Encode Intuitive Physics?" (arXiv 2606.09646). UNVERIFIED.

**HF checkpoint.** VERIFIED from the transformers docs.
- `VJEPA2Model.forward(pixel_values_videos, context_mask, target_mask, skip_predictor)` returns
  `predictor_output.last_hidden_state`.
- Config: 24 encoder layers of width 1024; predictor of 12 layers × 384; tubelet 2; patch 16.
- `output_hidden_states=True` returns per-layer encoder states.
- **The predictor is available for a behavioural steering readout.**

---

## 9. What makes it a slam dunk

**What the bar is.** A borderline submission reproduces the plots. A strong one reproduces them with controls. A slam
dunk leaves the interviewers believing three things:
1. every claim is backed by a causal test they could not poke a hole in;
2. there is one new insight about how V-JEPA represents physics that neither paper states;
3. you own every choice.

Joseph et al. were criticised for limited evaluation and unclear practical implications. A slam dunk answers exactly
those two points.

| Component | Borderline | Strong | Slam dunk |
|---|---|---|---|
| Layer probes | R² per layer, one pooling | + grouped CV, sweep, MLP probe | + selectivity vs shuffled/control labels, pixel and random-init baselines, mean vs disk-token pooling |
| Nullspace probing | Curve until it "drops" | Iterate to chance | + random-removal baseline, MLP/LEACE check after removal, sawtooth read as sin/cos pairs |
| Subspace steering | Steer, score with the same probe | Separate eval probe, held-out clips (Joseph) | + held-out *label values*, dose–response, 3 × 3 selectivity matrix, norm drift |
| Spline steering | A spline through centroids, a demo | Spline vs linear on one metric | Wurgaft's three metrics, knots fit on train bins only, periodic spline for direction, log vs linear test for speed |
| World model | Absent | Mentioned as future work | Steered context → **predictor** → future matches a truly faster/rotated clip |
| Talk | Method walk-through | Findings-first | One question, three rungs of evidence, one honest negative, backup slides for the predicted questions |

**The ten elements, with why each matters.**

1. **A faithful reproduction, then the delta.** Match Joseph's protocol first: mean-pooled linear probes, grouped CV by
   start position, nullspace to chance, a separate evaluation probe for steering. *Then* show what 64 directions buy you:
   plot the 2-D sin/cos plane per layer and show the circle forming at the emergence zone.
   *Why:* interviewers trust extensions only after you show you can reproduce. (Joseph et al.)
2. **Selectivity, not accuracy.** On every layer curve, overlay:
   - a shuffled-label or control-task probe;
   - a raw-pixel or frame-difference probe;
   - a randomly initialised ViT-L.

   Claim "V-JEPA encodes speed at layer L" only where it beats all three.
   *Why:* a disk on black is nearly pixel-decodable, and probe capacity can create signal. (Hewitt & Liang; Belinkov)
3. **Nullspace probing that survives scrutiny.**
   - Iterate until chance.
   - Plot a random-subspace-removal curve of the same rank next to it.
   - After the last linear round, fit an MLP probe, or use LEACE, to show whether the variable is gone or only linearly
     hidden.
   - Read the direction sawtooth as sin/cos frequency pairs.

   *Why:* INLP guarantees only linear unrecoverability; "~40 dimensions" is meaningless without a random baseline.
   (Ravfogel 2022; Kumar et al. 2022)
4. **Held-out steering that tests generalisation.**
   - Build the steering subspace and fit splines on train clips and train label values.
   - Evaluate on held-out clips steered to held-out values, for example every 4th speed and every 4th direction; also
     try extrapolation beyond the fitted range.
   - Score with a separate evaluation probe, never the steering probe.

   *Why:* this is literally what the README's "meaningful held-out steering evaluation" asks for. (Joseph et al.)
5. **Dose–response and selectivity.**
   - Plot decoded change against steering magnitude.
   - Build a 3 × 3 matrix: steer speed, acceleration or direction; measure all three.
   - Report the activation norm along the steering path.
   - Check subspace angles first; Joseph reports 69–83° between direction and physics subspaces.

   *Why:* steering vectors are unreliable and can push activations off-distribution. (Tan et al. 2024; 2604.09839)
6. **Spline steering evaluated on Wurgaft's own terms.**
   - Fit centroids and splines on train bins only.
   - Define the "behaviour manifold" as the evaluation-probe readouts, or the predictor outputs, on real clips.
   - Report isometry, energy and pullback R² for spline vs linear at matched endpoints.

   *Why:* this makes the Part 2 comparison quantitative rather than a demo. (Wurgaft et al.)
7. **Geometry hypotheses stated before fitting.**
   - Direction lies on a closed curve, so linear interpolation between opposite angles collapses through the centre.
   - Speed may be log-spaced along its curve: test log-target vs linear-target probes and the spacing of the spline's knots.
   - If the speed manifold is effectively straight, say so: *"the spline gives no advantage for speed"* is a finding.

   *Why:* a stated, testable geometric claim is the "new insight". (Engels et al. 2024; Joseph et al.)
8. **The world-model readout (the differentiator).**
   - Run the predictor on the steered context (`predictor_output`).
   - Compare its predicted target embeddings with the *real* encoder embeddings of clips that truly have the steered
     speed or direction, by nearest-neighbour retrieval or an evaluation probe fit on real target tokens.
   - Do it unsteered first, to show the predictor preserves speed at all.

   *Why:* it answers "practical implications", joins mech interp to world models, and follows the behavioural-evaluation
   pattern of Wurgaft's Mountain Car, Alam's IntPhys flips and Othello-GPT. It is also your home ground.
9. **An interchange intervention.** Write a source clip's coordinates (in the speed subspace or on the speed spline)
   into a target clip. Check that the predictor's future takes the source's speed and keeps the target's direction.
   *Why:* this is the DAS-style causal test and the analogue of MemJEPA's state swap, which carried that paper. (Geiger et al.)
10. **One clean negative.** For example: steering fails outside the emergence zone, or acceleration cannot be steered
    because 0.67 s barely shows it. *Why:* it signals honesty and marks the method's boundary, and it becomes the
    limitations slide.

**15-minute talk skeleton (about 10 slides).**

| Time | Content |
|---|---|
| 0:00–1:00 | The question: V-JEPA holds physics variables, but in what geometry, and does its predictor use them? One-line answer. |
| 1:00–4:00 | Reproduction: layer curves with baselines, selectivity, the circle forming at the emergence zone. |
| 4:00–6:00 | Nullspace probing: dimensionality vs random removal, sawtooth read as frequencies. |
| 6:00–8:30 | Linear subspace steering: held-out values, dose–response, selectivity matrix. |
| 8:30–12:00 | Spline vs linear: geometry per variable, Wurgaft metrics, predictor readout, interchange result. |
| 12:00–13:30 | Failure cases and the negative. |
| 13:30–15:00 | What it means for world models, limitations, the next experiment. |

Backup slides: split protocol, pooling comparison, hook placement, MLP probes, seeds and CIs, per-variable spline plots.

**Questions to be ready for.**
- Why mean-pool, and why that layer?
- INLP removes only linear information, so what does "40 dimensions" mean?
- Isn't a spline through 64 centroids just interpolation?
- How is your held-out evaluation not leaking?
- Why should the predictor readout convince me more than a probe?
- What would change on natural video?

Practise answering each in under 45 seconds.

**Why this beats traditional candidates.** Most will stop at decodability plus same-probe steering, and few will touch
the predictor. You bring three things they usually lack:
- **Rigour habits:** fixed-in-advance protocols, matched controls, recounts from files.
- **Hands-on world-model experience:** JEPA predictors, latent planning, the state swap.
- **An audited agent workflow** that lets you run more controls in the same time.

The risk is that the same workflow produces code you cannot explain. Budget time to re-derive every component yourself.

---

## 10. Traps that make a submission borderline

Borderline means correct-looking but not trustworthy or not new: the interviewers cannot tell whether the result is real
or whether you understand it. Each trap is listed with how it shows up, why it hurts, and the fix.

1. **Probe accuracy with no controls.**
   - *Shows up as:* "Speed is decodable from layer 2 with R² 0.98."
   - *Why it hurts:* a disk on black is nearly pixel-decodable, and a flexible probe learns anything.
   - *Fix:* selectivity against control labels, pixel and random-init baselines. (Hewitt & Liang; Belinkov)
2. **Pooling that leaks or mismatches.**
   - *Shows up as:* probes on mean-pooled vectors while steering edits every token, with no stated mapping between the two.
   - *Why it hurts:* the steered quantity is not the probed quantity.
   - *Fix:* state the hook point and the pooling, compare mean vs disk-token pooling, and steer in the space you probed
     (or broadcast the pooled direction to every token and say so).
3. **Circular steering evaluation.**
   - *Shows up as:* steer along probe w, then report that probe w now reads the target.
   - *Why it hurts:* success is guaranteed by construction. This is the same failure as MemJEPA's shared scorer.
   - *Fix:* a separate evaluation probe on disjoint data (Joseph's own protocol), plus the predictor readout.
     (Kumar et al. 2022)
4. **"Held-out" that is a random split of the same generator.**
   - *Shows up as:* a 70/30 clip split where every label value appears in both halves.
   - *Why it hurts:* it tests memorisation of seen values.
   - *Fix:* hold out label values and start positions; add an extrapolation test. Choose layer, rank and knots on
     validation only, as DATA.md requires.
5. **Inflated INLP dimensionality, or claiming the variable is "removed".**
   - *Shows up as:* "Direction occupies 40 dimensions and is gone after removal."
   - *Why it hurts:* INLP removes only linear information; probe capacity and overfitting inflate the count.
   - *Fix:* iterate to chance, compare with random removal, check with an MLP or LEACE afterwards. (Ravfogel 2022; Kumar 2022)
6. **Norm blow-up reported as success.**
   - *Shows up as:* steering at ×10 magnitude "hits the target".
   - *Why it hurts:* the activation leaves the data distribution, so the readout is meaningless.
   - *Fix:* dose–response, norm drift along the path, off-target effects, and spline steering as the on-manifold
     comparison. (Tan et al. 2024; Wurgaft et al.)
7. **Direction treated as linear.**
   - *Shows up as:* MAE computed on raw degrees, a 64-class classifier with no notion of adjacency, or angles averaged
     directly.
   - *Why it hurts:* 359° and 1° become maximally wrong.
   - *Fix:* sin/cos targets, wrap-aware error, a periodic spline, and a demonstration that linear interpolation collapses
     through the centre. (Engels et al. 2024)
8. **One-layer conclusions, or copying the 8-direction findings.**
   - *Shows up as:* "Physics emerges at layer 8", from a single layer, or assuming the paper's sawtooth holds at 64
     directions.
   - *Why it hurts:* the emergence zone is a range, and the take-home data differs.
   - *Fix:* a full layer sweep with CIs, and re-checking the circle and sawtooth on your own data. (Joseph et al.)
9. **A spline that trivially interpolates.**
   - *Shows up as:* knots at all 64 centroids, evaluated on those same bins.
   - *Why it hurts:* the spline passes through every point by construction.
   - *Fix:* fit knots on train bins, evaluate at held-out bins, report Wurgaft's metrics, and compare against a straight
     line with the same endpoints. (Wurgaft et al.)
10. **Ignoring the acceleration confound.**
    - *Shows up as:* "Acceleration is linearly encoded" when the probe may read mean speed or final displacement.
    - *Why it hurts:* 16 frames is 0.67 s, so small accelerations are nearly invisible.
    - *Fix:* check correlation with speed and displacement, and report partial R² or matched-speed subsets.
11. **Overclaiming.**
    - *Shows up as:* "V-JEPA understands physics."
    - *Why it hurts:* it invites the same "toy data" critique Joseph's reviewers made.
    - *Fix:* state exactly what was measured (one disk, 16 frames, one encoder) and what would test generality.
12. **Scope creep.**
    - *Shows up as:* adding VideoMAE or natural video before the core question is answered.
    - *Why it hurts:* it spreads thin, which was MemJEPA's biggest cost.
    - *Fix:* list extra models as the next step, not the talk.
13. **Not owning the work.**
    - *Shows up as:* hesitating on "why this layer" or "how does the spline map back", or internal jargon on slides.
    - *Why it hurts:* the README says you are responsible for understanding everything, and the discussion tests it.
    - *Fix:* whiteboard-derive INLP, QR steering and spline steering; rehearse the six questions above.
14. **No limitations, or too many slides.**
    - *Shows up as:* 25 slides of plots and no failure case.
    - *Why it hurts:* it reads as a method dump.
    - *Fix:* about 10 slides, findings first, one negative, one limitations slide.
