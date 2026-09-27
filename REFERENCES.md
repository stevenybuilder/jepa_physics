# References — V-JEPA physics take-home

## 0. Take-home spec
- **Repo:** https://anonymous.4open.science/r/vjepa-physics-takehome-4E00/README.md (also `DATA.md`)
- Behind a Cloudflare check, so `curl` and WebFetch get a 403. In a real browser session, the site's API serves every file.
  README.md, DATA.md, all three manifests, and a sample `metadata.json` and `video.mp4` from each dataset were read on
  26 Sep 2026. The API path is `/api/repo/vjepa-physics-takehome-4E00/file/<path>`.
- **Local copy:** `vjepa-physics-takehome-4E00/` (README, DATA.md, data/, 25 MB). Checked 26 Sep 2026:
  - All 4,572 manifest rows have their video and metadata files; none are missing, empty or corrupt.
  - Each dataset has 64 label values. Speed and acceleration have exactly 24 clips per value; direction has 23–24.
  - Sample videos decode to 16 frames at 256×256, 24 fps.
- **Model:** frozen V-JEPA 2 ViT-L/16, `facebook/vjepa2-vitl-fpc64-256`. The reference implementation is `vjepa2_vit_large`
  in `facebookresearch/vjepa2`.
- **Data:** a single blue disk on a dark background; 256×256 px, 16 frames, 24 fps; each clip is about 4 KB.
  Layout: `data/<var>/manifest.jsonl` plus `videos/scene_XXXX/{video.mp4,metadata.json}`.

  | Dataset | Target | Clips | Values |
  |---|---|---:|---|
  | `direction` | `theta_degrees` | 1,500 | 64 equally spaced in [0°, 360°) |
  | `speed` | `magnitude` / `speed_mps` | 1,536 | 64 from 0.25 to 4.0 m/s |
  | `acceleration` | `magnitude` / `acceleration_mps2` | 1,536 | 64 from 0.25 to 10.0 m/s² |

  Metadata fields: `primary_label`, `theta_degrees`, `motion`, `speed_mps`, `acceleration_mps2`, `start_position_xy_m`,
  `fps`, `frames`. In the sample clips, direction clips use a fixed 1.0 m/s speed and acceleration clips start at speed 0.
- **Rules from the README**
  - Probes, layer choice, nullspace construction and splines must never touch the final held-out test clips.
  - Document how activations are extracted and pooled.
  - Keep all derived artifacts outside `data/`, and do not modify the supplied data.
- **Part 1:** reproduce Joseph et al.: layer-wise probes, iterative nullspace probing, and multi-probe subspace
  steering on held-out data.
- **Part 2:** apply manifold/spline steering (Wurgaft et al.) to speed, acceleration and direction, then compare it with Part 1.
- **Deliverable:** a talk of about 15 minutes, then open discussion. AI tools are allowed, but you must own every choice.

---

## 1. Joseph et al. (2026). *Interpreting Physics in Video World Models.* arXiv:2602.07050 (the Part 1 source)
Sonia Joseph, Quentin Garrido, Randall Balestriero, Matthew Kowal, Thomas Fel, Shahab Bakhtiari, Blake Richards†, Mike Rabbat† (Meta FAIR / Mila).
https://arxiv.org/abs/2602.07050

**Question.** Do video encoders represent physical variables in factorized form, like a physics engine, or in a distributed, task-specific way?

**Models.** V-JEPA 2 (L/H/G) and VideoMAE V2.

**Methods to reproduce**
- **Layer-wise probes** on the residual stream at every layer. Mainly linear probes on mean-pooled space-time tokens, plus
  attentive-MLP probes that keep patch structure (Appendix B).
- **Physics Emergence Zone.** Probe accuracy jumps sharply about one-third of the way through the network, peaks just
  after, and falls toward the output. This holds across model scales. IntPhys was used for possible-vs-impossible.
- **Motion variables** on a synthetic toy ball: speed and acceleration decode from early layers, while direction becomes
  decodable only at the Emergence Zone.
- **Direction geometry.** Direction-selective MLP units form a ring-shaped population code with sinusoidal tuning.
  Steering only the unit-circle subspace fails, so direction lives in a higher-dimensional space.
- **Iterative orthogonal probing (INLP-like, App. C.11).** Train a probe, orthogonalize its direction, retrain on the
  residual, and repeat until chance. Accuracy follows a sawtooth pattern, consistent with paired sin/cos features.
  Subspace sizes across layers 0–23: direction 14–136, speed 16–31, IntPhys 1–15 (Table 3).
- **Multi-probe subspace steering (App. C.12)**
  - Stack the K probe weights `W_k ∈ R^{2×d}`, which predict `[sin θ, cos θ]`, and take a QR decomposition to get the basis `V ∈ R^{d×2K}`.
  - Steer in three steps: `c = Vᵀx`, `x⊥ = x − Vc`; solve least squares for `c*` so that all probes predict θ*; then `x* = Vc* + x⊥`.
  - Held-out protocol: 70/30 split (240/103 videos), 25 steering probes (fit until R² < 0.1), and an evaluation probe
    trained only on test activations (R² = 0.99).
- **Circuit.** Spatiotemporally local attention heads inside the Emergence Zone support both the IntPhys and direction
  tasks. Ablating them hurts both, while ImageNet performance is spared.

**Claim.** The models do not use clean, factorized physics variables. Physics variables need many tens of directions to steer.

---

## 2. Wurgaft, Rager, Kowal et al. (2026). *Manifold Steering Reveals the Shared Geometry of Neural Network Representation and Behavior.* arXiv:2605.05115 (the Part 2 source, Goodfire)
Daniel Wurgaft*, Can Rager*, Matthew Kowal*, … Thomas Fel†, Noah D. Goodman†, Ekdeep Singh Lubana† (Goodfire / Stanford / UCL / Northeastern / Harvard / Technion).
https://arxiv.org/abs/2605.05115 · **Code:** https://github.com/goodfire-ai/causalab/tree/manifold_steering

**Question.** Does geometric structure in activations causally shape behavior? The paper argues the core problem of
steering is "finding the right geometry, not the right direction".

**Method**
- **Activation manifold M_h**
  - Run PCA on the activations down to 64 dimensions.
  - Compute concept centroids, the mean activation for each label value.
  - Fit a spline that passes through every centroid:
    - natural cubic spline for sequential variables;
    - periodic cubic spline for cyclic variables, using the intrinsic coordinate `θ = atan2(PC2, PC1)`, found without labels;
    - thin-plate spline (TPS, kernel `r² log r`) for 2D graphs or grids, with ghost points to close periodic dimensions.
  - The orthogonal complement of the 64-dim PCA space is kept unchanged during interventions (App. A.3, A.6).
- **Behavior manifold M_y.** Map each output distribution into Hellinger space (`p ↦ √p`), fit a spline there, and square
  decoded points to get distributions back (App. A.4).
- **Manifold steering.** Move along M_h, using K = 50 waypoints per path. Compare with **linear steering**, which cuts
  through off-manifold regions and produces unnatural outputs.
- **Pullback steering.** Optimize a path in activation space (L-BFGS) so that the behavior follows a geodesic on M_y.
  The recovered path traces the curvature of M_h, measured by intrinsic R². Example: pullback vs linear R² of 0.75 ± 0.04 vs 0.32 ± 0.05.
- **Three explicit geometries** (Euclidean, density/manifold, pullback) are compared in one framework.

**Tasks**
- Llama 3.1 8B, layer 28: weekdays and months (cyclic), letters and ages (sequential), and in-context graph tasks.
- **Video world model:** a *recurrent world model trained on Mountain Car* (the "visual world model" section).
  This is **not V-JEPA**, so applying manifold steering to V-JEPA 2 physics variables is the new part of Part 2.

**Porting notes for Part 2**
- Direction is cyclic, so use a periodic spline. The unsupervised `atan2(PC2, PC1)` coordinate is a good check against the ring code in Joseph et al.
- Speed and acceleration are sequential, so use natural cubic splines. Check whether knot spacing is non-uniform (for
  example, log-like spacing for speed).
- V-JEPA has no output distribution, so there is no ready-made M_y. The behavior readout has to be built, for example from:
  - a held-out probe fit on a disjoint split;
  - probes on the predictor's predicted tokens;
  - decoded future disk position.

---

## 2b. V-JEPA 2 code and checkpoint (the model under study)
- **Official repo:** https://github.com/facebookresearch/vjepa2 (MIT; a few modules are Apache 2.0)
  - Paper: Assran et al. (2025), *V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning*,
    arXiv:2506.09985. This ID was checked against the repo's citation.
  - Torch Hub: `torch.hub.load('facebookresearch/vjepa2', 'vjepa2_vit_large')` returns **(encoder, predictor)**
    (`src/hub/backbones.py`). The matching preprocessing is `vjepa2_preprocessor`.
  - Raw checkpoint: `https://dl.fbaipublicfiles.com/vjepa2/vitl.pt` (300M params, 256 px). It holds both the `encoder` and `predictor` state dicts.
  - V-JEPA 2-AC (the action-conditioned predictor) is only released for ViT-g, so it is not relevant here.
- **Hugging Face checkpoint:** https://huggingface.co/facebook/vjepa2-vitl-fpc64-256 (MIT, safetensors)
  - `AutoModel` loads `VJEPA2Model` and `AutoVideoProcessor` loads `VJEPA2VideoProcessor`. Install a recent `transformers`.
  - Encoder: 24 layers, hidden size 1024, 16 heads, patch 16, tubelet 2.
  - Predictor: 12 layers, hidden size 384, 12 heads, 10 mask tokens.
  - Config `frames_per_clip: 64`. Positions are encoded with RoPE (`VJEPA2RopeAttention`), not a fixed-size table, so 16-frame input
    should run. A 16-frame clip gives 8×16×16 = 2,048 tokens. **Verify with a forward pass before extracting.**
  - `model.get_vision_features(**inputs)` returns only the encoder output. `model(..., output_hidden_states=True)` gives the
    per-layer residual stream.
  - The predictor is reached through `model(pixel_values_videos, context_mask=[...], target_mask=[...])`: set the context
    tokens and the target tokens to predict, then read `predictor_output`. Passing `skip_predictor=True` skips it.
  - **Preprocessing trap:** the processor resizes the shortest edge to 292, then center-crops to 256.
    - On these 256 px clips that crops about 6% of the frame. A disk near the border can be clipped or pushed out of view.
    - Disk position and pixel speed also get scaled by 292/256, about 1.14×.
    - Consider `do_resize=False, do_center_crop=False` so only the rescale step and the ImageNet mean/std normalization run.
      Record whichever choice you make.
  - Normalization: ImageNet mean (0.485, 0.456, 0.406), std (0.229, 0.224, 0.225); rescale by 1/255.

---

## 3. Background references the two papers build on
- Ravfogel et al. (2020). *Null It Out: Guarding Protected Attributes by Iterative Nullspace Projection* (INLP). arXiv:2004.07667
- Elazar et al. (2021). *Amnesic Probing.* arXiv:2006.00995
- Hewitt & Liang (2019). *Designing and Interpreting Probes with Control Tasks.* arXiv:1909.03368
- Engels et al. (2024). *Not All Language Model Features Are Linear* (circular features). arXiv:2405.14860
- Riochet et al. (2021). *IntPhys 2019.* · Yi et al. (2020). *CLEVRER.*

The arXiv IDs in section 3 were written from memory and have not been checked here. Verify them before they go on slides.

---

## 4. Prior project: jepa_steering (results archive)
- **Code:** https://github.com/stevenybuilder/jepa_steering (public). Activation steering of the JEPA-WM predictor and
  CEM planning in latent space. Summaries used for the paper are in `paper/data/*.json`, and each one points to GCS objects.
- **Google Cloud Storage:** `gs://rgt-jepa-archive-2026/` (gcloud account stevenybusiness@gmail.com, project `project-flash-490419`)
  - `mechanism-20260913/` holds the layer pilot, steered CEM, planned prefix, CEM expansion, controlled geometry and action counterfactual runs.
  - `rep_geometry_transcoder/lcfm-replication-20260914-v1/` holds the LCFM replication episodes.
  - Also in the bucket: `fresh-campaign-20260912-v2/`, `fast_outerloop/`, `laptop-disk-cleanup-20260919/`, and numbered instance folders.
  - Checked 26 Sep 2026: all 406 objects referenced in `paper/data` exist, and a sample `.tgz` reads back as valid gzip.
- **Google Drive** (rclone remote `gdrive:`): `Research-Archives/JEPA-WM/`, 963 objects, 53.0 GiB.
  - It holds the completed core panel (960 episodes), live runs from 2026-09-08, the Vast storage-release records,
    the laptop offload from 2026-09-12, and more.
  - Checked 26 Sep 2026: files list and read back.
  - Warning: this rclone remote uses rclone's shared client_id, which is being retired during 2026. Create your own
    client_id to keep access.
