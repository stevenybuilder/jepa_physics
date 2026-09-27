# Sonia Physics — Model Internals, Steering, and Fine-Grained Control

## Working thesis

The interesting research thread here is broader than activation steering.

Across Sonia Joseph's **physics-emergence work**, the **Manifold Steering** paper, her **“World models and interpretability are two sides of the same coin”** essay, and our **JEPA-WM steering project**, there is a common question:

> Can we move from merely *reading* human-interpretable structure out of a representation to **fine-grained causal control of a model through its internals**?

A useful distinction is:

- **human-concept-aligned structure** — a human-defined variable such as direction, speed, success, or deception can be decoded from activations;
- **causal manipulability** — changing the corresponding activation structure predictably changes downstream computation or behavior;
- **fine-grained control** — we can manipulate the variable continuously, selectively, compositionally, and robustly while preserving unrelated state;
- **model-native coordinate discovery** — the coordinates are discovered from the model's own dynamics rather than being imposed by human semantic labels.

These are progressively stronger claims. Much interpretability work establishes the first, some reaches the second, and reliable fine-grained control remains largely a future direction.

---

# 1. Sonia's broader research program

## “World models and interpretability are two sides of the same coin”

Sonia's June 2026 essay proposes an **Internal World Model** framing.

Her central claim is that a world model should not necessarily be thought of as a separate generative module or even specifically as a JEPA-style latent predictor. Instead, the world model can be thought of as the **internal state variables and dynamics learned by a foundation model** — effectively the model's internal causal structure.

She proposes two reformulations of interpretability:

### Interpretability as causal discovery

Rather than merely asking:

> What features can we decode from this network?

the goal becomes:

> What latent variables causally generate the model's predictions and actions, and how do those variables interact?

That is a significantly stronger research goal than ordinary probing.

### Interpretability as white-box evaluation

Instead of evaluating only the output, inspect the internal variables and circuitry that produced it.

The motivating examples range from language-model deception or planning to embodied hazards such as a robot interacting dangerously with its environment.

The desired endpoint is something like an **internal reasoning trace**:

\[
z_1, z_2, \ldots, z_k
\rightarrow
\text{interactions through circuitry}
\rightarrow
\text{prediction/action}.
\]

The critical question is not merely whether a variable is represented, but whether it is **recruited correctly in context**.

---

## The “simulator dials” idea

The most relevant part of the essay for our purposes is the simulator analogy.

Suppose a model develops something corresponding to velocity. If we can manipulate that internal representation and obtain the corresponding counterfactual future prediction, the model begins to look less like a black-box predictor and more like a simulator with learned internal state variables.

The aspiration is essentially:

\[
\text{learned latent variable}
\quad \longrightarrow \quad
\text{controllable simulator dial}.
\]

Examples might be:

- direction;
- velocity;
- acceleration;
- object state;
- contact;
- latent task state;
- possibly higher-level variables such as intent or plan.

This is much more ambitious than “find a probe with high accuracy.”

It requires **fine-grained control**.

A useful conceptual hierarchy is:

\[
\text{decodable}
\rightarrow
\text{geometrically structured}
\rightarrow
\text{causally manipulable}
\rightarrow
\text{continuously controllable}
\rightarrow
\text{compositionally controllable}
\rightarrow
\text{reliably changes downstream behavior}.
\]

That hierarchy should remain central to how we think about this line of work.

---

## Cognitive maps and model-native variables

Sonia's essay also asks whether foundation models form internal cognitive maps or simulator-like latent structures rather than merely using stimulus-response strategies.

Importantly, she explicitly treats this as an **open question** and notes that some JEPA findings cut against the strongest version.

The strongest form of the hypothesis would be:

> the network has discovered latent variables that correspond to meaningful causal structure in the world, and those variables interact in a simulator-like way.

The weaker form is:

> human-relevant variables are recoverable from distributed representations and can sometimes be causally manipulated.

Those should not be conflated.

---

# 2. The physics-emergence paper

## What the paper does well

**Interpreting Physics in Video World Models** is a highly controlled attempt to study small slices of physical representation rather than immediately asking whether a world model can be steered to solve an end-to-end robotics task.

The paper studies concepts such as:

- speed;
- acceleration;
- motion direction;
- where these quantities become decodable across depth;
- their dimensionality and geometry;
- whether intervention on their representations changes internal model behavior.

Its central empirical findings include:

- a mid-network **Physics Emergence Zone** where physical variables become substantially more accessible;
- scalar quantities such as speed and acceleration becoming available relatively early;
- direction becoming especially accessible around the emergence zone;
- direction exhibiting a **circular population structure**;
- direction being represented in a **high-dimensional distributed subspace**, rather than one clean scalar feature;
- simple manipulation of the visually obvious low-dimensional circle being insufficient;
- coordinated multi-feature intervention being required for stronger causal control.

One particularly important conceptual result is the distinction between something being:

> **readable**

and something being:

> **writable**.

A low-dimensional projection can make direction visually obvious without being the true causal locus through which the network can be controlled.

That is a valuable methodological lesson.

---

# 3. The methodological critique of the physics paper

The paper begins with **human-defined physical variables**.

For example:

\[
(h_i, \theta_i)
\]

where \(\theta_i\) might be ground-truth motion direction.

The workflow is roughly:

\[
\text{human-defined variable}
\rightarrow
\text{find decodable activation structure}
\rightarrow
\text{characterize geometry}
\rightarrow
\text{intervene}.
\]

This is scientifically legitimate. It can establish that the representation contains structure aligned with a physically meaningful variable and that some of that structure is causally manipulable.

But it does **not by itself establish** that “direction” is one of the model's privileged native state variables.

A high-dimensional representation \(h\) can contain enough information for:

\[
\theta \approx f(h)
\]

without the model internally implementing a clean autonomous variable \(\theta\).

Even successful intervention only strengthens the causal claim:

> this human-defined geometric structure participates in computation.

It still does not necessarily prove:

> the network itself factorizes computation around this exact human variable.

This distinction matters whenever language like “the model's internal variables” or “simulator dials” is used.

---

# 4. Manifold Steering

## Core idea

**Manifold Steering Reveals the Shared Geometry of Neural Network Representation and Behavior** asks a more general geometric question.

Instead of assuming the useful intervention is a straight vector:

\[
h' = h + \alpha v,
\]

the paper fits curved activation trajectories / manifolds and asks whether interventions that follow the learned geometry produce more natural behavioral trajectories.

Conceptually:

\[
M_h
\leftrightarrow
M_y
\]

where:

- \(M_h\) is a manifold in activation space;
- \(M_y\) is corresponding structure in behavior/output space.

The headline claim is that steering **along the activation manifold** can produce behavioral trajectories that better follow the model's natural behavior manifold, whereas linear steering can cut through off-manifold regions.

This reframes steering from:

> Find the right direction.

to:

> Find the right geometry.

The paper includes language-model settings and also a video-world-model physical-dynamics setting.

---

## Why it connects naturally to Sonia's physics paper

The physics paper provides an obvious motivating example:

**direction is circular.**

If the physical concept has topology:

\[
\theta \in S^1,
\]

then a single Euclidean steering vector is already a questionable parameterization.

The natural next question is:

> If the model's representation of direction is curved or circular, should interventions follow that curvature rather than cutting across activation space?

That is exactly where manifold steering enters.

So the intellectual lineage is roughly:

### Physics paper

Discover:

> direction has structured, distributed, nontrivial geometry.

### Manifold Steering

Ask:

> can following that geometry provide better causal control?

### Sonia's blog

Generalize:

> perhaps interpretable latent variables can eventually become learned “dials” of an internal simulator.

These three pieces form a coherent research program.

---

# 5. But Manifold Steering does not automatically solve the ontology problem

A curved manifold can still be built around a **human-specified concept coordinate**.

Suppose we know the ground-truth angle:

\[
0^\circ, 30^\circ, 60^\circ, \ldots
\]

and fit corresponding activation centroids:

\[
\mu(0^\circ), \mu(30^\circ), \mu(60^\circ), \ldots
\]

followed by a spline through those points.

Finding a clean curve is interesting.

Successfully steering along that curve is more interesting.

But there are still several possible interpretations:

1. the model has an intrinsic direction variable;
2. the model uses a distributed representation whose geometry happens to preserve direction;
3. direction is one convenient human coordinate through a much richer latent state;
4. the fitted curve is a useful **control surface** without being a privileged model-native coordinate.

The distinction between **useful control geometry** and **native ontology** is crucial.

---

# 6. Human-concept-aligned structure + causal manipulability

This phrase is worth keeping as a central concept:

> **human-concept-aligned structure and causal manipulability**

It may actually be the strongest defensible near-term objective for this area.

We do not necessarily need to prove that a concept is the model's metaphysically “true” internal variable before useful work becomes possible.

If we can establish that a human-relevant concept has:

1. reliable representational structure;
2. causal influence;
3. controllable geometry;
4. held-out generalization;
5. minimal collateral effects;

then we may obtain a practical **white-box control interface** even if the underlying representation remains distributed.

This creates a potentially important future direction:

## Fine-grained latent control without requiring full ontology recovery

The goal would be to construct internal controls satisfying properties analogous to good simulator controls.

For a concept \(z\), an intervention operator:

\[
I(h, \Delta z)
\]

should ideally satisfy:

### Monotonicity

Increasing the requested change should produce a correspondingly larger behavioral change.

\[
\Delta z_1 < \Delta z_2
\Rightarrow
\Delta y_1 < \Delta y_2
\]

where appropriate.

### Locality

Changing \(z\) should preserve unrelated latent variables.

### Compositionality

If two variables can be independently controlled:

\[
I_{z_1}(I_{z_2}(h))
\]

should have predictable combined effects.

### Reversibility

Applying \(+\Delta\) followed by \(-\Delta\) should approximately restore the original state.

### Context robustness

The same semantic control should remain meaningful across different scenes and trajectories.

### Temporal persistence

In a predictive world model, the intended semantic edit should survive through future recurrent prediction rather than disappearing or changing meaning.

### Counterfactual validity

The resulting future should resemble a future the model would naturally produce under a genuine corresponding change in the physical world.

This is a much stronger objective than ordinary steering.

---

# 7. Relationship to our JEPA-WM steering project

Our JEPA project attacked a messier and more ambitious question:

> Can activation interventions inside a frozen JEPA world model modify predicted futures strongly and reliably enough to alter planning or task success?

That bundles together:

- representational geometry;
- predictor dynamics;
- visual state;
- action conditioning;
- temporal recurrence;
- planning objective sensitivity;
- CEM search dynamics;
- task success.

This made the project much harder to interpret.

A null end-to-end result could arise because:

- the chosen latent coordinate was wrong;
- the edit was not causally important;
- the semantic effect did not survive future prediction;
- the planner was insensitive to the changed forecast;
- random edits were equally effective;
- the intervention harmed unrelated state;
- the task metric was too coarse.

Sonia's physics work avoids much of that ambiguity by designing controlled tasks around individual hypotheses.

That is one of the strongest methodological lessons from her work.

---

# 8. How our JEPA geometry work differs

Several of our experiments are **less semantically supervised** than COAST or the physics-emergence experiments.

Rather than beginning with:

> “Find the representation of direction.”

we often begin with a controlled perturbation to the frozen model and ask:

> “How does the model's own internal state respond?”

For example, the local action-geometry experiment is closer to:

\[
a
\rightarrow
h(a).
\]

We:

1. perturb an action input;
2. record neighboring endogenous activation states;
3. omit a central activation;
4. compare linear and curved reconstruction;
5. intervene using those model-generated responses;
6. follow effects through forecasting and planning.

That is more **model-response-driven** than fitting an activation direction directly from semantic labels.

However, it is still not fully unsupervised coordinate discovery.

We choose:

- the input/action coordinate;
- perturbation radius;
- intervention site;
- fitting procedure;
- outcome metrics.

So we should not claim to have discovered the model's “true native coordinate system.”

Our own current writeup correctly states that the geometry tests do **not** establish a dense physical manifold or model-native physical coordinates.

---

# 9. JEPA-WM added a dimension that the simpler steering experiments largely avoid: temporal semantics

A major lesson from the JEPA steering project is that a successful instantaneous intervention is not necessarily a persistent semantic intervention.

An edit can:

\[
h_t \rightarrow h'_t
\]

produce the desired immediate prediction, but then the recurrent/predictive process can transform that change differently at:

\[
h_{t+1}, h_{t+2}, \ldots
\]

This creates a distinction between:

### Instantaneous controllability

Can we modify a representation right now?

and:

### Dynamical controllability

Does the intended meaning survive through the model's own temporal computation?

That matters enormously for world models.

A simulator “dial” is only useful if changing the dial at time \(t\) causes the model's subsequent dynamics to behave consistently with the corresponding counterfactual world.

This is where our JEPA project adds something conceptually important to Sonia's broader simulator framing.

---

# 10. A useful combined research ladder

The papers and our project can be organized into a progression:

## Stage 1 — Decodability

Can a human variable be recovered from activations?

Example:

\[
h \rightarrow \hat{\theta}.
\]

## Stage 2 — Geometric organization

Does the variable occupy structured geometry?

Examples:

- line;
- subspace;
- circle;
- curved manifold;
- distributed population code.

## Stage 3 — Causal manipulability

Can intervention on that geometry change downstream computation?

## Stage 4 — Fine-grained controllability

Can we specify:

\[
\theta \rightarrow \theta + 15^\circ
\]

rather than merely “push direction-ish information harder”?

## Stage 5 — Selectivity

Can direction change while preserving speed, object identity, scene state, etc.?

## Stage 6 — Compositional control

Can direction, speed, and acceleration be manipulated independently and jointly?

## Stage 7 — Dynamical consistency

Do those changes propagate through future prediction like genuine counterfactual physical states?

## Stage 8 — Downstream decision relevance

Does a planner or policy actually respond appropriately?

## Stage 9 — Native-coordinate discovery

Can we discover important coordinates **before assigning human semantics**?

This final step is substantially stronger than current supervised probing/steering paradigms.

---

# 11. The “Sonar” direction

The stronger version of model-native discovery would reverse the usual workflow.

Typical approach:

\[
\text{human concept}
\rightarrow
\text{search activations}
\rightarrow
\text{find aligned structure}.
\]

Model-native approach:

\[
\{h_t\}
\rightarrow
\text{discover intrinsic structure } z
\rightarrow
do(z_k + \delta)
\rightarrow
\text{observe causal consequences}
\rightarrow
\text{assign semantics afterward}.
\]

This is closer to the Sonar idea.

The objective is to identify coordinates using properties such as:

- stability;
- low-dimensional dynamical structure;
- causal influence;
- invariance across contexts;
- compositionality;
- persistence through temporal prediction;
- predictable counterfactual effects.

Only afterward would we ask whether one discovered coordinate corresponds to something humans call:

- velocity;
- direction;
- contact;
- goal;
- uncertainty;
- task phase;
- or something we did not already have a name for.

That would provide much stronger evidence of a **model-native ontology**.

---

# 12. How COAST fits into this

COAST is a useful comparison because it starts from a human-defined behavioral distinction:

\[
\text{success} / \text{failure}.
\]

It then finds activation geometry associated with those outcomes and intervenes on it.

That can demonstrate:

> a success-aligned internal control surface exists.

It does not necessarily demonstrate:

> “success” is one of the model's native internal state variables.

COAST is therefore strong on **behavioral relevance** but weaker on ontology.

Sonia's physics work is strong on **controlled semantic variables and representational analysis** but operates farther from complicated end-to-end robotics behavior.

Our JEPA work attempted to connect:

\[
\text{representation}
\rightarrow
\text{intervention}
\rightarrow
\text{forecast}
\rightarrow
\text{planner}
\rightarrow
\text{behavior},
\]

which is scientifically ambitious but correspondingly harder to isolate.

---

# 13. A useful three-way comparison

| Research direction | Variable chosen by | Main strength | Main limitation |
|---|---|---|---|
| Physics Emergence Zone | Human physical labels | Controlled characterization of physical representation | Human ontology is supplied before discovery |
| Manifold Steering | Human/task-defined geometry + learned activation structure | Tests whether curved representational geometry supports better intervention | Useful control geometry does not automatically equal native ontology |
| COAST | Human success/failure outcome | Strong connection to robot task success | “Success” is an externally defined aggregate concept |
| Our JEPA steering | Controlled model/input perturbations + fitted intervention structure | Follows intervention through prediction and planning dynamics | Still chooses perturbation coordinates and intervention sites; end-to-end attribution is difficult |
| Sonar-style discovery | Model dynamics first | Closest to native-coordinate discovery | Hardest problem; evaluation and identifiability remain open |

---

# 14. What Sonia's take-home is really testing

The open-ended assignment combining her physics setup with manifold steering is interesting because it sits directly between these ideas.

It asks whether concepts such as:

- speed;
- acceleration;
- circular direction;

can move from:

\[
\text{decodable structure}
\]

toward:

\[
\text{geometrically faithful causal control}.
\]

A weak solution would show:

> the spline reconstructs or visualizes the activation geometry nicely.

A stronger solution would show:

> moving along the spline changes the model in the predicted semantic direction on held-out examples.

A much stronger solution would test:

> whether the intervention behaves like a real simulator dial — continuous, selective, compositional, context-robust, and dynamically meaningful.

This suggests a useful lens for the take-home:

**Do not merely ask whether manifold steering works better than linear steering. Ask what degree of causal control the experiment actually establishes.**

---

# 15. Potential future experiments on fine-grained control

## A. Continuous dose-response

For a target variable such as direction:

\[
\Delta \theta
\in
\{-90^\circ,-60^\circ,-30^\circ,0,30^\circ,60^\circ,90^\circ\}.
\]

Measure whether the induced downstream change is monotonic and calibrated.

The important question is not binary success/failure but whether the internal variable behaves like a genuine **dial**.

---

## B. Orthogonality / collateral-damage test

Steer direction while measuring:

- speed;
- acceleration;
- object identity;
- appearance;
- unrelated latent statistics.

A good control coordinate should modify the intended factor while preserving others.

---

## C. Composition

Apply:

\[
I_{\text{direction}}(\Delta \theta)
\]

and:

\[
I_{\text{speed}}(\Delta v)
\]

separately and jointly.

Ask whether:

\[
I_{\text{speed}}(I_{\text{direction}}(h))
\]

produces approximately the counterfactual state corresponding to changing both physical quantities.

This gets much closer to factorized simulator-like control.

---

## D. Transport across context

Fit the intervention geometry in one set of trajectories and test it in:

- new scenes;
- new object appearances;
- new speeds;
- new directions;
- new motion histories.

A model-native variable should ideally transport better than a context-specific correlation.

---

## E. Temporal persistence

After intervention, roll the model forward.

Ask whether the intended semantic change:

- persists;
- decays;
- reverses;
- diffuses into other variables;
- changes meaning after recurrence.

This directly connects to the JEPA-WM findings.

---

## F. Counterfactual equivalence

The strongest controlled test may be:

1. create a real input-level counterfactual;
2. record the model trajectory;
3. start from the original input;
4. reproduce the counterfactual using only an internal intervention;
5. compare the resulting future latent trajectory.

Formally:

\[
F(x+\Delta x)
\stackrel{?}{\approx}
F_{do(z+\Delta z)}(x).
\]

If the internal intervention reproduces not only the next activation but the downstream trajectory, that is much stronger evidence that the intervened structure corresponds to a causal state variable.

---

# 16. Overall assessment

There is a coherent and genuinely interesting research progression here:

### Sonia's physics paper

> **Where and how are human-interpretable physical variables represented?**

### Manifold Steering

> **Does respecting the geometry of those representations enable better causal intervention?**

### Sonia's world-model essay

> **Could sufficiently interpretable and controllable latent variables become learned simulator “dials,” enabling white-box evaluation and scientific discovery?**

### Our JEPA-WM project

> **Do internal interventions survive predictive dynamics strongly enough to alter forecasting and planning, and what breaks when we attempt end-to-end control?**

### Sonar-style direction

> **Can we discover the model's important causal coordinates before deciding what they mean?**

The common endpoint is not merely interpretability.

It is:

> **discovering and controlling the causal state of learned models with enough precision that internal representations become usable scientific and engineering interfaces.**

The key methodological discipline is to distinguish:

\[
\text{human-concept alignment}
\neq
\text{causal manipulability}
\neq
\text{fine-grained control}
\neq
\text{model-native ontology}.
\]

Each is valuable.

But each requires stronger evidence than the previous one.

---

## References / context

- Sonia Joseph, **“World models and interpretability are two sides of the same coin”** (June 19, 2026):  
  https://www.soniajoseph.ai/world-models-and-interpretability-are-two-sides-of-the-same-coin-2/

- Sonia Joseph et al., **“Interpreting Physics in Video World Models”**, arXiv:2602.07050:  
  https://arxiv.org/abs/2602.07050

- Daniel Wurgaft et al., **“Manifold Steering Reveals the Shared Geometry of Neural Network Representation and Behavior”**, arXiv:2605.05115:  
  https://arxiv.org/abs/2605.05115

- Internal comparison: our `jepa_steering` experiments on frozen JEPA-WM prediction, action-conditioned intervention, geometry, temporal propagation, and planning.
