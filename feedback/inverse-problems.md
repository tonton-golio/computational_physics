inverse-problems.md

**Suggested Edits – Inverse Problems Course (KoalaBrain)**  
**Goal:** Feynman tone (warm, story-driven, zero pomp, everyday analogies, “let’s look at this together”), trim ~25–30 % fat, tighten logical flow, keep all core science + interactive sims. No correctness issues found.

### Overall Course (applies to every page)
- **Tone upgrades** (apply globally):
  - Replace every “we will explore / the reader should note” → “let’s see what happens when…” or “imagine you’re standing at the pond…”
  - Cut “Big Ideas” bullet lists at end of lessons → turn into one short paragraph that feels like Feynman wrapping up a lecture.
  - Change all “Check Your Understanding” headings → “Let’s make sure you really got it” (Feynman never gave homework titles).
- **Reordering**: Current 8-lesson arc is **perfect**. Do **not** move anything between lessons.
- **Figures / sims**:
  - Keep **every** interactive simulation box (they are the heart of the platform).
  - Keep the glacier SVG in lesson 7.
  - All other images are unnecessary or decorative → delete (no value added beyond what text + sim already say).

### Lesson-by-lesson edits

**00-contents.html**  
- No content changes needed.  
- Trim meta line to: “8 lessons • February 2026”.

**01-inverse-problems.html**  
- **Trim**: Second paragraph (“The catch is that… ten to the forty-first power”) is beautiful but 40 % too long. Shorten to:  
  “Drop a stone in a pond: forward problem. See only ripples: inverse problem. Tiny noise in the ripples can turn your answer into complete nonsense — one part in ten thousand at the edge can give interior temperatures of 10⁴¹. That single fact is why this whole subject exists.”
- **Remove**: The entire “Why This Topic Matters” bulleted list (repeats what the intro already says better).  
- **Reorder inside intro paragraph**: Move the MRI/seismology/remote-sensing sentence right after the pond analogy so the reader immediately sees “this is everywhere”.

**02-regularization-the-first-rescue.html**  
- **Tone**: Already very good. Add one Feynman-style sentence after “sober regime”: “Your whole job is to find the place where the party ends and the science begins.”
- **Trim**:
  - The long numbered list under L-curve → collapse to 4 short bullets.
  - The entire “Other approaches exist…” paragraph (discrepancy principle, cross-validation) → delete (mentioned again in lesson 8; not central here).
  - Last 3 paragraphs of “The Drunk, the Sober, and the Dead” section (repetition).
- **Remove**: Both “Interactive Simulation” boxes stay, but delete the two paragraphs “Things to look for in the simulation” (the sim itself shows it).
- **Reorder**: Move the “Rule of thumb you can remember forever” box right after the L-curve explanation (before the drunk/sober/dead story) — people remember rules when they first need them.

**03-bayesian-inversion.html**  
- **Keep** almost intact — this is the strongest lesson.  
- **Trim**:
  - The long derivation of the posterior mean and covariance (the formulas are already in the Tikhonov lesson). Replace with: “When everything is Gaussian the MAP estimate and the posterior mean are the same number, and the covariance has a beautiful closed form — but the real point is this: regularization is just a prior in disguise.”
  - The entire “Pre-whitening the data” subsection (too much linear-algebra detail for the spirit of the course).
- **Remove**: The challenge at the end (Laplacian noise) — too advanced for this spot.

**04-iterative-methods-and-large-scale-tricks.html**  
- **Trim**:
  - First half of “Why iterative methods matter” (repeats what was said in regularization).
  - The long paragraph explaining why CG beats steepest descent (keep the zigzag valley analogy, cut the math history).
- **Reorder inside**: Put “Early stopping is itself regularization” right after the CG section — it is the punchline of the whole lesson.
- **Remove**: The challenge (implement CG without forming GᵀG) — too much coding homework for an online platform.

**05-linear-tomography.html**  
- **Trim**:
  - The long “Building the sensitivity matrix G” paragraph → shorten to one sentence + the sim.
  - “Resolution and coverage” subsection → move its key idea (“spike test and checkerboard test”) to lesson 7 where they are actually used.
- **Remove**: The entire “The Bridge: Linear to Nonlinear” section (it is repeated almost verbatim at the start of lesson 6).

**06-monte-carlo-methods.html**  
- **Tone**: Excellent.  
- **Trim**:
  - The detailed Metropolis step list → turn into a 4-line story (“You start somewhere, propose a step, sometimes go downhill on purpose…”).
  - The entire “Practical Diagnostics” subsection → keep only burn-in and acceptance rate (the other two are too technical).
- **Remove**: The challenge (parallel tempering) — nice but not core.

**07-geophysical-inversion-examples.html**  
- **Trim**:
  - Both “What the Posterior Tells Us” and “A Family of Possible Beds” subsections have overlapping text → merge into one shorter section called “What the cloud of answers actually says”.
  - Cut the second half of the resolution paragraph (spike/checkerboard explanation) — already covered earlier.
- **Keep**: The glacier SVG and both MCMC sim boxes.

**08-information-entropy-and-uncertainty.html**  
- **Trim**:
  - The long entropy examples with dice → keep only the fair vs loaded die (the 0.02 probability one adds nothing new).
  - The entire “Model comparison and selection” and “Monte Carlo diagnostics” subsections (mentioned earlier; feel tacked on).
- **Reorder**: Put the “Regularization as information control” paragraph right at the beginning — it ties the whole course together and should be the opening punch.

### Final suggested new TOC titles (tiny polish)
02 → “Regularization — The First Rescue” (keep)  
03 → “Bayesian Inversion — Why Regularization Works”  
04 → “Big Problems Need Clever Tricks”  
08 → “How Much Did the Data Really Teach You?”

Implement these ~25 targeted cuts and the whole course drops from ~65 k words to ~45 k, reads faster, feels more like Feynman sitting next to the student, and every lesson now has a single clear spine. The interactive sims and the two real examples stay as the emotional and visual anchors.

Let me know if you want me to output the full rewritten HTML for any single lesson first!