scientific-computing.md

**Suggested Edits — Scientific Computing Course (Feynman-style cleanup)**

**Overall course-wide changes (apply to every lesson)**
- Tone: Rewrite every paragraph in Feynman voice — warm, direct “you”, short sentences, everyday analogies, delight in the trickiness of nature/computers, zero pomp. Example: replace “The first thing a computational scientist needs to learn…” with “Your computer is a wonderful liar…”
- Remove every “text-white text-gray-300” fluff and repeated “leading-relaxed” styling — keep only the raw story.
- “Why This Topic Matters” list → fold into one lively paragraph that ends with “That’s why we’re here.”
- “Big Ideas” → move to the top of the lesson (right after the intro) so the student sees the map first, Feynman-style.
- “What Comes Next” → keep, but make it one excited sentence that teases the next adventure.
- “Check Your Understanding” → keep 3 questions max, make them playful (“Try this and watch what happens…”).
- “Challenge” → keep, but start with “Here’s a toy you can play with right now…”
- Any mention of “interactive simulation” without an actual figure → either remove the box or replace with “(Imagine a little movie here where you drag a slider and watch the wheel spin backwards…)”.
- All code blocks → keep, but add one-sentence Feynman comment at the top: “Watch this little magician at work…”

**Per-lesson edits**

**00-contents.html**  
- No content change needed. Just make the meta line “11 short adventures · February 22, 2026” to sound more playful.

**01-scientific-computing.html**  
- Already almost perfect Feynman tone.  
- Remove the entire second paragraph (“Once you understand the lies…”). It repeats the intro.  
- Merge the two long “why it matters” paragraphs into one punchy one.  
- Suggested new opening (Feynman version): “Your computer lies to you. It stores one-third as 0.333… and then quietly chops off the rest. Multiply by three and — surprise! — you don’t always get one. That tiny lie is the first thing every scientist who uses a computer has to meet face to face.”

**02-bounding-errors.html**  
- Section order is perfect.  
- Trim the long “condition number of root” example (the 10th-root story) to half length — it’s fun but wanders.  
- Remove the entire paragraph that begins “But information theory tells us…” — it’s clever but not central.  
- Move “Big Ideas” to top.  
- In the challenge, remove “and plots COND(f) versus n” — one plot is enough; the second is busy-work.

**03-linear-equations.html**  
- Excellent flow.  
- Remove the entire “Never compute A⁻¹ explicitly” boxed warning — it’s repeated later; one punch is enough.  
- Shorten the determinant = 10^{-15} example to one sentence.  
- Challenge: remove the last sentence about plotting both curves — one plot tells the story.

**04-linear-least-squares.html**  
- Reorder inside: put “QR is better because it never squares the condition number” right after the normal-equations paragraph (student needs the “why” immediately).  
- Remove the entire Householder code block and the “5 lines of NumPy” paragraph — too much detail for this lesson; save for advanced note.  
- Challenge: shorten to “Fit a degree-10 polynomial… using normal equations vs QR. Watch the condition number explode in one case and stay tame in the other.”

**05-eigenvalue-problems.html**  
- Power method → inverse iteration → Rayleigh → QR is logical.  
- Remove the entire “Gershgorin circles” subsection — beautiful but not core to the iterative story.  
- Challenge: remove “verify that the power method converges linearly…” — the student will see it on the plot; no need to state it twice.

**06-nonlinear-equations.html**  
- Perfect.  
- Remove the long “double root” paragraph in Check Your Understanding — it’s advanced; keep only the simple questions.  
- In challenge, remove “Map out the basins…” — too much for one exercise; just say “Try three different starting points and see which root each finds.”

**07-nonlinear-systems.html**  
- Reorder: put the Broyden update picture right after the secant equation (student needs the visual “why minimal change” immediately).  
- Remove the entire “If you start with B₀ = I …” paragraph — nice observation but not essential.  
- Challenge: keep only Newton vs Broyden comparison; drop the three initial points — one good experiment is enough.

**08-optimization-methods.html**  
- Flow good, but move “Big Ideas” to top.  
- Remove the entire paragraph that starts “Metaheuristics (simulated annealing…) are not algorithms — they are frameworks…” — too philosophical for this spot.  
- In challenge: remove “increase the dimension to 10 and repeat” — the 2-D Rosenbrock already shows everything; higher-D is just more typing.

**09-fast-fourier-transform.html**  
- Remove the entire “Periodicity caveat” subsection — it repeats the leakage story.  
- Keep the aliasing slider mention but change box to: “(Drag the slider yourself — watch the wheel suddenly spin the wrong way. That’s aliasing in action.)”  
- Challenge: shorten to “Recover three tones from noise, then isolate the 35 Hz one with a filter.” — one clean task.

**10-initial-value-problems.html**  
- Reorder slightly: after “stiff problems” paragraph, immediately show the Lorenz example (student wants to see chaos right away).  
- Remove the entire “Backward Euler costs more per step…” question in Check Your Understanding — it’s good but the challenge already covers it.  
- Challenge: keep only the Lorenz part; drop the “start two trajectories” — one plot of the butterfly is enough magic.

**11-partial-differential-equations.html**  
- This one is the longest and most verbose.  
- Remove the entire “Check Your Understanding” question 3 (Lax theorem) — too abstract for the end.  
- In the 2-D heat challenge, remove “compare wall-clock time” — accuracy vs stability is the point; timing is bonus.  
- Final paragraph is pure Feynman gold — leave untouched.

These changes will cut ~25-30 % of the text, make every lesson feel like a lively conversation with Feynman, and keep the logical spine intact while removing the parts that feel like textbook filler. The course will read faster, hit harder, and be far more fun — exactly what a computational-physics platform should be. Ready to implement?