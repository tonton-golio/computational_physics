# Content Style Guide

Writing conventions for lesson markdown files. For file structure, placeholders, and registration steps see [content-authoring.md](./content-authoring.md).

---

## Voice and Tone — The Feynman Standard

Write like Richard Feynman taught: with wonder, honesty, and the conviction that any idea can be made clear if you find the right way to look at it. Every lesson should feel like a brilliant friend is explaining something at a whiteboard — excited, unpretentious, and never afraid to say "nobody really knows why."

* **Exploratory and wonder-driven.** Start from curiosity, not from definitions. Ask "what would happen if…?" and "why should we believe that…?" before giving answers. Make the reader feel they are discovering the idea alongside you, not being lectured at.
* **Second person ("you") and first-person plural ("we").** Use "you" when addressing the reader and "we" when reasoning through a derivation together. "Let's see what happens" is better than "it can be shown."
* **Thought experiments over formalism.** Whenever possible, set up a concrete mental picture before introducing equations. Feynman never started with the general case — he started with the simplest possible example that captures the essential physics.
* **Active voice by default.** Prefer "the algorithm picks the arm with the highest score" over "the arm with the highest score is picked by the algorithm."
* **Present tense for exposition.** Use past tense only when it serves the narrative.
* **Honest about what is hard.** Do not gloss over subtleties or wave your hands at a gap. If a step is tricky, say so. If something is still an open question, say that too.

### What to avoid

* **No chapter or lesson numbers.** Never write "in lesson 3" or "in Chapter 2." Reference other lessons by name using markdown links.
* **No external source references.** Do not include "Further reading" sections, textbook citations, paper references, or "see Ch. 3.1" notes. This course is self-contained — every concept should be explained within the lessons themselves.
* **No academic stiffness.** Avoid "it is well known that," "one can show that," "the reader will recall." If you catch yourself writing like a textbook, rewrite it.

---

## Lesson Structure

### Title

Start every file with a single `# H1` heading. This is the lesson title. Do not add subtitles or taglines beneath it.

```md
# Nonlinear Equations
```

### Opening — Narrative Hook

Open with a vivid scenario, question, or analogy that grounds the topic in something concrete before introducing any formalism. One to three paragraphs is typical.

```md
# The Metropolis Algorithm

Suppose you have a huge box full of air molecules — maybe $10^{23}$ of them.
You want to know the average energy, the pressure, the heat capacity.
In principle every configuration is possible, but almost all of the interesting
physics lives on a tiny fraction of them. How do you find that fraction
without checking every state?
```

### Body

Use `## H2` for major sections and `### H3` for subsections. Do not use `#### H4` — if you need deeper nesting, restructure the section.

### Topic Home Pages

Each topic has a `home.md` landing page. These are short, focused pages with exactly two sections:

```md
# Topic Name

[Narrative hook: 2-4 paragraphs that make the reader curious about the topic.
Use a vivid scenario, question, or thought experiment — the same Feynman-style
opening used in lessons. No definitions, no formalism. Make them want to keep reading.]

## Why This Topic Matters

* Real-world application or consequence.
* Another application.
* Another application.
```

Home pages should **not** include:

* Lesson listings, roadmaps, or course maps (the UI handles navigation)
* "Key mathematical ideas" sections
* Recurring characters or personas ("Meet Alex," "Rosie the Rubber Band")
* Visual/simulation galleries
* Glossaries, cheat sheets, or reference tables
* "What you will learn" or learning outcomes
* Cross-links to other topics

Keep home pages under 30 lines. Their job is to hook the reader and justify why the topic is worth their time — nothing more.

### Closing — Structured Wrap-Up

End every lesson with the following four sections in order. Do not skip any of them, and do not rename them.

```md
## Big Ideas

* First key takeaway.
* Second key takeaway.

## What Comes Next

One to two paragraphs teasing the next lesson and connecting it to what was just covered.

## Check Your Understanding

1. Conceptual question about the material.
2. Another question.

## Challenge

A harder, open-ended problem that pushes the reader beyond the lesson.
```

Guidelines for each closing section:

* **Big Ideas** — 2-4 bullet points. Write pithy insights, not summaries. Think "the one sentence you would tell a friend at a bar." Avoid restating definitions.
* **What Comes Next** — 1-2 paragraphs that explain *why* the next lesson is the natural question to ask after this one. For the last lesson in a topic, wrap up the topic arc instead. Reference the next lesson by name using a markdown link.
* **Check Your Understanding** — 3 questions that test understanding, not recall. Ask "why does X happen?" or "what would change if…?" — never "what is the definition of X?"
* **Challenge** — One genuinely hard, open-ended problem. Prefer problems that require implementation, estimation, or synthesis across ideas. The reader should not be able to answer it by scanning the lesson text.

Do not use other closing patterns: no `## What We Just Learned`, no bold summary lines (`**What we just learned in one sentence:**`), no italic teaser paragraphs separated by `---`, no `## Further reading`, no `## Takeaway`.

---

## Self-Containment

Every lesson and every topic must be self-contained. The reader should never need to look something up outside this platform.

* **No external references.** Do not cite papers, textbooks, arXiv links, Wikipedia pages, or YouTube videos. If a concept is important enough to mention, explain it in the lesson itself.
* **No "Further reading" sections.** These have been removed and must not be reintroduced.
* **No inline citations.** Do not write "(Vaswani et al., 2017)" or "(Goodfellow et al., 2016)." If you need to credit an idea historically, weave it into the narrative: "Vaswani's team showed that attention alone is enough."
* **No lesson or chapter numbers.** Never write "in lesson 3" or "see Chapter 2." Always reference other lessons by name with a markdown link: `[the Metropolis algorithm](./metropolisAlgorithm)`.

---

## Lesson Ordering Philosophy

When ordering lessons within a topic, follow the principle of **tool-before-task**: teach the tools students need before asking them to use those tools on harder problems. Place training/optimization material before complex architectures. Place foundations before applications. End with theory or capstone lessons that reflect on the material.

---

## Paragraphs

Keep paragraphs short — two to five sentences. Use single-sentence paragraphs sparingly, for emphasis:

```md
That is all a neuron does.
```

---

## Math

### Inline and Display

* Inline math: `$...$`
* Display math: `$$...$$` on its own lines

```md
The energy of a single mode is $E = \hbar\omega(n + \tfrac{1}{2})$.

The partition function is

$$
Z = \sum_{\text{states}} e^{-\beta E}
$$
```

### Multi-line Equations

Use `\begin{aligned}` inside `$$` for multi-line derivations:

```md
$$
\begin{aligned}
F &= ma \\
  &= m \frac{dv}{dt}
\end{aligned}
$$
```

### Equation Gloss

After important display equations, add an italicized plain-language explanation:

```md
$$
\frac{dx}{dt} = \alpha x - \beta x y
$$

*In words: the prey population grows on its own but shrinks when predators are present.*
```

This pattern helps readers who can follow the prose but need a moment to parse the notation. Not every equation needs a gloss — use it for key results and for equations that introduce new quantities.

---

## Terminology

* **Bold** for introducing a new technical term the first time it appears:

  ```md
  A **fluid** is a material with zero shear modulus.
  ```

* *Italics* for emphasis and for Latin or foreign terms.
* Do not use quotation marks to introduce terms.

---

## Lists

* Use `*` (asterisk) for unordered lists.
* Use `1.`, `2.`, … for ordered lists when sequence matters (algorithms, step-by-step procedures).

```md
* First property
* Second property

1. Compute the gradient.
2. Update the weights.
3. Repeat until convergence.
```

---

## Callouts

Use blockquotes with a bold label. Two standard labels are defined:

### Key Intuition

For the core conceptual takeaway of a section — the insight you want the reader to remember.

```md
> **Key Intuition.** The partition function $Z$ is the Rosetta Stone of
> statistical mechanics: every thermodynamic quantity is a derivative of
> $\ln Z$.
```

### Challenge

For harder problems or exercises the reader should attempt, beyond the standard "Check your understanding" questions.

```md
> **Challenge.** Implement Newton's method for a system of two nonlinear
> equations. Compare the convergence rate to the bisection method.
```

Do not invent ad hoc callout labels. If a note does not fit either label, write it as regular prose or as a parenthetical.

---

## Code Blocks

### Languages

Python is the primary language. Pseudocode is the secondary language.

### Code Toggle

When providing both Python and pseudocode, use the code-toggle pattern:

````md
```python
def forward(x, W, b):
    return np.maximum(0, W @ x + b)
```
<!--code-toggle-->
```pseudocode
FUNCTION forward(x, W, b):
    RETURN max(0, W · x + b)
```
````

Pseudocode uses `UPPERCASE` keywords: `FOR`, `RETURN`, `FUNCTION`, `IF`, `WHILE`, `CLASS`, `INIT`, `FORWARD`.

### Standalone Code

When only one language is needed, use a fenced block with the language tag. Do not add a code-toggle comment.

---

## Simulation and Figure Placeholders

Place `[[simulation id]]` and `[[figure id]]` on their own line, after the text that explains what the reader will see:

```md
The simulation below lets you drag the endpoints and watch the
interpolating polynomial update in real time.

[[simulation polynomial-interpolation]]
```

Do not place placeholders in the middle of a derivation or paragraph. Always use the `[[simulation id]]` syntax — never JSX-style component tags.

---

## Cross-References

Use markdown links to reference other lessons:

```md
We saw in [Regularization](./regularization) that adding a penalty term
prevents overfitting.
```

For lessons in a different topic, use a relative path from the content root or reference the route:

```md
This connects to the [Metropolis algorithm](/topics/complex-physics/metropolis-algorithm).
```

---

## Headings Checklist

| Level | Purpose | Example |
|-------|---------|---------|
| `#`   | Lesson title (exactly one per file) | `# Nonlinear Equations` |
| `##`  | Major section | `## Newton's Method` |
| `###` | Subsection | `### Convergence Rate` |
| `####`| **Do not use** | — |

---

## Common Mistakes to Avoid

### Structure

* Multiple `# H1` headings in one file — exactly one per file.
* `#### H4` headings — restructure into `### H3` or split sections.
* Italic subtitles or taglines under the `# H1` title.
* Non-standard closing patterns — always use the four-section wrap-up (`## Big Ideas`, `## What Comes Next`, `## Check Your Understanding`, `## Challenge`).

### Formatting

* Mixing `-` and `*` for bullets — always use `*`.
* `\boxed{}` for highlighting results — use a **Key Intuition** callout instead.
* Blockquote figure descriptions (`> **Figure: ...**`) — use `[[figure id]]` placeholders.
* JSX-style component tags (`<ComponentName />`) — use `[[simulation id]]` placeholders.
* Ad hoc callout labels (e.g., "Try this," "You might be wondering," "Aside," "Bayesian side note") — stick to **Key Intuition** and **Challenge** only. Everything else should be regular prose.

### Voice and references

* Chapter/lesson numbers — use lesson names with markdown links.
* "Further reading" sections, textbook citations, inline paper references, or bibliography sections.
* "It is well known," "one can show," "the reader will recall," "as shown in Ch. 3" — rewrite in Feynman voice.
* YAML frontmatter keyword blocks (`__Topic 3 keywords__`) or `# Readings` sections — remove these legacy patterns.

### Legacy patterns to watch for

These patterns existed in older content and were removed during standardization. Do not reintroduce them:

* `__bold keywords__` pseudo-frontmatter blocks
* `# Readings` with textbook chapter references (e.g., "Ch. 3.1-5")
* Section-numbered headings (e.g., "## 3.1 Eigenstates of…")
* Recurring characters or personas in home pages ("Meet Alex," "Rosie the Rubber Band")
* Course maps, roadmaps, ASCII diagrams, cheat sheets, or glossaries in home pages
* `## What We Just Learned` / `## What's Next` (replaced by the standard four-section closing)
* Bold summary lines (`**What we just learned in one sentence:**`)
* Italic teaser paragraphs separated by `---` at end of lessons
* `## Takeaway` sections
* `## Further Reading` / `## References` sections
