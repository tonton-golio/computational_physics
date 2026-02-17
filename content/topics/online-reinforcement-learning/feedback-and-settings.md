# Forms of Feedback and Problem Settings

The available feedback determines both algorithm design and regret guarantees.

## Full-information feedback

At round $t$, the learner observes the whole loss vector:

$$
(\ell_t(1), \ell_t(2), \ldots, \ell_t(K)).
$$

Typical setting: prediction with expert advice.

## Bandit (partial) feedback

Only the chosen action's loss is observed:

$$
\ell_t(a_t).
$$

This requires explicit exploration and usually introduces an extra $\sqrt{K}$ factor in regret rates.

## Contextual feedback

A context vector $x_t$ arrives before action selection.
The learner chooses $a_t$ using $(x_t, \text{history})$ and only receives feedback for the chosen action.

## Additional realism

- Delayed feedback.
- Noisy feedback.
- Drifting or non-stationary environments.
- Adversarially chosen outcomes.

These variants alter concentration arguments, estimator variance, and final regret bounds.

Bandit estimation quality and exploration budget are central themes in the next lessons.
