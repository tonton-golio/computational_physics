# Assignments and Project Ideas

## Empirical assignment track

These assignments are designed to be small and concrete. Each one has a clear starting point and a clear success metric. Do not panic — you are not starting from a blank file. Each assignment provides the structure; you provide the implementation.

**Assignment 1: Hedge in expert advice.** Implement the Hedge (exponential weights) algorithm for a prediction-with-expert-advice problem. Generate $N = 8$ experts with fixed (but unknown to the learner) prediction strategies over $T = 10{,}000$ rounds. Plot the cumulative loss of your learner vs the cumulative loss of each expert, and shade the regret region.

*Success metric:* Your cumulative regret curve should grow sublinearly. Specifically, the ratio $R_T / T$ should be below 0.02 by $T = 10{,}000$. Your plot should clearly show the learner tracking the best expert within a small additive gap.

**Assignment 2: UCB1 and EXP3 in multi-armed bandits.** Implement UCB1 (for stochastic bandits) and EXP3 (for adversarial bandits). Test UCB1 on a 10-arm Bernoulli bandit with known gap structure. Test EXP3 on an adversarial sequence where the best arm switches every 500 rounds.

*Success metric:* UCB1 regret should be $O(\log T)$ in the stochastic setting — plot regret vs $\log T$ and verify approximate linearity. EXP3 regret should scale as $O(\sqrt{KT \log K})$ — plot regret vs $\sqrt{T}$ and verify approximate linearity. Include error bars from at least 50 independent runs.

**Assignment 3: SARSA and Q-learning in a gridworld.** Implement both SARSA and Q-learning on a cliff-walking gridworld (a grid with a cliff along one edge, reward of -1 per step, and -100 for falling off the cliff). Use epsilon-greedy with $\epsilon = 0.1$. Run both algorithms for 500 episodes and plot the per-episode return.

*Success metric:* SARSA should learn a safe path that avoids the cliff. Q-learning should learn the optimal (shortest) path along the cliff edge. Your per-episode return plots should clearly show this behavioral difference. Include a visualization of the learned policies (arrows on the grid).

**Assignment 4: DQN on a compact control benchmark.** Implement DQN with a replay buffer and target network on the CartPole-v1 environment (or a comparable compact benchmark). Train for at least 500 episodes.

*Success metric:* Your agent should consistently achieve the maximum episode length (500 steps) within 300 episodes. Plot the learning curve (episode return vs episode number) and include an ablation: run once without the replay buffer and once without the target network to show that both are necessary for stable learning.

## Theoretical assignment track

These assignments ask you to work through the key proofs and arguments by hand.

**Assignment 5: Regret bound for Hedge.** Derive the $O(\sqrt{T \log N})$ regret bound for the Hedge algorithm. Start from the potential function $\Phi_t = \sum_i w_t(i)$. Upper-bound $\log(\Phi_{T+1}/\Phi_1)$ using the multiplicative update, and lower-bound it using the weight of the best expert. Combine to get the final bound. Clearly state where the choice $\eta \asymp \sqrt{\log N / T}$ comes from.

**Assignment 6: UCB concentration argument.** Prove that UCB1 achieves gap-dependent regret $O\left(\sum_{a:\Delta_a > 0} \frac{\log T}{\Delta_a}\right)$. Use Hoeffding's inequality to bound the probability that the confidence interval for an arm fails, and show that suboptimal arms are not pulled too often.

**Assignment 7: Bellman contraction and fixed-point uniqueness.** Prove that the Bellman optimality operator $\mathcal{T}$ is a $\gamma$-contraction in the supremum norm. Use the Banach fixed-point theorem to conclude that $V^*$ is unique and that value iteration converges geometrically. Compute the number of iterations needed for $\epsilon$-accuracy.

**Assignment 8: On-policy vs off-policy.** Formally state the convergence conditions for Q-learning (Robbins-Monro conditions on step sizes, sufficient exploration). Explain why SARSA converges to $Q^\pi$ for the behavior policy $\pi$, while Q-learning converges to $Q^*$ regardless of the behavior policy. Discuss what breaks when you add function approximation (the deadly triad).

## Course project

Choose one applied setting and report reproducible experiments. The project should demonstrate that you can take a real problem, formulate it as an online learning or RL problem, implement a solution, and analyze its performance.

**Project idea 1: Online advertising or recommendation simulator.** Build a contextual bandit simulator where contexts represent user profiles and actions represent items to show. Compare EXP4 against epsilon-greedy and a baseline that ignores context.

*Grading metric:* (1) Cumulative regret vs the oracle policy that knows the true reward model, plotted over 50,000 rounds. (2) Regret per 1,000-round block, showing learning speed. (3) Comparison of at least three algorithms. (4) Discussion of how context dimensionality affects performance. (5) Clean, reproducible code with a single command to regenerate all figures.

**Project idea 2: Inventory or control MDP.** Model a simple inventory management problem as an MDP (ordering decisions, stochastic demand, holding and shortage costs). Apply value iteration (with known model) and Q-learning (model-free) and compare.

*Grading metric:* (1) Average cost per step vs the optimal policy, over 10,000 steps. (2) Convergence plot showing how Q-learning estimates approach the true Q-values over time. (3) Sensitivity analysis for at least two parameters (e.g., demand variance, discount factor). (4) Comparison of model-based vs model-free approaches. (5) Clean, reproducible code.

**Project idea 3: Custom game or robotics-inspired simulator.** Design a small game or physical control problem (maze navigation, simple robotic arm, resource allocation) and train an RL agent. If the state space is large, use DQN.

*Grading metric:* (1) Learning curve showing episode return over at least 1,000 episodes. (2) Final policy visualization. (3) Ablation study removing one key component (e.g., replay buffer, exploration). (4) Comparison against a random baseline and a hand-crafted heuristic. (5) Clean, reproducible code.

**Project idea 4: Gymnasium-style benchmark.** Pick a standard Gymnasium environment (LunarLander, MountainCar, Acrobot) and implement DQN or an actor-critic method from scratch (no high-level RL libraries for the core algorithm).

*Grading metric:* (1) Learning curve with error bars from at least 10 seeds. (2) Comparison against at least one other algorithm. (3) Hyperparameter sensitivity plot for at least two hyperparameters. (4) Analysis of failure modes — when and why does the agent fail? (5) Clean, reproducible code.

## General project deliverables

Every project, regardless of the specific topic, must include:

1. **Problem definition and assumptions** — what is the state, action, reward, and feedback model?
2. **Baselines and chosen algorithm(s)** — what are you comparing against, and why did you pick this algorithm?
3. **Experimental protocol** — how many runs, how many steps, what hyperparameters did you sweep?
4. **Regret or return analysis** — the quantitative results, with proper error bars.
5. **Failure modes and next improvements** — what went wrong, what would you try with more time?
