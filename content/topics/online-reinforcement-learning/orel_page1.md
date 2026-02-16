
KEY: concepts
* **Reinforcement learning** is an area of machine learning concerned with how software agents ought to take actions in an environment in order to maximize some notion of cumulative reward.

* A **Markov decision process (MDP)** is a 5-tuple $(S, A, P, R, \gamma)$, where $S$ is the set of states, $A$ is the set of actions, $P$ is the state transition probability, $R$ is the reward function, and $\gamma$ is the discount factor.

* A **policy** is a mapping from states to actions. A **value function** is a mapping from states to real numbers. A **Q-function** is a mapping from state-action pairs to real numbers.

* A **state-value function** is a mapping from states to real numbers. A **state-action value function** is a mapping from state-action pairs to real numbers.

* **Regret** is the difference between the best possible reward and the reward actually received.

* **Exploration** is the process of finding new actions that lead to better rewards. **Exploitation** is the process of using the current knowledge to maximize the reward.

* **Off-policy learning** is a learning method in which the behavior policy and the target policy are different. **On-policy learning** is a learning method in which the behavior policy and the target policy are the same.

* **Online learning** is a learning method in which the agent learns from each step of interaction with the environment. **Batch learning** is a learning method in which the agent learns from a batch of data. **Offline learning** is a learning method in which the agent learns from a dataset that is already collected. Batch versus Online. Batch is great if we dont need to update dynamically, because we save compute
> A key assumption we make is that the new data is from the same distribution as the data on which we trained. And samples are i.i.d.



**Kinds of feedback**
    * We see diffent cases in terms of what kinda feedback we get. We may have full feedback as in the case of the stock market (you dont have to buy as stock to know the price), however for a medical treatment, we have to execute the strategy to assess the succes. The limited (bandit) feedback scenario exists in the middle. This difference affects the exploration-exploitation trade-off. (think $\epsilon$-greedy)
**Environmental resistance**
    * Another axis differenciating online learning problems, is how the environment react to the algorithm. This spam-fitering; the spammers will adapt, overcome, prosper. We label this kinda set-up: *adversarial*. If we have adversarial environment reaction, we cannot do batch learning.

    We introduce *regret* as a metric for evaluation. We use hindsight to calculate this.
**Structural complexity**
    * We may have a stateless system; think single medical treatemnt of patients. They have no affect on each other.
    However if we do multiple treatments on a single patient, they influnce each other. This is a centextual problem, a class of problem where batch ML performs well.
    In cases where we have high depence, we can use Markov Decision Processes (MDP).
