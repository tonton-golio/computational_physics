# Additional Topics



    ---
    ### epsilon-greedy policy
    The epsilon-greedy policy is a simple policy that is used in reinforcement learning. It is a greedy policy that is biased towards exploration. It is a policy that is used to balance exploration and exploitation.
    $$
    \pi(a|s) = \begin{cases}
    1-\epsilon + \frac{\epsilon}{|A|} & \text{if } a = \underset{a}{\operatorname{argmax}} Q(s,a) \\
    \frac{\epsilon}{|A|} & \text{otherwise}
    $$
    If epsilon decreases over time, the policy will converge to the optimal policy almost surely.

    ---
    ### Q-learning
    Q-learning is a model-free reinforcement learning algorithm. It can be used to find the optimal action-selection policy for any given (finite) Markov decision process (MDP). It does not require a model of the environment, and it can handle problems with stochastic transitions and rewards, without requiring adaptations.
    $$
    Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left( r_t + \gamma \max_{a} Q(s_{t+1}, a) - Q(s_t, a_t) \right)    
    $$
    """


    * **Temporal difference learning** is a model-free reinforcement learning method that learns from both full episodes and single steps.

* **Q** is a function that maps from a state and an action to a real number which holds the value of the that action in that state.



## Lecture 2 notes (9 feb 2023)
    



///


