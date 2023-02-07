from streamlit_profiler import Profiler
p = Profiler()
p.start()

from utils.utils_global import *

'# Online and Reinforcement Learning'

text_intro = """
*Pre-start preperations*
"""

st.markdown(text_intro)

tabs = st.tabs(["Hoeffding's inequality",
                "Markov’s inequality",
                "Chebyshev's inequality",
                "Illustration"])

with tabs[0]: # Hoeffding's inequality
    r"""
    Let $X_1, \cdots, X_n$ be independent random variables such that $a_{i}\leq X_{i}\leq b_{i}$ almost surely. Consider the sum of these random variables,

    $$
        S_n = X_1 + \cdots + X_n.
    $$
    Then Hoeffding's theorem states that, for all t > 0,

    """
    #$$
    #\operatorname{P}
    # \left( S_{n}-\mathrm {E} \left[S_{n}\right]\geq t\right) 
    #\leq 2\exp \left(-{\frac {2t^{2}}{\sum _{i=1}^{n}(b_{i}-a_{i})^#{2}}}\right)
    #$$
    r"""
    $$
    \operatorname {P} \left(\left|S_{n}-\mathrm {E} \left[S_{n}\right]\right|\geq t\right)\leq 2\exp \left(-{\frac {2t^{2}}{\sum _{i=1}^{n}(b_{i}-a_{i})^{2}}}\right)
    $$
    """
    def extra():
        cols = st.columns(3)


        n_exp = cols[0].slider('n_exp', 1, 100, 1)
        n = cols[1].slider('n',10, 1000, 100, 10)
        t = cols[2].slider('t', 0.0, 10., 1.,)
        Xs = []
        ab = np.random.randn(n,2)*2+10
        ab = np.sort(ab, axis=1)
        
        for i in range(n_exp):
            Xs.append(np.random.uniform(ab[:,0],ab[:,1]))
        Xs = np.array(Xs)

        def plot():
            fig = plt.figure(figsize=(6,3))
            plt.hist(ab[:,0], alpha=.6)
            plt.hist(ab[:,1], alpha=.6)

            plt.close()
            st.pyplot(fig)


        S = np.sum(Xs, axis=1)

        
        E = np.sum(np.mean(ab, axis=1))
        delta = abs(S-E)
        
        LHS = sum(delta >= t) / n_exp
        

        RHS = 2*np.exp(-2*t**2 / sum((ab[:,1]-ab[:,0])**2))
        f"""
        $$
        {LHS} \leq {RHS}
        $$
        """
        

with tabs[1]: #Markov’s inequality
    r"""
    Markov's inequality gives an upper bound for the probability that a non-negative function of a random variable is greater than or equal to some positive constant.[wikipedia]

    If $X$ is a nonnegative random variable and $a > 0$, then the probability that $X$ is at least $a$ is at most the expectation of $X$ divided by $a$:
    $$
    \operatorname {P} (X\geq a)\leq {\frac {\operatorname {E} (X)}{a}}.
    $$
    """


with tabs[2]: # Chebyshev's inequality
    r"""
    Only a definite fraction of values will be found within a  specific distance from the mean of a distribution. 
    $$
    \Pr(|X-\mu |\geq k\sigma )\leq {\frac {1}{k^{2}}}
    $$
    """

with tabs[3]:
    """
    We will draw 20 Bernoulli random variables 1M times with bias 1/2. A Bernoulli random variable is a random variable that can only take two possible values, usually 0 and 1.

    Questions are from [here](https://drive.google.com/file/d/1mEh5ZdJ3H3DIrG5GfpHrrA3jLByw0oYZ/view)
    """
    
    def q1(p_heads=.5, n_exp = 1000, n_draws = 20):
        
        
        X = np.random.choice(np.arange(2),(n_exp, n_draws), p=np.array([p_heads,1-p_heads]))
        count_heads = np.mean(X, axis=1)
        lst = []
        alphas = np.arange(0.,1.05,0.05)
        for alpha in alphas:
            lst.append(len(count_heads[count_heads>=alpha])/n_exp)


        # q 1.3 Markov bound
        # Markov bound :  expectation of X / alpha
        Expectation_value = 0.5
        markov_bound = Expectation_value/alphas
        markov_bound[markov_bound>1] = 1

        # q 1.4 Chebyshev bound
        #count_heads
        #var = E (X-mu)
        var = np.var(X)
        chebyshev_bound = var/alphas**2
        chebyshev_bound[chebyshev_bound>1] = 1


        # q 1.5 Hoeffding bound

        hoeffdings_bound = 2 * np.exp( -1 * 2*alphas**2 / 20 ) 


        fig, ax = plt.subplots(figsize=(4,3))
        ax.plot(alphas, lst, label=r"""$\frac{1}{20}\sum X_i \geq \alpha $""")
        
        ax.plot(alphas, markov_bound, label='Markov bound')
        ax.plot(alphas, chebyshev_bound, label='Chebyshev bound')
        ax.plot(alphas, hoeffdings_bound, label="Hoeffding's bound")

        ax.set(xlabel=r'$\alpha$', ylabel='freq')
        
        
        plt.legend(ncol=2, bbox_to_anchor=(.07,1.1))
        plt.close()
        return fig
    
    cols = st.columns(2)
    cols[0].markdown(r"""
    1.2) T granularity of $\alpha$ is sufficient because, with only 20 throws, the mean  number of heads is limited to alpha-values separated by 1/20.
    
    
    1.6) Chebyshev's bound does the best in limiting the outcome space. All bounds are respected!


    1.7) $\alpha = 1$ does not allow for a single loss, and so the probability is
    $$
        \frac{1}{2^{20}} \approx \frac{1}{1 \text{ million}}
    $$
    $\alpha = 0.95$ allows a single loss, This loss may come at any time. 
    so this should be 21 times as likely as the perfect case
    """)

    p_heads = cols[1].slider('probability of heads', 0., 1.,0.5, 0.05)
    cols[1].pyplot(q1(p_heads, n_exp = 1000, n_draws = 20))




'---'
'## PyTorch'

"""
pyTorch is ML package for python, developed by facebook/meta. 

"""

def pytorch_toturial1():


    # Download training data from open datasets.
    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    # Download test data from open datasets.
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )

    batch_size = 64

    # Create data loaders.
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    for X, y in test_dataloader:
        st.write(f"Shape of X [N, C, H, W]: {X.shape}")
        st.write(f"Shape of y: {y.shape} {y.dtype}")
        break



    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    st.write(f"Using {device} device")

    # Define model
    class NeuralNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.flatten = nn.Flatten()
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(28*28, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 10)
            )

        def forward(self, x):
            x = self.flatten(x)
            logits = self.linear_relu_stack(x)
            return logits

    model = NeuralNetwork().to(device)
    st.write(model)



    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


    def train(dataloader, model, loss_fn, optimizer):
        size = len(dataloader.dataset)
        model.train()
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            # Compute prediction error
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                st.write(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")



    def test(dataloader, model, loss_fn):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        st.write(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


    epochs = 5
    for t in range(epochs):
        st.write(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)
    st.write("Done!")



    classes = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]

    model.eval()
    x, y = test_data[0][0], test_data[0][1]
    with torch.no_grad():
        pred = model(x)
        predicted, actual = classes[pred[0].argmax(0)], classes[y]
        st.write(f'Predicted: "{predicted}", Actual: "{actual}"')

def pytorch_example_fromCHATGPT():
    # Load the MNIST dataset and apply transformations
    train_dataset = datasets.MNIST(root='data/', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = datasets.MNIST(root='data/', train=False, transform=transforms.ToTensor(), download=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=100, shuffle=False)

    # Define the CNN architecture
    class CNN(nn.Module):
        def __init__(self):
            super(CNN, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
            self.fc1 = nn.Linear(7 * 7 * 64, 1024)
            self.fc2 = nn.Linear(1024, 10)

        def forward(self, x):
            x = nn.functional.relu(self.conv1(x))
            x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
            x = nn.functional.relu(self.conv2(x))
            x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
            x = x.view(-1, 7 * 7 * 64)
            x = nn.functional.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    # Initialize the CNN model, loss function and optimizer
    model = CNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    num_epochs = 2
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("Epoch [{}/{}], Loss: {:.4f}".format(epoch + 1, num_epochs, loss.item()))

    # Test the model
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print("Accuracy of the model on the 10000 test images: {} %".format(100 * correct / total))

    # Save the model for future use
    #torch.save(model.state_dict(), 'mnist_cnn.pth')



'---'


"""
# Multi armed bandit

[link to youtube vid](https://www.youtube.com/watch?v=e3L4VocZnnQ)

We are a professor staying in a small town for 300 days. In the town there are 3 restaurants. They bring different amounts of happiness per meal, but at the beginning we dont know.


We have to balance two concepts; exploration and exploitation.


The most naive strategies are the two extremes;
* we could explore all the time, so go to a random restaurant every day
* we could exploit maimally, so the first 3 days we explore. And after having eaten at each once, we just stick with the best one for the remaining 297 days.





We measure the goodness of a strategy by the regret, $\rho$. Regret is the difference from the maximum amount of happiness
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from tqdm import tqdm

'### Greedy'

restaurants = { # (mu, sigma)
    "City wok"      : (8, 1),
    "McDonald's"  : (5,3),
    "Da Cavalino" : (10,3)}

def explore():
    #print('explore')
    r = np.random.choice(list(restaurants.keys()))
    
    (mu, sig) = restaurants[r]
    score = np.random.normal(mu, sig, None)
    return r, score

def exp2df(experiences):
    df = pd.DataFrame.from_dict(experiences).T
    df.rename(columns={0:'Restaurant', 1:'Score'}, inplace=True)
    return df

def exploit_old(experiences):
    df = exp2df(experiences)
    best = df.groupby('Restaurant').mean()
    r = best.sort_values(by='Score').index[-1]
    
    (mu, sig) = restaurants[r]
    score = np.random.normal(mu, sig, None)
    return r, score

def exploit(score_table, method='standard'):
    #print('exploit')
    if len(score_table) == 0: return explore()
    elif len(score_table) == 1: r = list(score_table.keys())[0]
    else: r = where2go(score_table, method)
        
    (mu, sig) = restaurants[r]
    score = np.random.normal(mu, sig, None)
    return r, score

def where2go(score_table, method='standard'):
    if method == 'standard':
        mean_score = np.array([(key , score_table[key]['sum']/score_table[key]['count']) for key in score_table])

        r = mean_score[np.argmax(mean_score[:, 1].astype(float)),0]

    elif method == 'UCB':
        t = sum(score_table[key]['count'] for key in score_table.keys())
        #print('t=',t)
        mean_score = np.array([(key , 
            score_table[key]['sum']/score_table[key]['count'] + np.sqrt(t/score_table[key]['count'])) for key in score_table])
        
        r = mean_score[np.argmax(mean_score[:, 1].astype(float)),0]
    return r



def gogogo(n=300, epsilon=10, method='standard'):
    score_table = {}
    experiences = {}
    for i in range(n):
        rand_num = np.random.uniform(0,1)
        if rand_num < epsilon/100:

            r, score = explore()
            experiences[i] = (r, score)
            if r in score_table:
                score_table[r]['count'] += 1
                score_table[r]['sum']   += score
            else:
                score_table[r] = {'count' : 1, 'sum': score}

        else:
            r, score = exploit(score_table, method)
            experiences[i] = (r, score)

            if r in score_table:
                score_table[r]['count'] += 1
                score_table[r]['sum']   += score
            else:
                score_table[r] = {'count' : 1, 'sum': score}


    df = exp2df(experiences)
    total_score = df.Score.sum()
    #print(f'Total score = {total_score}')
    return total_score


scores = []
epls = np.logspace(0,2,15, dtype=int)
print(epls)
for epl in tqdm(epls):
    scores_tmp = []
    for i in range(10):
        s = gogogo(300, epl)
        scores_tmp.append(s)
    scores.append(scores_tmp)


scores = np.array(scores)

fig = plt.figure(1)
plt.errorbar(epls, np.mean(scores, axis=1), np.std(scores, axis=1), lw=0, elinewidth=1)
plt.xscale('log')
plt.xlabel('epsilon')
plt.ylabel('total score')
plt.title(r'$\epsilon$-greedy; striking a balance between exploration and exploitation')
plt.show()
plt.close()
st.pyplot(fig)



"A natural extension of this algorithm, lets the chance of exploration fall with iteration-count."



'---'

r"""
### Upper confidence bound (UCB)
We may obtain a better algortihm by adressing a central flaw
> When comparing means, these may have very different numbers of samples.

The new mean is defined as;
$$
\mu_r = \hat{\mu_r} + \sqrt{\frac{2\ln(t)}{N_t(r)}},
$$
maximize this, i.e., maximize Hoffding's inequality.

where $t$ is the iteration step, and $N_t(r)$ is the number of times restaurant $r$ has been visited so far.

"""





"---"


import numpy as np
import matplotlib.pyplot as plt
import streamlit as st


"""
# ORel
## Notes Lecture 1
"""
cols = st.columns((2,1))
cols[0].markdown(r"""

Batch versus Online. Batch is great if we dont need to update dynamically...

> A key assumption we make is that the new data is from the same distribution as the data on which we trained. And samples are i.i.d.

Example of online learning: Stock market investing... and others 

""") # intro
cols[1].image('one_armed_bandit.jpeg')

st.markdown(r"""
    ---
    **Kinds of feedback**

    We see diffent cases in terms of what kinda feedback we get. We may have full feedback as in the case of the stock market (you dont have to buy as stock to know the price), however for a medical treatment, we have to execute the strategy to assess the succes. The limited (bandit) feedback scenario exists in the middle. This difference affects the exploration-exploitation trade-off. (think $\epsilon$-greedy)
""") # kinds of feedback
st.markdown(r"""
    ---
    **Environmental resistance**

    Another axis differenciating online learning problems, is how the environment react to the algorithm. This spam-fitering; the spammers will adapt, overcome, prosper. We label this kinda set-up: *adversarial*. If we have adversarial environment reaction, we cannot do batch learning.

    We introduce *regret* as a metric for evaluation. We use hindsight to calculate this.
""") # Environmental resistance
st.markdown(r"""
    ----
    **Structural complexity**

    We may have a stateless system; think single medical treatemnt of patients. They have no affect on each other.
    
    However if we do multiple treatments on a single patient, they influnce each other. This is a centextual problem, a class of problem where batch ML performs well.

    In cases where we have high depence, we can use Markov Decision Processes (MDP).


""") # structural complexity


### Assignment 1


#### question 3: rain jacket
def rain_jacket():

    # see page 3 for discount factor https://web.engr.oregonstate.edu/~afern/classes/cs533/notes/infinite-horizon-MDP.pdf
    C = 2 # cost of being rained on
    U = 1 # cost of taking jacket when not needed
    y = gamma = 0.5 # discount factor, bounded (0,1)
    p = .8 # probability of rain
    s = jacket_state = 0 # 0 is home, 1 is work

    def rain(p):
        if np.random.uniform(0,1,None) < p:
            return True
        else: return False

    def bring_just_in_case(q):
        if np.random.uniform(0,1,None) < q:
            return True
        else: return False

    def move(s):
        if s == 1: return 0
        else: return 1


    qs = np.linspace(0,1,10)
    costs = []
    for q in qs:
        total_cost = 0 
        for i in range(100):
            # make trip to DIKU
            if rain(p):
                if s == 0:  # bring jacket ... 
                    s = move(s)
                    total_cost += 0
                else: total_cost += C
            else: # no rain... so should i bring jacket??!?!?!

                if bring_just_in_case(q):
                    s = move(s)
                    total_cost += U

            # make trip home
            if rain(p):
                if s ==1:
                    # bring jacket ... 
                    s = move(s)
                    total_cost += 0
                else:
                    total_cost += C # if we dont have jacket, we shall be rained upon

            else:
                if bring_just_in_case(q):
                    s = move(s)
                    total_cost += U

        costs.append(total_cost)


    plt.figure()
    plt.plot(qs,costs)
    plt.ylabel('Total costs')
    plt.xlabel('Prob bring just in case')
    plt.show()


def question_4():
    pass

