
from utils.utils_global import *
from utils.utils_orel import *


set_rcParams(style_dict = { 
    'patch.facecolor': (0.4, 0.65, 0.1),
    'axes.facecolor': (0.14, 0.16, 0.3),
    'figure.facecolor': (0.2, 0.05, 0.3),
    'xtick.color': 'white',
    'ytick.color': 'white',
    'text.color': 'white',
    'figure.autolayout': True,
    'axes.labelcolor': "lightgreen" 
    })


def pre_start():
    text_intro = """
    ## Pre course start
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



    def sequential_tutorial():
        # basic operations
        tabs = st.tabs('tensor, activation functions, optimizers, neural network, pre-trained models'.split(', '))


        with tabs[0]: # the tensor
            """
            In pytorch the fundamental object is the tensor; 
            we may make a tensor either from a list or from a numpy array.

            """
            lst = [[1,2],[3,4]]
            x_1 = torch.tensor(lst)
            x_1

            arr = np.array(lst)
            x_2 = torch.from_numpy(arr)
            x_2


            """
            We can also make a tensor, that has the shape of another tensor.
            """

            x_ones  = torch.ones_like(x_2)
            x_ones

            x_rand = torch.rand_like(x_2, dtype=torch.float)
            x_rand


            """
            Just as in numpy we have rand, ones and zeros which all take a shape argument.
            """

            shape = (2,5)
            rand_tensor = torch.rand(shape)
            ones_tensor = torch.ones(shape)
            zero_tensor = torch.zeros(shape)


            rand_tensor
            ones_tensor
            zero_tensor

            """
            Tensor attributes, include
            * shape,
            * dtype
            * device
            """

            tensor = torch.rand(3,4)

            st.write(f"""
                Shape of tensor: **{tensor.shape}**\n
                Datatype of tensor: **{tensor.dtype}**\n
                Device tensor is stored on: **{tensor.device}**""")

            
            'slicing and indexing works just the same as in the numpy'


            """
            to combine tensors we have the following ops"
            * `torch.cat`
            * `torch.stack`
            """



            """
            #### Arithmatic
            * matrix multiplications `tensor_1 @ tensor_2`
            """

        with tabs[1]: #activation functions
            
            '### Activation functions'
            cols = st.columns((1,2))
            cols[0].markdown("""
            * `nn.ReLU`
            * `nn.Softmax`
            * `nn.Sigmoid`
            * `nn.Tanh`
            * `nn.Linear`

            """)
            x = np.linspace(-3,3,100)
            fig, ax = plt.subplots(figsize=(4,3))
            ax.plot(x, nn.ReLU()(torch.from_numpy(x)).numpy(), label='ReLU')
            ax.plot(x, nn.Softmax(dim=0)(torch.from_numpy(x)).numpy(), label='Softmax')
            ax.plot(x, nn.Sigmoid()(torch.from_numpy(x)).numpy(), label='Sigmoid')
            ax.plot(x, nn.Tanh()(torch.from_numpy(x)).numpy(), label='Tanh')
            #ax.plot(x, nn.Linear(1,1)(torch.from_numpy(x)).numpy(), label='Linear')
            ax.set(xlabel='input', ylabel='output', title='Activation functions')
            plt.grid()
            plt.legend()
            plt.close()
            cols[1].pyplot(fig)

        with tabs[2]: #optimizers
            """
            #### Optimizers
            * `torch.optim.SGD`
            * `torch.optim.Adam`
            * `torch.optim.Adagrad`
            * `torch.optim.Adadelta`
            * `torch.optim.RMSprop`
            * `torch.optim.AdamW`
            * `torch.optim.Adamax`
            * `torch.optim.ASGD`
            * `torch.optim.LBFGS`
            * `torch.optim.Rprop`
            * `torch.optim.SparseAdam`
            
            """

        with tabs[3]: # neural network

            cols = st.columns((1,1))
            cols[0].markdown("""
            > We define our neural network by subclassing nn.Module, and initialize the neural network layers in __init__. Every nn.Module subclass implements the operations on input data in the forward method. [[1](https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html)]
            """)

            

            class NeuralNetwork(nn.Module):
                def __init__(self):
                    super(NeuralNetwork, self).__init__()
                    self.flatten = nn.Flatten()
                    self.linear_relu_stack = nn.Sequential(
                        nn.Linear(28*28, 512),
                        nn.ReLU(),
                        nn.Linear(512, 512),
                        nn.ReLU(),
                        nn.Linear(512, 10),
                    )
                    

                def forward(self, x):
                    x = self.flatten(x)
                    logits = self.linear_relu_stack(x)
                    return logits



            cols[0].markdown('''
            Loading data
            ''')
            training_data = datasets.FashionMNIST(
                root="data",
                train=True,
                download=True,
                transform=transforms.ToTensor()
                )

            test_data = datasets.FashionMNIST(
                root="data",
                train=False,
                download=True,
                transform=transforms.ToTensor()
            )

            cols[0].markdown('''We need to use a data loader. The Data loader combines a dataset and a sampler, and provides an iterable over the given dataset.''')

            

            device = 'cpu'
            model = NeuralNetwork().to(device)

            learning_rate = 1e-3
            batch_size = 64

            train_dataloader = DataLoader(training_data, batch_size=batch_size)
            test_dataloader = DataLoader(test_data, batch_size=batch_size)
            try: 
                model.load_state_dict(torch.load('model_weights.pth'))
                epochs = 0
            except:
                epochs = 1
            
            cols[0].markdown('''
            We define the following **hyperparameters** for training:
            * Number of Epochs 
            * Batch Size 
            * Learning Rate''')


            loss_fn = nn.CrossEntropyLoss()

            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


            cols[0].markdown("""Now we define the train and test loops""")

            def train_loop(dataloader, model, loss_fn, optimizer):
                size = len(dataloader.dataset)
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
                        print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


            def test_loop(dataloader, model, loss_fn):
                size = len(dataloader.dataset)
                num_batches = len(dataloader)
                test_loss, correct = 0, 0
                with torch.no_grad():
                    for X, y in dataloader:
                        X, y = X.to(device), y.to(device)
                        pred = model(X)
                        test_loss += loss_fn(pred, y).item()
                        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
                test_loss /= num_batches
                correct /= size
                print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

            cols[0].markdown("""We can now train and test our model:""")
            for t in range(epochs):
                print(f"Epoch {t+1}\n-------------------------------")
                train_loop(train_dataloader, model, loss_fn, optimizer)
                test_loop(test_dataloader, model, loss_fn)
            print("Done!")

            
            torch.save(model.state_dict(), 'model_weights.pth')
            
            # classes of the FashionMNIST dataset
            classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

            cols[1].markdown("""We can now test our model on a image from the test dataset:""")
            test_img_idx = cols[1].slider('Select a test image', 0, 100, 0, 1, key='test_image')
            X = test_data[test_img_idx][0]
            logits = model(X)
            pred_probab = nn.Softmax(dim=1)(logits)
            y_pred = pred_probab.argmax(1)
            
            fig = plt.figure()
            plt.imshow(X.cpu().numpy().reshape(28, 28), cmap='gray')
            plt.title(f"Predicted: {classes[y_pred]}")
            plt.close()
            cols[1].pyplot(fig)

        with tabs[4]: # pre-trained models
            """
            #### pre-trained models
            * `torchvision.models.alexnet`
            * `torchvision.models.densenet`
            * `torchvision.models.inception`
            * `torchvision.models.resnet`
            * `torchvision.models.squeezenet`
            * `torchvision.models.vgg`
            * `torchvision.models.googlenet`
            * `torchvision.models.shufflenet`
            * `torchvision.models.mobilenet`
            * `torchvision.models.mnasnet`
            * `torchvision.models.segmentation`
            * `torchvision.models.video`
            * ...
            """



    def RL_tutorial():
        ''
        '## Cart Pole'
        cols = st.columns((1,2))
        cols[0].markdown('''
        The goal is to balance a pole on a cart. The cart can move left or right, and the pole can rotate. The reward is 1 for every step the pole is balanced, and 0 otherwise. The episode ends when the pole is more than 15 degrees from vertical, or the cart moves more than 2.4 units from the center. The state space is continuous, and the action space is discrete. The state space is 4 dimensional, and the action space is 2 dimensional. 
        ''')
        cols[1].image('assets/Orel/images/plot_learning_progress.png')



        # list all files in the pages/ani directory
        loc = 'assets/Orel/images/ani'
        files = os.listdir(loc)
        
        nums = [int(file.split('.')[0].split('_')[2]) for file in files]
        file_dict = {num:file for num, file in zip(nums, files)}
        
        def view_option_1():
            file = file_dict[st.select_slider('Select an itr number ', sorted(nums))]
            st.video(loc+'/'+file)

        #view_option_1()

        def view_option_2():
            """here i make a grid, using st.columns, and then i put all the videos in the grid"""
            st.write('Here is a grid of all the videos')
            cols = st.columns(4)
            for i, itr in enumerate(sorted(nums)):
                cols[i%4].caption(itr)
                cols[i%4].video(loc+'/'+file_dict[itr], format='mp4') 

        '---'
        view_option_2()



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
        print('page loaded')

def multi_armed_bandit():
    ''
    
    st.markdown(r"""
    # Multi armed bandit

    [link to youtube vid](https://www.youtube.com/watch?v=e3L4VocZnnQ)

    We are a professor staying in a small town for 300 days. In the town there are 3 restaurants. They bring different amounts of happiness per meal, but at the beginning we dont know.

    We have to balance two concepts; exploration and exploitation.

    We measure the goodness of a strategy by the regret, $\rho$. Regret is the difference from the maximal expectation value happiness.
    """)

    '---'
    cols = st.columns((1,1))
    cols[0].markdown("""
    ### Greedy
    We choose the restaurant which historically has yielded the highest average happiness. However $\epsilon$\% of the time we choose a random restaurant.
    """)

    restaurants = { # (mu, sigma)
        "City wok"      : (8, 1),
        "McDonald's"  : (5,3),
        "Da Cavalino" : (10,3)}


    scores, epls = many_bandit_runs(restaurants, n_epsilons=10, n_exp=2,  method='standard')
    cols[1].pyplot(show_bandit_scores(scores, epls))


    '---'

    cols = st.columns((1,1))
    cols[0].markdown(r"""
    ### Upper confidence bound (UCB)

    We may obtain a better algortihm by adressing a central flaw
    > When comparing means, these may have very different numbers of samples.

    The new mean is defined as;
    $$
    \mu_r = \hat{\mu_r} + \sqrt{\frac{2\ln(t)}{N_t(r)}},
    $$
    maximize this, i.e., maximize Hoffding's inequality.
    where $t$ is the iteration step, and $N_t(r)$ is the number of times restaurant $r$ has been visited so far.

    """)
    scores, epls = many_bandit_runs(restaurants, n_epsilons=10, n_exp=2,  method='UCB')
    cols[1].pyplot(show_bandit_scores(scores, epls))

def week1_notes():
    st.header('Notes lecture 1')
    cols = st.columns((2,1))
    cols[0].markdown(r"""

    Batch versus Online. Batch is great if we dont need to update dynamically...

    > A key assumption we make is that the new data is from the same distribution as the data on which we trained. And samples are i.i.d.

    Example of online learning: Stock market investing, spam filtering, interweb suggestions, and others 

    """) # intro
    cols[1].image('https://billig-billy.dk/cache/8/7/2/2/0/5/bandit-hat-fit-500x500x100.webp')

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

def lecture2_notes():
    """"""
    r"""
    ## Lecture 2 notes (9 feb 2023)
    In the stateless seeting: we have a loss matrix which looks like the following:
    $$
    \begin{bmatrix}
        l_{1,1} & l_{2,1} & \ldots & l_{t,1}\\
        l_{1,2} & l_{2,2} & \ldots & l_{t,2}\\
        \vdots & \vdots & \vdots & \vdots \\
        l_{1,k} & l_{2,k} & \ldots & l_{t,k}\\
    \end{bmatrix}
    $$
    Notice, we have $k$ actions in our action space.
    ### Performance meaure:
    Regret:
    $$
        R_T = \sum_{t=1}^T l_{t, A_t} - \min_a \sum_{t=1}^T l_{t,a}
    $$
    The above equation describes the regret as the loss of the algorithm minus the best action applied contiously, i.e., the best row of the loss matrix in hindsight.

    It the regret is order $T$ $\Rightarrow$ no learning. We want sublinear regret, which means we are learning something.

    #### Expected regret
    $$
        \mathbb{E}[R_T] = \mathbb{E}[\sum_{t=1}^T l_{t, A_t}] - \mathbb{E}[\min_a \sum_{t=1}^T l_{t,a}]
    $$


    ##### Oblivious adversary
    $l_{t,a}$ are independent of actions. Meaning the adversary is not able to know our actions.
    
    We will only consider oblivious adversary, in which case the last term in the equation above becomes deterministic.

    ##### Adaptive adversary
    $l_{t,a}$ may depend on actions.

    
    #### Pseudo regret
    only defined in i.i.d. setting

    $$
        \bar{R}_T = \mathbb{E}[\sum l_{t, A_t}] - min_a \mathbb{E}[\sum l_{t,a}]
    $$
    notice, here we have the minimum of the expectation rather than the expectation of the minimum.
    $$
    \begin{align*}
        \bar{R}_T &= \mathbb{E}[\sum l_{t, A_t}] - min_a \mu(a)T\\
         &= \mathbb{E}[\sum l_{t, A_t} - \mu^*]\\
         &= \mathbb{E}[\sum_t^T \Delta(A_t)]\\
         &= \sum_a \Delta(a) \mathbb{E}[N_T(a)]
    \end{align*}
    $$
    where $\mu^* = \min_a \mu(a)$.

    We also define $\Delta(a) = \mu(a)- \min \mu(a) = \mu(a) - \mu^*$
    


    Notice, the pseudo regret is always less than or equal to the expected regret. We always use pseudo-regret for i.i.d. :shrug.

    The reason for this is that the comparator is more reasonable. The comparator we are using is $T\mu^*$...


    """


    r"""
    ---
    Now we shall consider the iid bandit case (remember the bandit case is one where feedback is limited).

    #### Exploration-exploitation trade-off
    * action space width = 2
    * T = known total time
    * Delta is known

    The actions yield $1/2 \pm \Delta$ respectively. If $\Delta$ is small, we should explore for a while, before determining which action is best. And then after that just repeat that action. 


    We may bound the pseudo regret by:
    $$
        \bar{R}_T \leq \frac{1}{2}\epsilon T\Delta + \delta(\epsilon) \Delta (1-\epsilon)T \leq (\frac{1}{2}\epsilon + \delta(\epsilon))\Delta T
    $$
    In which $\delta(\epsilon)$ is the probability that we selected the suboptimal action. We can bound this by:
    $$ 
        \leq \mathbb{P} (\hat{\mu_{\epsilon T}} (a) \leq \hat{\mu_{\epsilon T}} (a^*))
    $$ 
    We can bound this chance of having chosen a bad arm by:
    $$
        \leq \mathbb{P} (\hat{\mu_{\epsilon T}} (a^*) \geq 
        \hat{\mu_{\epsilon T}} (a^*) + \frac{1}{2}\Delta)
        + \mathbb{P}(\hat{\mu_{\epsilon T}} (a) \leq \mu(a) - \frac{1}{2}\Delta)
    $$
    We may employ Hoeffding's inequality to bound the first term:
    $$
        \leq 2 e^{- \epsilon T \Delta^2 / 4}
    $$
    So now we can choose an epsilon to minimize the probability of choosing the bad arm.
    $$
        \epsilon^* = \frac{4\ln (T\Delta^2)}{T\Delta^2}
    $$
    yielding a pseudo regret of:
    $$
        \bar{R}_T \leq \frac{2(\ln(T\Delta^2)+1)}{\Delta}
    $$

    To summarize the above, it takes a longer time to dicern which action is better if the actions are closer together. It goes with $\Delta^2$. Each time we choose the wrong action, we lose $\Delta$, so pseudo regret goes with $\Delta$.
    """

    

    T = 100
    action_space = [0,1]
    Delta = .1
    reward_func = lambda a : 1/2 + [-1,1][a] * np.random.normal(0,Delta,None)

    def explore(action_space, reward_func):
        action = np.random.choice(action_space)
        return action, reward_func(action)

    def exploit(experiences, reward_func):
        action = max(experiences, key=lambda a: np.mean(experiences[a]))
        return action, reward_func(action)

    experiences = {}



    r"""
    #### Lower confidence bound (LCB)
    $$
        L_t^{CR} = \hat{\mu_{t-1}}(a) - \sqrt{\frac{3\ln t}{2 N_{t-1}(a)}}
    $$
    This has the following algorithm:
    ```
    play each arm once (i.e., K plays)
    for t = K to T:
        A_t = argmin_a L_t^{CR}(a)
    ```
    """

def lecture3_notes():
    """"""

    r"""
    # Markov Decision Process (MDP)
    #### FROM YT VIDS
    [Intro to MDP video by Computerphile](https://www.youtube.com/watch?v=2iF9PRriA7w)
    [David silver from Deepmind om MDP](https://www.youtube.com/watch?v=lfHX2hHRMVQ)

    A simple case is where we have a fully observable environment.

    #### Markov Property
    The future is independent of the past given the present.
    $$
        \mathbb{P}\left[
            S_{t+1} | S_t \right] = 
            \mathbb{P} \left[
                S_{t+1} | S_1, \ldots, S_t \right]
    $$

    #### State transition matrix
    The probability of transitioning from state $s$ to state $s'$ is given by:
    $$
        P_{ss'} = \mathbb{P} \left[ S_{t+1} = s' | S_t = s \right]
    $$
    This is a matrix where the rows are the starting states and the columns are the ending states.

    #### Reward function
    The reward function is a function that maps states to rewards. It is denoted by $R$.

    #### Discount factor
    The discount factor is a number between 0 and 1. It is denoted by $\gamma$. It is used to discount future rewards. This is useful because we may want to prioritize immediate rewards over future rewards.

    The return $G_t$ is the total discounted reward from time $t$ to the end of the episode. It is given by:
    $$
        G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \ldots = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}
    $$ 


    
    
    """

    '---'
    
    r"""
    ## Theory of discounted MDP
    Discounted MDPs are described by a tuple $(S, A, P, R, \gamma)$ where:
    - $\mathcal{S}$ is the set of states
    - $\mathcal{A}$ is the set of actions; $\mathcal{A} = \cup_{s\in S} \mathcal{A}_s$
    - $P$ is the state transition function
    - $R$ is the reward function
    - $\gamma$ is the discount factor

    #### The algorithm
    The algorithm is given by:
    * agent observes state $s_t$ and takes an action $a_t \in \mathcal{A}_{s_t}$.
    * agent receives reward $r_t := r(s_t, a_t \sim R(s_t, a_t)$ and observes new state $s_{t+1}\sim P(\cdot|s_t, a_t)$.
    * repeat until $s_{t+1} = \text{terminal}$
    Notice the dot in $P(\cdot|s_t, a_t)$, this denotes an arbitrary state.
    """
    st.image("https://www.researchgate.net/publication/350130760/figure/fig2/AS:1002586224209929@1616046591580/The-agent-environment-interaction-in-MDP.png")
    r"""
    This above process produces a trajectory (AKA history) 
    $$
    H_t = (s_0, a_0, s_1, a_1, \ldots, s_t, a_t)
    $$
    Although a decision process may in theory depend on the entire history. In MDPS we adhere to the Markov property, which states that the future is independent of the past given the present. This means that the state transition function only depends on the current state and action. This is given by:
    $$
        P(s_{t+1} =s' | s_0, a_0, s_1, a_1, \ldots, s_t, a_t) = P(s_{t+1} =s' | s_t, a_t)\\
        \Rightarrow\\
        R(s_1, a_1, s_2, a_2, \ldots, s_t, a_t) = R(s_t, a_t)
    $$
    """

    r"""
    ### Classification of MDPs
    * Finite-horizon MDPs; with objective function: 
    $$
    \max \mathbb{E} \left[ \sum_{t=1}^{N-1} r(s_t, a_t) + r(s_N) \right]
    $$
    * Infinite-horizon discounted MDPs; with objective function: 
    $$
    \max \mathbb{E} \left[ \sum_{t=1}^{\infty} \gamma^{t-1} r(s_t, a_t) \right]
    $$
    * Infinite-horizon undiscounted MDPs; with objective function:
    $$
    \max \lim_{N\rightarrow\infty}\frac{1}{N} \mathbb{E} \left[ \sum_{t=1}^{N} r(s_t, a_t) \right]
    $$
    """
    "---"
    r"""
    > For todays lecuter we will focus our attention on infinite-horizon discounted MDPs.
    
    #### The L-state RiverSwim MDP
    The L-state RiverSwim MDP is given by:
    * $\mathcal{S} = \{1, 2, \ldots, L\}$
    * $\mathcal{A} = \{ \text{left}, \text{right} \}$
    * $P(s_{t+1} = s' | s_t, a_t) = \begin{cases} 
        0.6 & \text{if } s' = s_t = s_1 \text{ and } a_t = \text{right} \\
        .4 & \text{if } s' = s_t + 1 \text{ and } a_t = \text{right} \\
        .55 & \text{if } s' = s_t + 0 \text{ and } a_t = \text{right} \\
        .05 & \text{if } s' = s_t-1 \text{ and } a_t = \text{right} \\
         1 & \text{if } s' = s_t - 1 \text{ and } a_t = \text{left} \\
        0.6 & \text{if } s' = s_t = s_L \text{ and } a_t = \text{right} \\
         0 & \text{otherwise} \end{cases}$
    * $R(s_t, a_t) = \begin{cases}
        0.05 & \text{if } s_t = 1 \text{ and } a_t = \text{left} \\
        1 & \text{if } s_t = L \text{ and } a_t = \text{right} \\
        0 & \text{otherwise} \end{cases}$

    Now lets calculate the objective function
    $$
    \max \mathbb{E} \left[ \sum_{t=1}^{\infty} \gamma^{t-1} r(s_t, a_t) \right]
    $$
    
    ### Policy
    A policy is a mapping from states to actions. We denote a policy by $\pi$. We denote the action taken by $\pi$ in state $s$ by $\pi(s)$. A policy my be;
    * deterministic or stochastic
    * history dependent or stationary

    table:
    | | deterministic | stochastic |
    | --- | --- | --- |
    | history dependent | $\pi : \mathcal{H}_t\rightarrow \mathcal{A}$ | $\pi : \mathcal{H}_t\rightarrow \Delta(\mathcal{A})$|
    | stationary | $\pi :\mathcal{S}\rightarrow \mathcal{A}$ | $\pi : \mathcal{S}\rightarrow \Delta(\mathcal{A})$|

    $$
    \begin{align*}
        r^\pi &\in \mathbb{R}^{\mathcal{S}}\\
        r^pi(s) &= \sum_{a\in\mathcal{A}} R(s,a) \pi(a|s)
    \end{align*}
    $$

    $$
        P_{s,s'}^\pi = \sum_{a\in\mathcal{A}} P(s'|s,a) \pi(a|s)
    $$

    ### Value function
    The value function is a mapping from states to real numbers. We denote the value function by $V$. We denote the value of state $s$ by $V(s)$. The value function is defined as:
    $$
    V(s) = \mathbb{E}^\pi \left[ \sum_{t=1}^{\infty} \gamma^{t-1} r(s_t, a_t) | s_0 = s \right]
    $$
    Where $\mathbb{E}^\pi$ denotes the expectation with respect to the policy $\pi$. 

    The Value function lets us get the potential for reward. Kinda like giving us potential of reward when we are close to a rewarding state.

    ### Policy evaluation
    Policy evaluation is the process of computing the value function for a given policy. We need this to go from our complex objective to an interpretable value function.
    
    We have three methods:
    * Direct computation
    * Iterative policy evaluation
    * Monte Carlo policy evaluation

    #### Direct computation
    We can do this by using the Bellman equation:
    $$
    V^\pi(s) = \mathbb{E}_{a\sim \pi(s)}[
        r(s,a)] + \gamma \mathbb{E}_{a\sim \pi(s)}\left[
            \sum_{x\in\mathcal{S}} P(x|s,a) V^\pi(x)
        \right]
    \\
    V^\pi = r^\pi + \gamma P^\pi V^\pi
    $$
    This is invertible:
    $$
        V^\pi = (I - \gamma P^\pi)^{-1}r^\pi
    $$
    The Bellman operator is:
    $$
        T^\pi f:= r^\pi + \gamma P^\pi f
    $$
    inserting the value function into the Bellman operator gives us:
    $$
        V^\pi = T^\pi V^\pi
    $$
    In other words; $V^\pi$ is the unique fixed point of the Bellman operator $T^\pi$.

    #### Iterative policy evaluation
    We can also use iterative policy evaluation. We can do this by applying the bellman operator to the value function iteratively. We do this until we reach a fixed point.

    ### Optimal value function
    The optimal value function is the value function for the optimal policy. We denote the optimal value function by $V^*(s)$. Notice $V^{\pi^*(s)} = V^*(s)$.

    another important theorem says:
    suppose the state space $\mathcal{S}$ is finite. Then there exists a policy $\pi^* \in \prod^{SD}$.
    So we can restrict our attention to $\prod^{SD}$.
    (SD = stationary deterministic)

    ### Major solution methods
    * Value iteration
    * Policy iteration
    * linear programming

    ### Value iteration
    ```python
    input epsilon > 0$
    initialize V_0(s) = 0# for all states ($s\in\mathcal{S}$)
    $V_1 = r_max / (1-gamma)$
    n=0
    while $||V_{n+1} - V_n|| > \epsilon \frac{1-\gamma}{2\gamma}$
        update for $ s\in\mathcal{S}$
            $V_{n+1}(s) = \max_{a\in\mathcal{A}} \left[ r(s,a) + \gamma \sum_{x\in\mathcal{S}} P(x|s,a) V_n(x) \right]$
        $n = n+1$
    return $V_n$
    ```

    ### Policy iteration
    * select an initial policy $\pi_0$ and $\pi_1$ arbitrarily ($ \pi_0\neq \pi_1 $). and set n=0
    * while $\pi_n \neq \pi_{n+1}$
        * $V_n = \text{policy evaluation}(\pi_n)$. $(I - \gamma P^\pi_n)V_n = r^{\pi_n}$
        * $\pi_{n+1} = \text{policy improvement}(V_n)$. $\pi_{n+1}(s) = \arg\max_{a\in\mathcal{A}} \left[ r(s,a) + \gamma \sum_{x\in\mathcal{S}} P(x|s,a) V_n(x) \right]$ for all $s\in\mathcal{S}$
        * $n = n+1$
    * return $V_n, \pi_n$
    """
    # python code for L-state RiverSwim MDP
    
    def river_swim(L, gamma):
        pass

    def example___():
        """
        # example
        We reciveve orders with probability $\alpha$. We can either process all orders or process none.
        * the cost per unfilled order per period i $c>0$ and the setup cost to process unfilled orders is $K>0$.
        * Assume the total number of orders that can remain unfilled is $n$
        * assume a discount factor $\gamma < 1$
        """
        cols = st.columns(2)
        c = cols[0].slider("cost per unfilled order, c", 0.0, 1.0, 0.1, 0.1)
        K = cols[0].slider("setup cost to process unfilled orders, K", 0.0, 1.0, 0.1, 0.1)
        n = cols[0].slider("total number of orders that can remain unfilled, n", 0, 100, 10, 10)
        alpha = cols[0].slider("probability of receiving an order, alpha", 0.0, 1.0, 0.5, 0.1)
        gamma = cols[0].slider("discount factor, gamma", 0.0, 1.0, 0.9, 0.1)
        

        T = 300
        prob_fill = np.linspace(0.01,.31,20)
        costs = []
        for p in prob_fill:
            total_cost = 0
            number_of_unfilled_orders = 0
            for t in range(T):
                # get order?
                if np.random.rand() < alpha:
                    number_of_unfilled_orders += 1
                    if number_of_unfilled_orders > n:
                        number_of_unfilled_orders = n
                # process orders?
                if number_of_unfilled_orders>0 and np.random.rand() < p:
                    number_of_unfilled_orders = 0
                    total_cost += K
                else:
                    total_cost += number_of_unfilled_orders*c
            costs.append(total_cost)
        fig = plt.figure()
        plt.plot(prob_fill, costs)
        plt.xlabel("probability of filling order")
        plt.ylabel("cost")
        plt.title("cost as a function of probability of filling order")
        cols[1].pyplot(fig)
        plt.close()

        r"""
        but here we didnt include the discount factor
        


        """

def lecture_feb_23_notes():
    ''
    r"""
    It seems I missed a lecture this morning...
    > topic was: Stochastic bandits + Adversarial full info

    ---
    # Policy Evaluation from data

    A Markov decision procces becomes a markov reward process if we choose a policy.

    Let's remember the valuefunction associated with a policy $\pi$:
    $$
        V(a) = \mathbb{E} \left[ \sum_{t=0}^\infty \gamma^t r_t | s_0 = a \right]
    $$


    ### Policy evaluation
    * we may asses this with the Bellman equation, which as we recall is invertible.
    $$
    V^\pi = r^\pi +  \gamma P^\pi V^\pi 
    \Rightarrow 
    V^\pi = (I - \gamma P^\pi)^{-1} r^\pi
    $$


    """
    #Let's have a look at the riverSwim vizualization:
    L = st.slider("number of states", 2, 20, 5, 1)
    states = np.arange(L)
    transitions = riverSwim_transtions(L)
    actions = np.arange(2)
    reward = np.zeros(L); reward[-1] = 1
    mdp = MDP(states, actions, transitions, reward, gamma=None, state_names=None, action_names=None, policy=None)
    mdp
    

    r"""
    # Policy Evaluation from data
    The data here indicates that we donty have full information regarding the MDP. We only have access to the state and the reward. We can use this to estimate the value function.

    Online versus offline; we can either update the value function after each step, or after each episode.

    ## Specifics
    * Given a dataset $\mathbb{D}$ collected from an unknown MDP with policy $\pi$.
    $$
        \mathbb{D} = \left\{
            (s_t, a_t, r_t), \quad t=0,1,\dots,T
            \right\}
    $$

    ### Off-policy evaluation (OPE)
    apply a policy $\pi_b$ called the loggig policy to collect the data. This allows us to estimate the target policy, $\pi$.

    ###### Should we switch
    if $V^{\pi_B} > V^{\pi_A}$ then we should switch policies to $\pi_B$. (we need some margin here.)

    Note; one can find the unknown $V^{\pi_B}$ using the data collected with $\pi_A$.

    
    ### On-policy optimization (OPO)
    ...

    ## Methods
    #### Model-based
    If we know the MDP, we can use Bellmans equation to estimate the value function.
    If the policy is stationary deterministic, an MRP is induced and we can use the Bellman equation to estimate the value function.
    $$
        V^\pi = r^\pi +  \gamma P^\pi V^\pi \Rightarrow V^\pi = (I - \gamma P^\pi)^{-1} r^\pi
    $$

    But we dont know $P^\pi$, so we need to estimate it.
    $$
        \hat{P}^\pi_{s,s'} = \frac{N(s,s') + \alpha}{N(s)+ \alpha S}
    $$
    in which $N(s,s')$ is the number of times we have been in state $s$ and ended up in state $s'$. $N(s)$ is the number of times we have been in state $s$. $\alpha$ is a smoothing parameter, and $S$ is the number of states.

    $\alpha$ i.e. the smoothing can be picked as:
    * $\alpha \geq 1$; arbitrary smoothing
    * $\alpha = 0$; no smoothing - corresponds to maximum likelihood estimation.
    * $\alpha = 1/S$; Laplace smoothing - corresponds to Bayesian estimation. (biased, but bias vanishes as $N(s) \rightarrow \infty$)

    We also need to estimate the reward function:
    $$
        \hat{r}^\pi (s) = \frac{\alpha + \sum_{t=1}^T r_t \mathbb{I}\{s_t=s\}}{\alpha + N(s)}
    $$
    Again, this should converge, meaning the approximated value function should converge;
    $$  
        \mathbb{P}\left(
                    \lim_{N \rightarrow \infty} \hat{V}^\pi = V^\pi
            \right) = 1,
    $$
    given that the policy is sufficiently exploratory.


    We have yet to consider confidence intervals for different time-steps. 


    Computationa complexity: $O(S^3)$
    #### Model-free

    ###### Temporal Difference Learning (TD)
    Again, we keep the policy constant; and we have a dataset which looks like:
    $$
        \mathbb{D} = \left\{
            (s_t, a_t, r_t), \quad t=0,1,\dots,n
            \right\}
    $$
    We want ot obtain;
    $$
        \mathbb{E}\left[r_t + \gamma V(s_{t+1})\right]
    $$
    But this is all random. So we insert the givens;
    $$
        \mathbb{E}\left[r_t + \gamma V(s_{t+1}) | s_t, \hat{V} \right] = 
        \mathbb{E}_{a\sim \pi(s_t)} \left[R(s_t, a) + \gamma \sum_{s'} P(s'| s_T, a)  \hat{V}(s')|s_t, \hat{V}\right]
    $$
    If $\hat{V}(s_t) = \hat{V}(s_{t+1})$ we are golden. But this is often not the case.
    So we update the value function after each step;
    $$
        \hat{V}(s_t) \leftarrow \hat{V}(s_t) + \alpha_t \left[r_t + \gamma \hat{V}(s_{t+1}) - \hat{V}(s_t)\right]
    $$
    in which $\alpha_t$ is the learning rate. This is called the TD(0) algorithm.

    This is an instance of **bootstrapping**, because we use the value function to estimate itself.

    algorithm:
    ```Python
    input: D, alpha, gamma
    output: V
    ---
    select V arbitrarily
    for each (s_t, a_t, r_t) in D:
        V(s_t) <- V(s_t) + alpha * (r_t + gamma * V(s_{t+1}) - V(s_t))
    return V
    ```
    Notice, alpha my be time dependent. How to choose it? It should be square summable but not summable. (i.e. 
    $$
    \sum_{t=1}^T \alpha_t^2 < \infty
    $$ 
    but 
    $$
    \sum_{t=1}^T \alpha_t = \infty
    $$

    > (Something something Robbins-Monro conditions)...

    Examples of typicals alphas include:
    * $\alpha_t = \frac{1}{t+1}$
    * $\alpha_t = \frac{2}{\sqrt{\log(t) \ldots}}$
    * $\ldots$

    ###### Convergence:


    The computational complexity of the algorithm is $O(1)$, and the space complexity is $1$.

    TD is also great, cus we can just build on it if we are given more data (we call this kinda method **incremental**).


    ###### TD($\lambda$)
    TD(0) is a special case of TD($\lambda$) in which $\lambda = 0$.
    TD($\lambda$) is a generalization of TD(0) in which we use the value function to estimate itself, but we also use the value function to estimate itself in the future. This is called **eligibility traces**.

    $$
        \hat{V}(s_t) \leftarrow \hat{V}(s_t) + \alpha_t \sum_l^\infty (1-\lambda) \lambda^{l-1} \delta_t^l\\
         = \hat{V}(s_t) + \alpha_t \sum_{n=0}^\infty \lambda^n \delta_{n} (
            r_{t+n+1} + \gamma \hat{V}(s_{t+n+1}) - \hat{V}(s_{t+n})
        )
    $$

    if lambda is 0, we recover TD(0). If lambda i one, we get monte carlo sampling.
    """


def lectureNotes_march_02():
    ''
    """

    # Lecture Notes - March 2nd
    We have introduced a whole lot of different frameworks...
    """
    st.image('https://miro.medium.com/max/1400/1*ywOrdJAHgSL5RP-AuxsfJQ.png')
    
    # Cold-open...
    st.markdown("""
    
    We have a couple different types of MDPs:
    * Episodic
    * Discounted
    * Average reward

    If the MDP is known, then we can just do policy iteration or value iteration. But if the MDP is unknown, then we need to do model-free RL. We can for example use policy evaluation from data, or we can do off-policy evaluation (not covered in this course), or we can do off-policy optimization (which we do do). Notice, that all the methods mentioned for solving unknown MDPs are off-policy.
    """)

    '---'


    # What is off-policy optimization, and what is Q-learning?
    st.markdown(r"""
    ## Off-policy optimization and Tabular Q-learning
    The focus of today is is off-policy optimization, which is Q-learning. 
    > Notice, this method does not desal with online learning, which is a different topic.
    
    > Also notice, we are still in the kingdom of discounted MDPs.

    > Off-policy referes to the fact that we are trying a policy $\pi_b$ that is different from the optimal policy $\pi^*$. 

    Given some data $\mathbb{D}$ collected under some policy $\pi_b$, we want to find the optimal policy $\pi^*$.
    
    The action-value function ($Q$-function) is defined as:
    $$
        Q^\pi(s, a) = \mathbb{E}^\pi\left[
            \sum_{t=1}^\infty \gamma^{t-1} r(s_t, a_t) | s_0 = s, a_0 = a
            \right]
    $$
    meaningl $Q^\pi(s, a)$ is the expected (future) return of starting in state $s$ and taking action $a$ under policy $\pi$.

    Notice the optimal Q-functiuon is related to the optimal value function by:
    $$
        V^*(s,) = \max_{a\in\mathcal{A}} Q^*(s, a)
    $$
    So we may use the Q-function to write the Bellman optimality equation:
    $$
        Q^*(s, a) = R(s, a) + \gamma \sum_{x\in\mathcal{S}} P(x|s,a) \max_{a'\in\mathcal{A}} Q^*(x, a')
    $$

    We also have the Optimal Bellman Operator:
    $$
        \mathcal{T} f(s,a) := R(s, a) + \gamma \sum_{x\in\mathcal{S}} P(x|s,a) \max_{a'\in\mathcal{A}} f(x, a')
    $$

    The optimal Q-function is the unique fixed point of the Optimal Bellman Operator:
    $$
        Q^*(s, a) = \mathcal{T} Q^*(s, a)
    $$

    So we can use fixed point iteration to find the optimal Q-function: *see scientific computing*.

    Our prof. suggestes the method: Robbins-Monro Algorithm:
    $$
        x_{n+1} = x_n - \alpha_n \nabla f(x_n) = x_n - \alpha_n (h(x_n) + \epsilon_n)
    $$
    Satisfaction of Robbins-Monro conditions:
    $$
    \begin{align}
        \sum_{n=1}^\infty \alpha_n = \infty,
    &&
        \sum_{n=1}^\infty \alpha_n^2 < \infty
    \end{align}
    $$
    
    Below is a demo of the Robbins-Monro algorithm:
    """)  

    # lets implement this
    def stochastic_approximation_demo(n=69, alpha=0.1, x0=0.5):
        def f(x):
            return x**2 -1

        def noisy_f(x):
            return f(x) + np.random.normal(0, 0.1)
        xs = [x0]
        for _ in range(n):
            xs.append(xs[-1] + alpha * noisy_f(xs[-1]))
        
        fig = plt.figure(figsize=(8,3))
        ax = fig.add_subplot(111)
        ax.set_title('Stochastic approximation of $f(x) = x^2 - 1$')
        ax.plot(xs)
        ax.set_xlabel('iteration')
        ax.set_ylabel('x')
        st.pyplot(fig)

    
    stochastic_approximation_demo()
    r"""
    How do we apply this to Q-learning? I.e., SA for $F(Q) = \mathcal{T}Q-Q$?

    $$
        \mathbb{E}[Y_t|Q, s_t, a_t] = \mathcal{T}Q(s_t, a_t) - Q(s_t, a_t) = F(Q)(s_t, a_t)
    $$
    > (I skipped a couple steps... sorry)

    Specifically to update the Q-function, we use the following update rule:

    $$
        Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha_t \left( r_{t} + \gamma \max_{a'\in \mathcal{A}_{s_t}} Q(s_{t+1}, a') - Q(s_t, a_t) \right)
    $$

    """
    # Model-based method for OPO
    st.markdown(r"""
    ---
    ### Model-based method for OPO
    The model is unknown, but we have Data $\mathbb{D}$, which is a sequence of tuples $(s_t, a_t, r_{t+1}, s_{t+1})$.
    
    The idea is is to estimate the MDP using data and apply the centainty equivalence principle.
    - step 1: Compute estimate $\hat{P}$ of the transition probabilities and $\hat{R}$ of the reward function.
    - step 2: Compute the optimal value function $V^*$ using the Bellman optimality equation.


    > **pro tip**: replace $N(s,a) = \max\{ 1, \sum_{s' \in \mathcal{S}} N(s,a,s') \}$. So we avoid division by zero.

    Smootherd estimator for $P$:
    $$
        \hat{P}(s' | s, a) = \frac{N(s,a,s')+\alpha} {N(s,a)+\alpha S}
    $$

    Smootherd estimator for $R$:
    $$
        \hat{R}(s, a) = \frac{\ldots}{\ldots}
    $$
    > different value of $\alpha$: we did this last lecture, see notes from feb. 23.

    When we have estimated $P$ and $R$ by $\hat{P}$ and $\hat{R}$, we just compute the bellman optimality equation as usual. OFC we obtain $\hat{Q}^*$ instead of $Q^*$. This in turn lets us outout the optimal policy $\widehat{\pi}^*$.

    This gets asymptotic convergence to the optimal policy $\pi^*$ (almost surely). Its is obviously required that $\pi_b$ is sufficiently exploratory.

    > Since we progress asymptotically, and there is some cost associated with data collection, we should set a reasonable stopping criterion.


    We can bound the error in the estimates of $P$ and $R$ by Hoeffding's inequality:
    $$
        \left| \hat{P}(s' | s, a) - P(s' | s, a) \right| 
        \leq \sqrt{\frac{N(s,a)^2}{2(N(s,a) +\alpha S)} \log \frac{2S^2 A}{\delta}} + \frac{\alpha^2 S}{N(s,a), +\alpha S}
    $$
    and similarly for $\hat{R}$.

    This lets us bound the error in the optimal value function:
    $$
        \text{dist}\left( M, \widehat{M}\right) \leq O(1/\sqrt{n})
    $$
    This informs us of how many samples we need to get to say 95% or 99% confidence in the estimate of the optimal value function.

    **Pros and Cons**
    - Cons:
        - If we do it many times, we get high variance in $Q^*$.
        - Computational complexity of computing $Q^*$ is $O(S^3)$.
        - if we get a new sample, we need to recompute $Q^*$.
    * Pros:
    """)

    # Behaviour policy
    st.markdown(r"""
    Use epsilon greedy in conjunction with Q-learning

    $$
        \pi_{\epsilon - \text{greedy}}(s) = 
        \begin{cases}
            \text{argmax}_a Q_t(s,a) & \text{with probability } 1 - \epsilon \\
            \text{uniform random from }\mathcal{A} & \text{with probability } \epsilon
        \end{cases}
    $$
    Note; this is non-stationary. Choosing $\text{argmax}_a Q_t(s,a)$ is the exploitation part, and choosing uniformly at random is the exploration part.
    """)

    # Promotion function
    st.markdown(r"""
    Promote actions which have not been tried bery often from state $a$. This is done by adding a bonus to the Q-value of the action. This bonus is called the promotion function.
    $$
        B_{t+1}(s,a) = \begin{cases}
            0 && \text{if} (s,a) = (s_t, a_t) \\
            B_t(s,a) + \psi(n(s,\neq a)) && \text{if} s=s_t \text{but} a \neq a_t \\
            B_t(s,a) && \text{otherwise}
        \end{cases}
    $$
    """)
    # A couple term to read up on
    st.markdown(r"""
    ### A couple terms to read up on
    - **direct method versus indirect methdod**: The above described method is direct because we dont calculate the model.
    - **synchronous method versus asynchronous method**: the method just described is an asynchronous method. 
    - Ergoticity: an MDP is ergodic if every state is reachable from every other state under any policy.
    - In Deep Q-learning, a deep neural network is used to approximate the Q-values instead of using a lookup table
    """)
    # quote: all models are wrong, but some are useful. -- George Box

    # CHAT GPT DESCRIPTION OF Q-LEARNING
    r"""
    ---
    ---
    ---
    ## ChatGPTs description of Q-learning 
    To implement $Q$-learning for the riverswim MDP using a history of actions and rewards, you would need to first initialize a $Q$-table, which is a table that stores the $Q$-values for each state-action pair. You can use the following steps to update the $Q$-table:
    1. Initialize the $Q$-table with random values.
    1. For each episode, initialize the agent's position to a random starting location.
    1. While the agent has not reached the goal location or exceeded the maximum number of time steps:
        1. Use an exploration policy, such as epsilon-greedy or softmax, to select an action based on the current Q-values.
        1. Apply the selected action and observe the resulting reward and next state.
        1. Update the Q-value for the current state-action pair using the Q-learning update rule:
        $$
            Q(s, a) = Q(s, a) + \alpha * (r + \gamma * \max(Q(s', a')) - Q(s, a))
        $$
        where $s$ is the current state, a is the selected action, r is the observed reward, s' is the next state, alpha is the learning rate, and gamma is the discount factor.
    1. Repeat steps 2-3 for a specified number of episodes or until the Q-values converge.

    Once the Q-table has been updated, you can use it to select the optimal action for any given state by selecting the action with the highest Q-value.
    """

def cart_pole():
    #RL.py

    

    env = gym.make("CartPole-v1")

    # set up matplotlib
    is_ipython = 'inline' in matplotlib.get_backend()
    if is_ipython:
        from IPython import display


    plt.ion()


    # if gpu is to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Transition = namedtuple('Transition',
                            ('state', 'action', 'next_state', 'reward'))


    class ReplayMemory(object):

        def __init__(self, capacity):
            self.memory = deque([], maxlen=capacity)

        def push(self, *args):
            """Save a transition"""
            self.memory.append(Transition(*args))

        def sample(self, batch_size):
            return random.sample(self.memory, batch_size)

        def __len__(self):
            return len(self.memory)


    class DQN(nn.Module):

        def __init__(self, n_observations, n_actions):
            super(DQN, self).__init__()
            self.layer1 = nn.Linear(n_observations, 128)
            self.layer2 = nn.Linear(128, 128)
            self.layer3 = nn.Linear(128, n_actions)

        # Called with either one element to determine next action, or a batch
        # during optimization. Returns tensor([[left0exp,right0exp]...]).
        def forward(self, x):
            x = F.relu(self.layer1(x))
            x = F.relu(self.layer2(x))
            return self.layer3(x)

    BATCH_SIZE = 128 # BATCH_SIZE is the number of transitions sampled from the replay buffer
    GAMMA = 0.99 # GAMMA is the discount factor as mentioned in the previous section
    EPS_START = 0.9 # EPS_START is the starting value of epsilon
    EPS_END = 0.05 # EPS_END is the final value of epsilon
    EPS_DECAY = 1000    # EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
    TAU = 0.005 # TAU is the update rate of the target network
    LR = 1e-4 # LR is the learning rate of the AdamW optimizer

    # Get number of actions from gym action space
    n_actions = env.action_space.n
    # Get the number of state observations
    state, info = env.reset()
    n_observations = len(state)

    policy_net = DQN(n_observations, n_actions).to(device)
    target_net = DQN(n_observations, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(10000)


    steps_done = 0


    def select_action(state):
        global steps_done
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * steps_done / EPS_DECAY)
        steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)


    episode_durations = []


    def plot_durations(show_result=False):
        plt.figure(1)
        durations_t = torch.tensor(episode_durations, dtype=torch.float)
        if show_result:
            plt.title('Result')
        else:
            plt.clf()
            plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())
        # Take 100 episode averages and plot them too
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated
        if is_ipython:
            if not show_result:
                display.display(plt.gcf())
                display.clear_output(wait=True)
            else:
                display.display(plt.gcf())



    def optimize_model():
        if len(memory) < BATCH_SIZE:
            return
        transitions = memory.sample(BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
        optimizer.step()


    num_episodes = 240


    save_num = 4
    save_run = np.linspace(0, num_episodes, save_num, endpoint=False, dtype=int)
    saved_runs = {}

    for i_episode in range(num_episodes):
        # Initialize the environment and get it's state
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        if i_episode in save_run :
            #print(state)
            saved_runs[i_episode] = [state]
        for t in count():
            action = select_action(state)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
                if i_episode in save_run :saved_runs[i_episode].append(next_state)
            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            optimize_model()

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            target_net.load_state_dict(target_net_state_dict)

            if done:
                episode_durations.append(t + 1)
                plot_durations()
                break



    def animate_pole_at_timestep(X, label='itr_40', loc='ani'):
        """
        The columns are:
        Cart Position: representing the horizontal position of the cart on the track.

        Cart Velocity: representing the horizontal velocity of the cart.

        Pole Angle: representing the angle between the vertical direction and the line connecting the center of mass of the pole to the cart.

        Pole Velocity, representing the velocity of the tip of the pole.
        """
        df = pd.DataFrame(X, columns=['Cart Position', 'Cart Velocity', 'Pole Angle', 'Pole Velocity'])
        # animation of the cartpole

        y = np.zeros(len(df))
        x = df['Cart Position']
        r = length = 0.05
        x2 = pole_tip = x + np.sin(df['Pole Angle']) * r
        y2 = pole_tip = y + np.cos(df['Pole Angle']) * r



        fig, ax = plt.subplots()
        camera = Camera(fig)

        for i in range(len(df)):
            # marker for the cart is a rectangle
            plt.scatter([x[i]], [y[i]], color='blue', marker='s', s=100)
            plt.plot([x[i], x2[i]], [y[i], y2[i]], color='red')

            camera.snap()

        animation = camera.animate()
        animation.save(f'{loc}/animation_{label}.mp4')
        plt.close()

    for key, val in saved_runs.items():
        print(key)
        saved_runs[key] = torch.cat(val)
        animate_pole_at_timestep(saved_runs[key], label=f'itr_{key}', loc='../assets/Orel/images/ani')

    #np.savez(saved_runs, '../assets/Orel/saved_runs.npz')

    print('Complete')
    plot_durations(show_result=False)
    plt.savefig('../assets/Orel/images/plot_learning_progress.png')
    plt.ioff()
    plt.show()

def lunar_lander():



    #env = gym.make("LunarLander-v2", render_mode="rgb_array",
    #                continuous = False,
    #                gravity= -10.0,
    #                enable_wind = False,
    #                wind_power = 15.0,
    #                turbulence_power = 1.5,)

    import random

    # Create an environment
    env = gym.make('LunarLander-v2')

    # Define some variables
    max_steps = 1000
    total_reward = 0
    episodes = 100

    observations = []

    # Loop through the number of episodes
    for episode in range(episodes):
        # Reset the environment
        state = env.reset()
        for step in range(max_steps):
            # Render the environment
            #env.render()
            
            # Choose a random action
            action = env.action_space.sample()
            
            # Take the action
            next_state, reward, done, info = env.step(action)
            
            # Add the reward to the total reward
            total_reward += reward
            
            # Update the state
            state = next_state
            if episode == episodes-1: observations.append(state)
            # Check if the episode is done
            if done:
                print(f"Episode: {episode} Reward: {total_reward}")
                break

    # Close the environment
    env.close()
    observations = np.array(observations)



    #column names are  ['x', 'y', 'vx', 'vy', 'angle', 'angular velocity', 'left leg contact', 'right leg contact']

    df = pd.DataFrame(observations, columns=['x', 'y', 'vx', 'vy', 'angle', 'angular velocity', 'left leg contact', 'right leg contact'])


    def makeBox(x,y, angle, width, height):
        def deg2rad(deg):
            return deg * np.pi / 180

        angle = deg2rad(angle)
        # make a box
        box = np.array([[x-width/2,y-height/2],
                        [width,y-height/2],
                        [x+width/2,y+height/2],
                        [x-width/2,y+height/2],
                        [x-width/2,y-height/2]])
        #print(box)
        # rotate the box
        def rotation_matrix(angle):
            return np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        box = box @ rotation_matrix(angle)
        #print(box)
        return box



    fig = plt.figure()
    camera = Camera(fig)
    for i in range(len(df)):
        if i % 10 == 0:
        
            
            box = makeBox(df.x[i],df.y[i],df.angle[i],0.01,0.01)
            print(np.round(box.T,3))
            plt.plot(box[:,0], box[:,1], 'k-', lw=2)
            camera.snap()
    plt.show()

    animation = camera.animate()
    animation.save('animation.mp4')




if __name__ == '__main__':
    functions = [pre_start, multi_armed_bandit, week1_notes, lecture2_notes, lecture3_notes, lecture_feb_23_notes, lectureNotes_march_02, #cart_pole, #lunar_lander
                ]
    with streamlit_analytics.track():
        
        navigator(functions)
