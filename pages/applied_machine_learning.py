import streamlit as st
from utils.utils_global import *
from utils.utils_ADL import *

from scipy.ndimage import gaussian_filter
import plotly.graph_objects as go

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap.umap_ import UMAP
# from torch_scatter import scatter_mean
import graphviz

def landing_page():
    ''
    """# Applied Machine Learning"""


def lecture_2():
    ''
    "# Lecture 2: April 26, 2023"
    
    def loss_functions_():
        "## Loss Functions"
        # classification
        st.markdown(r"""### Classification""")
        cols = st.columns(2)
        
        with cols[0]:
            st.markdown(r"""
            We have a couple different loss functions show on the right
            For **classification** we have
            * Zero-One Loss
                * zero if wrong classification, one if correct
            * Hinge Loss
                $$
                    \mathcal{L} = \max(0, 1-y_n \hat{y}_n)
                $$
            * Binary Cross Entropy (logistic loss or log loss)
                $$
                    \mathcal{L} = -\frac{1}{N}\sum_{n=1}^N \left(y_n \log \hat{y}_n + (1-y_n) \log (1-\hat{y}_n)\right)
                $$

            #### Unbalanced Data

            If we are dealing with unbalanced data (e.g. 99% of the data is one class, 1% is the other), we should be careful in picking a loss function. Approprate loss functions include
            * F1 Score
            * Binary Cross Entropy
            """)
        
        x = np.linspace(-3,3,100)
        zero_one = np.where(x<0, 1, 0)
        hinge = np.where(x<1, 1-x, 0)
        exponential = np.exp(-x)*.1
        binary_cross_entropy = -np.log(1/(1+np.exp(-x)))
        fig = plt.figure(figsize=(6,4))
        plt.plot(x, zero_one, label='Zero-One')
        plt.plot(x, hinge, label='Hinge')
        plt.plot(x, exponential, label='Exponential')
        plt.plot(x, binary_cross_entropy, label='Binary Cross Entropy')
        
        plt.legend()
        plt.title('Loss Functions')
        plt.xlabel('x')
        plt.ylabel('Loss')
        plt.grid()
        with cols[1]:
            st.pyplot(fig)

        st.markdown(r"""
        ### Regression
        Use one of the following
        * Mean Squared Error
            $$
                \mathcal{L} = \frac{1}{N}\sum_{n=1}^N (y_n - \hat{y}_n)^2
            $$
        * Mean Absolute Error -> problematic because not differentiable at 0
            $$
                \mathcal{L} = \frac{1}{N}\sum_{n=1}^N |y_n - \hat{y}_n|
            $$
        * Huber Loss -> a combination of the two above, differentiable at 0
            $$
                \mathcal{L} = \frac{1}{N}\sum_{n=1}^N \begin{cases}
                    \frac{1}{2}(y_n - \hat{y}_n)^2 & \text{if } |y_n - \hat{y}_n| \leq \delta \\
                    \delta |y_n - \hat{y}_n| - \frac{1}{2}\delta^2 & \text{otherwise}
                \end{cases}
            $$
        * Log-Cosh Loss -> industry standard, double differentiable everywhere
            $$
                \mathcal{L} = \frac{1}{N}\sum_{n=1}^N \log(\cosh(y_n - \hat{y}_n))
            $$
        * Quantile Loss
            $$
                \mathcal{L} = \frac{1}{N}\sum_{n=1}^N \begin{cases}
                    \alpha |y_n - \hat{y}_n| & \text{if } y_n - \hat{y}_n \geq 0 \\
                    (1-\alpha) |y_n - \hat{y}_n| & \text{otherwise}
                \end{cases}
            $$
            """)

    def gradient_descent_():
        ''
        "## Gradient descent"
        cols = st.columns(2)
        cols[0].markdown(r"""
        
        Hyperparameter: learning rate. We want to use the biggest learning rate for which we dont fail.
        """)

        # lets draw a loss landscape
        x = np.linspace(-5,5,100)
        y = np.linspace(-5,5,100)
        X, Y = np.meshgrid(x,y)
        Z = X**2 + Y**2 + np.sin(X*8) 
        # add noise
        Z += np.random.normal(0, 3, Z.shape)
        # make smooth
        
        Z = gaussian_filter(Z, sigma=1) # to imoprt gaussian_filter: 

        
        fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y)])
        fig.update_layout(title='Loss Landscape', autosize=False,
                            width=500, height=500,
                            margin=dict(l=15, r=15, b=15, t=35))
        cols[1].plotly_chart(fig)

    def test_train_split_():
        ''
        '## Test Train Split'
        cols = st.columns(2)
        cols[0].markdown(r"""
        
        Split data into training data, validation data, and test data. Thus we can notice if we have overfitted to out training data. simply use sklearn.model_selection.train_test_split.

        We should visualize of loss curves on the training and validation data. When the validation loss starts to increase, we have overfitted to the training data, and we should backtrack to the lowest validation loss. Then these parameters are the ones we should use for the test data.
        """)
        fig = plt.figure(figsize=(6,4))
        epochs = np.arange(0, 15)
        # training loss should be exponentially decreasing
        training_loss = np.exp(-epochs) + np.random.normal(0, .01, epochs.shape)
        offset = 1
        validation_loss = np.exp(-epochs) + offset + np.random.normal(0, .01, epochs.shape)

        plt.plot(epochs, training_loss, label='Training Loss')
        plt.plot(epochs, validation_loss, label='Validation Loss')
        plt.legend()
        plt.title('Loss Curves')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid()
        cols[1].pyplot(fig)

        '### k-fold cross validation'
        """
        If we have a small dataset; we can emplot this method. 
        1. We split the data into k folds, 
        1. and then we train on k-1 folds, and validate on the remaining fold. 
        1. We repeat this k times, and then average the results. 

        This way we can use all the data for training and validation.
        """


        '### Cross Validation for time series'
        """
        If we have a time series, we should not use k-fold cross validation, because we will be using future data to predict past data. Instead we should use a sliding window.
        1. We split the data into k windows,
        1. and then we train on k-1 windows, and validate on the remaining window.
        1. We repeat this k times, and then average the results.

        """ 

    def decision_trees_():
        ''
        "## Decision Trees"
        cols = st.columns(2)
        with cols[0]:
            r"""
            Say we are dealing with a dataset with many missing values, a descision tree can handle this.
            
            ### Boosted descision trees (works great)
            * invariant under monotonic transformations (scaling, shifting)
            * robust to outliers, missing values, and irrelevant features
            * can handle mixed data types
            * works off-the-shelf (typically)

            A problem with a single tree,is overfitting! We can solve this by using a forest of trees, and then average the results. This is called a random forest.

            **Boosting** from adaboost
            $$
                y_\text{boost}(x) = \frac{1}{N} \sum_{n=1}^N \ln{\alpha_n} h_i(x)
            $$
            in which $\alpha$ is the boost rate $\alpha = \frac{1-\text{err}}{\text{err}}$, and $h_i$ is the weak learner (a single tree).

            The method focuses new trees on the examples that were misclassified by the previous trees. This is done by weighting the examples, and then training a new tree on the weighted examples. The weights are updated after each tree is trained. (This is called boosting)
            """
        with cols[1]:
            # lets try it on the XOR-dataset
            X = np.random.randint(0,2, (1000,2)).astype(float)
            y = np.logical_xor(X[:,0], X[:,1]).astype(float)            
            X += np.random.normal(0, .1, X.shape)*.5

            # test train split
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

            fig, axes = plt.subplots(2,2, figsize=(6,6))
            axes[0,0].scatter(X_train[:,0], X_train[:,1], c=y_train, marker='x', label='train', alpha=.5)
            axes[0,0].scatter(X_test[:,0], X_test[:,1], c=y_test, marker='o', label='test', alpha=.5)

            axes[0,0].set_xlabel('x1')
            axes[0,0].set_ylabel('x2')
            axes[0,0].legend(loc='center')
            axes[0,0].set_title('XOR dataset (ground truth)')

            # import ada boost
            from sklearn.ensemble import AdaBoostClassifier
            from sklearn.tree import DecisionTreeClassifier
            #import xgboost as xgb
            from xgboost import XGBClassifier
            # import ligjhtgbm
            

            for i, classifier in enumerate([AdaBoostClassifier, DecisionTreeClassifier, XGBClassifier], start=1):
                ax = axes.flatten()
                y_pred = classifier().fit(X_train, y_train).predict(X_test)
                ax[i].scatter(X_train[:,0], X_train[:,1], c=y_train, marker='x', label='train', alpha=.5)
                ax[i].scatter(X_test[:,0], X_test[:,1], c=y_pred, marker='o', label='test', alpha=.5)
                ax[i].set_xlabel('x1')
                ax[i].set_ylabel('x2')
                classifier_name = classifier.__name__
                ax[i].set_title(f'{classifier_name}')
                ax[i].legend(loc='center')
            plt.tight_layout()
            
            st.pyplot(fig)
            plt.close()


    loss_functions_()
    gradient_descent_()
    test_train_split_()
    decision_trees_()
        
def lecture_3():
    ''
    '## Lecture 3: April 26, 2023'
    '### Neural networks'
    cols = st.columns(2)
    with cols[0]:
        "To get an intuition for neural networks, watch Andrej Karparthys video on building MicroGrad: [link](https://www.youtube.com/watch?v=VMj-3S1tku0&t=3s&pp=ygUJbWljcm9ncmFk)"

        """
        #### Activation functions
        * sigmoid, tanh, ReLU, Leaky ReLU, ELU, Maxout
        """

        # view these functions
        x = np.linspace(-3,3,100)
        funcs = {
            'sigmoid': lambda x: 1/(1+np.exp(-x)),
            'tanh': np.tanh,
            'ReLU': lambda x: np.maximum(0, x),
            'Leaky ReLU': lambda x: np.maximum(.1*x, x),
            'ELU': lambda x: np.where(x>0, x, np.exp(x)-1),
            #'Maxout': lambda x: np.maximum(x, -x)
        }
        fig = plt.figure(figsize=(6,3))
        for i, (name, func) in enumerate(funcs.items(), start=1):
            plt.plot(x, func(x), label=name)
        plt.legend()
        plt.grid()
        st.pyplot(fig)
        plt.close()

        """
        #### Regularization techniques
        * $x' = (x-\mu) / \sigma$
        * ...
        """

        """
        #### Architectures:
        * Feedforward NN (MLP)
        * Recurrent NN (RNN) -> allows for memory by letting nodes in the same layer be connected.
        * Long short-term memory (LSTM) -> RNN with memory gates
        * Adversarial
        * Graph NN
        * etc.
        """
    with cols[1]:
        st.image('https://www.asimovinstitute.org/wp-content/uploads/2019/04/NeuralNetworkZoo20042019.png')

    '''
    #### Dropout
    drop nodes in the network. We can either select these randomly and evaluate performance of the pruned network. Or we can use a dropout rate, which is the probability of dropping a node. This is a hyperparameter that needs to be tuned.

    Typically, we let the network train for some number of epochs until the loss is decreasing a monotonic fashion. Then we start to apply dropout. This is called annealing.

    But note; there are plenty of ways to do this...🥵
    '''


    """
    #### Hyperparameter tuning
    To get a NN to work well, it need tuning. For more on this topic see Advanced deep learning - MLOps. 
    """

def dimensionality_reduction():
    ''
    '## Dimensionality reduction'
    
    # PCA
    cols = st.columns(2)
    with cols[0]:
        """
        ### PCA
        Principal component analysis. We look for linear combinations of features in our data which describe most variance in our data.

        So intialliy, we find the axis of most variance (the principal component). Thereafter, we look for the dimension containing most variance in the space orthogonal to the first principal component. This is the second principal component. And so on.

        In Practice, this is done not iteratively, but instead by finding the eigenvalues and eigenvectors of the covariance matrix of the data. The eigenvectors are the principal components, and the eigenvalues are the variance explained by each principal component.

        **Note**: you should apply a standard scalar first 🥸
        """
    with cols[1]:
        X = np.random.normal(0, 1, (200,2))
        X[:,1] = X[:,0]*1.2 + X[:,1]
        X /= 5
        pca = PCA(n_components=2)
        PC = pca.fit(X)
        # plot PCs as lines
        fig, ax = plt.subplots(figsize=(6,4))
        colors = ['g', 'b']
        fig.suptitle('PCA om random, correlated data', fontsize=16)
        plt.scatter(X[:,0], X[:,1], c='k', marker='x', label='data', alpha=.5)
        for i, (x, y) in enumerate(zip(PC.components_[0], PC.components_[1])):
            plt.plot([0, y], 
                     [0, x], label=f'PC{i+1}', lw=3, c=colors[i])
        ax.set(xlim=(-1.,1.), ylim=(-1.5,1.5))
        plt.legend()
        plt.close()
        st.pyplot(fig)
    '---'
    # t-SNE
    cols = st.columns(2)

    with cols[0]:
        """
        ### t-SNE
        t-distributed stochastic neighbor embedding. We look for non-linear combinations of features in our data which describe most variance in our data. 

        For the example here, let's work with the Iris dataset. We have 4 dimensions, so it's kinda hard to visualize... Therefore we embed it into 2 dimensions.

        t-SNE is not a projection method, instead it is a manifold method. What is perserved under the transformation; is the likeness between samples. So, what happens is; we throw our high-dimensional data into a smaller space, and try to perserve the inter-sample distances. 

        To get both high and low density regions right (weighted with equal importance), we need to consider the density around a point in question when determining the similarity too other points.

        (remember to normalize the similarity matrix)

        **Perplexity**: regards to the expected density -> how many neighbors to consider when determining the similarity between points. This lets us decide; are we interested in global structure or local structure? (low perplexity -> global structure, high perplexity -> local structure). This is a trade-off.
        """
        with st.expander('Iris dataset', expanded=False):
            iris = sns.load_dataset('iris')
            iris
    with cols[1]:
        
        X = iris.drop('species', axis=1)
        y = iris['species']
        perplexity = 30 #
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        X_embedded = tsne.fit_transform(X)

        fig, ax = plt.subplots(figsize=(6,4))
        fig.suptitle('t-SNE embedding of Iris dataset', fontsize=16)
        sns.scatterplot(x=X_embedded[:,0], y=X_embedded[:,1], hue=y, ax=ax)
        plt.close()
        st.pyplot(fig)
    '---'

    # UMAP
    cols = st.columns(2)
    with cols[0]:
        """
        ### UMAP
        Uniform Manifold Approximation and Projection (UMAP) is yet another embedding method.

        [paper](https://arxiv.org/pdf/1802.03426.pdf)
        """

    '---'
    'lets try embedding the MNIST dataset with these three methods'

    # load data (10k)
    Mnist = datasets.MNIST(root='assets/advanced_deep_learning/data/', train=True, download=True)
    X = Mnist.data.numpy()
    y = Mnist.targets.numpy()
    X = X.reshape(X.shape[0], -1)

    # take a subset
    idx = np.random.choice(np.arange(X.shape[0]), 500, replace=False)
    X = X[idx]
    y = y[idx]

    # PCA
    pca = PCA(n_components=2)

    X_embedded_PCA = pca.fit_transform(X)
    # t-SNE
    tsne = TSNE(n_components=2, 
                # to she barnes hut: 
                method='barnes_hut',

                perplexity=st.slider('t-SNE Perplexity', 5, 50, 30),
                random_state=42
                )
    X_embedded_TSNE = tsne.fit_transform(X)

    umap = UMAP(n_components=2, random_state=42, n_neighbors=5
                ) # to import from umap.umap_ import UMAP
    X_embedded_UMAP = umap.fit_transform(X)

    # plot
    fig, ax = plt.subplots(1,3, figsize=(12,4))
    fig.suptitle('MNIST dataset embedded with PCA, t-SNE and UMAP', fontsize=16)
    sns.scatterplot(x=X_embedded_PCA[:,0], y=X_embedded_PCA[:,1], hue=y, ax=ax[0]) ; ax[0].set_title('PCA')
    sns.scatterplot(x=X_embedded_TSNE[:,0], y=X_embedded_TSNE[:,1], hue=y, ax=ax[1]); ax[1].set_title('t-SNE')
    sns.scatterplot(x=X_embedded_UMAP[:,0], y=X_embedded_UMAP[:,1], hue=y, ax=ax[2]); ax[2].set_title('UMAP')
    plt.close()
    st.pyplot(fig)

    '---'

    """
    Also try trimap or pacmap
    """

    '---'
        
    # Autoencoders
    cols = st.columns(2)
    with cols[0]:
        """
        ### Autoencoders
        NN based approach, which consists of an encoder and decoder pair. The output tries to match the input. The two components are connect by a latent layer. If we set the latent layer width to eg 3, and we are able to get outputs which match the inputs well, we have determined that we can losslessly compress our data into 3 dimensions, i.e., the intrinsic dimension is at most 3. 

        #### Variational autoencoders
        Use $\mu$ and $\sigma$ in the latent layer, and are thus able to be generative.
        """
    with cols[1]:
        st.image('https://www.compthree.com/images/blog/ae/ae.png', caption='autoencoder')

def recurrent_neural_networks():
    ''

    """
    ## Recurrent neural networks,

    the outout of a cell of the network is passed as input to the next cell along with the next obersevation. This lets us predict the next value. Alternatively me may gauge global properties of the timeseries.
    
    """

def graph_neural_networks():
    ''
    '# Graph neural networks'
    cols = st.columns(2)
    with cols[0]:
        r'''
        
        Data structured as a matrix, is interpreted by a convolutional networks, as being equidistant from its nearest neighbours. A graph considers data in which points may connected to different numbers on neighbours and with variable distance.

        So we can be free from making assumptions about the data structure, and instead let the network learn the structure. This lets us work with geometrical data.

        Many of the networks previously discussed work for graphs: LSTMs, CNNs, autoencoders, etc. But we can also use graph specific networks.

        Instead of convolving over pixels, we convolve over nodes or edges. This outputs another graph.
        
        An example of a convolution is **edgeconv**
        $$
            \tilde{x}_j = \sum_{i=1}^n f(x_j, x_j-x_i)
        $$
        '''
    with cols[1]:
        st.image('https://tkipf.github.io/graph-convolutional-networks/images/gcn_web.png')
    '---'
    # example
   
    
    '''
    ### Example
    lets make a matrix, and display it as a graph

    then we make a model which takes the matrix as input, and outputs a new matrix
    '''
    cols = st.columns((4,1,4))
    with cols[0]:
    
        # make matrix
        n = 10
        A = np.random.poisson(0.4, size=(n,n))
        A = np.round(A)
        A = A + A.T
        A = np.where(A>0, 1, 0)
        A = A - np.eye(n)
        # display
        G = graphviz.Digraph()
        for i in range(n):
            G.node(str(i))
        for i in range(n):
            for j in range(n):
                if A[i,j] == 1:
                    G.edge(str(i), str(j))

        st.graphviz_chart(G, use_container_width=True)

        # make model
        class EdgeConv(nn.Module):
            def __init__(self, in_channels, out_channels):
                super(EdgeConv, self).__init__()
                self.mlp = nn.Sequential(
                    nn.Linear(in_channels*2, out_channels),
                    nn.ReLU(),
                    nn.Linear(out_channels, out_channels)
                )
            def forward(self, x, edge_index):
                row, col = edge_index
                out = torch.cat([x[row], x[col]], dim=1)
                #st.write(out)
                out = self.mlp(out)
                out = self.scatter_mean(out, row, dim=0, dim_size=x.size(0)) # to import from torch_scatter import scatter_mean
                return out
            
            def scatter_mean(self, src, index, dim=0, out=None, dim_size=None, fill_value=0):
                # he we write it out in torch
                if out is not None:
                    out = out.fill_(fill_value)
                else:
                    size = list(src.size())
                    size[dim] = dim_size
                    out = src.new_full(size, fill_value)
                return out.scatter_add_(dim, index, src)

            
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv1 = EdgeConv(10, 32)
                self.conv2 = EdgeConv(32, 32)
                self.conv3 = EdgeConv(32, 10)
            def forward(self, x, edge_index):
                #st.write(x.shape, edge_index.shape)
                x = self.conv1(x, edge_index)
                x = self.conv2(x, edge_index)
                x = self.conv3(x, edge_index)
                return x
            
        # train model
        model = Net()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()
        for epoch in range(5):
            optimizer.zero_grad()
            output = model(torch.Tensor(A), torch.Tensor(np.array(np.where(A==1))).long())
            loss = criterion(output, torch.Tensor(A))
            loss.backward()
            optimizer.step()

        output = output.detach().numpy()
        #output
        # display output
        G = graphviz.Digraph()
        for i in range(n):
            G.node(str(i))
        for i in range(n):
            for j in range(n):
                if output[i,j] > 0.5:
                    G.edge(str(i), str(j))

    with cols[1]:
        for i in range(6):
            '.'
        r'$\Rightarrow$'
    with cols[2]:
        st.graphviz_chart(G, use_container_width=True)
            


    '---'


def generative_adversarial_networks():
    ''
    "# Generative adversarial networks"

    cols = st.columns(2)
    with cols[0]:
        """
        *One way to generate, is variational autoencoders (see advanced machine learning)*. By making the latent layer stochastic we can sample from it in a way which yield probable sampels. This makes the latent latent embedding continuous, and we can interpolate between samples.


        GANs is a different approach; we have a gnerator and and discriminator. The generator makes a smaple, and the discriminator evaluates whether it look like a real sample. The generator is trained to fool the discriminator.
        
        """
    with cols[1]:
        st.image('https://d18rbf1v22mj88.cloudfront.net/wp-content/uploads/sites/3/2018/03/29200233/GAN_en.png')


    '---'

    # lets train a gan on the mnist dataset
    cols = st.columns(2)
    with cols[0]:
        """
        ### Example

        """

    with cols[1]:
        # load data
        batch_size = 32
        mnist = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
        dataloader = torch.utils.data.DataLoader(mnist, batch_size=batch_size, shuffle=True)
        # make model
        class Generator(nn.Module):
            def __init__(self):
                super(Generator, self).__init__()
                self.model = nn.Sequential(
                    nn.Linear(100, 128),
                    nn.ReLU(),
                    nn.Linear(128, 256),
                    nn.ReLU(),
                    nn.Linear(256, 512),
                    nn.ReLU(),
                    nn.Linear(512, 784), # 28*28
                    nn.Sigmoid()
                )
            def forward(self, x):
                x = self.model(x)
                return x
        class Discriminator(nn.Module):
            def __init__(self):
                super(Discriminator, self).__init__()
                self.model = nn.Sequential(
                    nn.Linear(784, 512),
                    nn.ReLU(),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Linear(256, 1),
                    nn.Sigmoid()
                )
            def forward(self, x):
                x = self.model(x)  # output is probability of being real
                return x
            
        generator = Generator()
        discriminator = Discriminator()
        # train model
        
        #use Minimax loss
        def loss_Minimax(D_x, D_G_z):
            return -torch.mean(torch.log(D_x) + torch.log(1 - D_G_z))
        
        criterion = nn.BCELoss()
        optimizer_g = torch.optim.Adam(generator.parameters(), lr=0.01)

        optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=0.01)

        for epoch in range(5):
            for i, (images, _) in enumerate(dataloader):
                for i in range(10):
                    # train discriminator
                    optimizer_d.zero_grad()
                    images = images.view(images.size(0), -1)
                    real_labels = torch.ones(images.size(0), 1)
                    fake_labels = torch.zeros(images.size(0), 1)
                    # train on real
                    D_x = discriminator(images)
                    loss_real = criterion(D_x, real_labels)
                    loss_real.backward()
                    #print('loss_real', loss_real)

                    # train on fake
                    z = torch.randn(images.size(0), 100)
                    fake_images = generator(z)
                    D_G_z = discriminator(fake_images)
                    loss_fake = criterion(D_G_z, fake_labels)
                    loss_fake.backward()


                    optimizer_d.step()


                # train generator
                optimizer_g.zero_grad()
                z = torch.randn(images.size(0), 100)
                fake_images = generator(z)
                D_G_z = discriminator(fake_images)
                loss_g = criterion(D_G_z, real_labels)
                loss_g.backward()
                optimizer_g.step()

            # print statistics
            print('Epoch: %d, loss_d: %.3f, loss_g: %.3f' % (epoch + 1, loss_real.item() + loss_fake.item(), loss_g.item()))





        # display output
        z = torch.randn(1, 100)
        fake_images = generator(z)
        fake_images = fake_images.view(28, 28).detach().numpy()
        fig, ax = plt.subplots()
        plt.imshow(fake_images, cmap='gray')
        ax.axis('off')
        st.pyplot(fig)






if __name__ == '__main__':
    functions = [lecture_2,
                 lecture_3,
                 dimensionality_reduction,
                 recurrent_neural_networks,
                 graph_neural_networks, 
                    generative_adversarial_networks]
    # with streamlit_analytics.track():
    navigator(functions)