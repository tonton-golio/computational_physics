
from utils.utils_ADL import *
#import PCA
from sklearn.decomposition import PCA
import matplotlib as mpl

import graphviz
from scipy.signal import convolve2d
filepath_assets = 'assets/advanced_deep_learning/'


# set rc params
#mpl.rcParams['figure.dpi'] = 100



def landing_page():
    '''Landing page for advanced deep learning section. For now I'll just fill this with notes for the first lecture.'''

    r"""
    # Advanced Deep Learning
    #### Lecture 1: notes (April 24, 2023)

    For an intro to backpropagation see: [MicroGrad github page](https://github.com/karpathy/micrograd) and [walkthrough: Youtube video](https://www.youtube.com/watch?v=VMj-3S1tku0) . Basically we have to define a value class, which automatically propagates the gradient:
    """
    with st.expander('Value class', expanded=False):
        """
        ```python

    class Value:
        '''stores a single scalar value and its gradient'''

        def __init__(self, data, _children=(), _op=''):
            self.data = data
            self.grad = 0
            # internal variables used for autograd graph construction
            self._backward = lambda: None
            self._prev = set(_children)
            self._op = _op # the op that produced this node, for graphviz / debugging / etc

        def __add__(self, other):
            other = other if isinstance(other, Value) else Value(other)
            out = Value(self.data + other.data, (self, other), '+')

            def _backward():
                self.grad += out.grad
                other.grad += out.grad
            out._backward = _backward

            return out

        def __mul__(self, other):
            other = other if isinstance(other, Value) else Value(other)
            out = Value(self.data * other.data, (self, other), '*')

            def _backward():
                self.grad += other.data * out.grad
                other.grad += self.data * out.grad
            out._backward = _backward

            return out

        def backward(self):

            # topological order all of the children in the graph
            topo = []
            visited = set()
            def build_topo(v):
                if v not in visited:
                    visited.add(v)
                    for child in v._prev:
                        build_topo(child)
                    topo.append(v)
            build_topo(self)

            # go one variable at a time and apply the chain rule to get its gradient
            self.grad = 1
            for v in reversed(topo):
                v._backward()
        ```
        """


    r"""
    After creating this value class, we have to define a class for setting up a network (acutally we need a couple classes).
    """
    with st.expander('Network classes', expanded=False):
        """
        ```python
    class Module:

        def zero_grad(self):
            for p in self.parameters():
                p.grad = 0

        def parameters(self):
            return []

    class Neuron(Module):

        def __init__(self, nin, nonlin=True):
            self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
            self.b = Value(0)
            self.nonlin = nonlin

        def __call__(self, x):
            act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
            return act.relu() if self.nonlin else act

        def parameters(self):
            return self.w + [self.b]

        def __repr__(self):
            return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"

    class Layer(Module):

        def __init__(self, nin, nout, **kwargs):
            self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

        def __call__(self, x):
            out = [n(x) for n in self.neurons]
            return out[0] if len(out) == 1 else out

    class MLP(Module):

        def __init__(self, nin, nouts):
            sz = [nin] + nouts
            self.layers = [Layer(sz[i], sz[i+1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))]

        def __call__(self, x):
            for layer in self.layers:
                x = layer(x)
            return x
    ```
        """

    """
    And there we go; now we have propagate gradients through the network. Note; atm it ouly works with addition and multiplication.
    """

    # lets draw up a simple neural network
    # 3 input neurons
    # 2 hidden layers with 4 neurons each
    # 1 output neuron
    g = graphviz.Digraph()
    for i in range(3):
        g.node('x'+str(i))
    for i in range(4):
        g.node('h1'+str(i))

    for i in range(4):
        g.node('h2'+str(i))

    g.node('y')

    for i in range(3):
        for j in range(4):
            g.edge('x'+str(i), 'h1'+str(j))

    for i in range(4):
        for j in range(4):
            g.edge('h1'+str(i), 'h2'+str(j))

    for i in range(4):
        g.edge('h2'+str(i), 'y')

    # display, laying down
    st.graphviz_chart(g, use_container_width=True)
    
def artificial_neural_networks():

    # load text
    text_dict = getText_prep_new(filepath_assets+'ann.md')

    # Title
    st.markdown('# Artificial Neural Networks', unsafe_allow_html=True)

    # load MNIST dataset
    transform = transforms.ToTensor()
    train_data = datasets.MNIST(root=filepath_assets+'data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root=filepath_assets+'data', train=False, download=True, transform=transform)

    # print intro
    st.markdown(text_dict['intro'], unsafe_allow_html=True)

    # DataLoader
    torch.manual_seed(69)

    train_loader = DataLoader(train_data, batch_size=100, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=500, shuffle=False)

    # Visualize data
    for images,labels in test_loader:
        break     # grab first batch of images
    visualize_from_dataloader(images[:8], labels[:8])
        
    # Instantiate model
    model = MultilayerPerceptron()

    # criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # print structure of model and some text
    cols = st.columns(2)
    cols[0].markdown(text_dict['multilayer perceptron model'], unsafe_allow_html=True)
    view_model(model, st=cols[1]) # view graph of model
    
    #print number of parameters 
    cols[1].markdown('**Number of parameters:** '+str(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    '---'
    
    # train model
    st.markdown(text_dict['training'], unsafe_allow_html=True)
    # cache model
    #@st.cache(allow_output_mutation=True)
    if st.button('Train model'):
        train_losses, test_losses = train_MNIST(model, train_loader,test_loader, optimizer, criterion, epochs=2, streamlit_view=True)
        
        '---'
        # evaluate model
        st.markdown(text_dict['evaluation'], unsafe_allow_html=True)

        # get one batch of test images
        for images,labels in test_loader:
            break     # grab first batch of images
        # how many to show =4
        n_images = 4
        y_pred = model(images[:n_images].view(n_images, 784))
        
        # Tally the number of correct predictions
        predicted = torch.max(y_pred.data, 1)[1]
        visualize_from_dataloader(images[:n_images], predicted)

def convolutional_neural_networks():
    # load text
    text_dict = getText_prep_new(filepath_assets+'cnn.md')
    
    # title
    st.markdown('# Convolutional Neural Networks', unsafe_allow_html=True)

    # intro text
    st.markdown(text_dict['intro'], unsafe_allow_html=True)
    # to center the image
    st.markdown('''<div style="display: flex; justify-content: center; align-items: center;">
    <img src="https://miro.medium.com/v2/resize:fit:1100/format:webp/1*L1SVH2rBxGvJx3L4aB59Cg.png" width=500>
    </div>
    ''', unsafe_allow_html=True)

    # kernels
    with st.expander('kernels', expanded=True):
        cols = st.columns((1,2))
        with cols[0]:
            st.markdown(text_dict['kernels'])
        with cols[1]:
            # lets show some of these: averaging, gaussian blur, and Dilated (Atrous) Convolution
            n = 20
            # avg
            x_averaging = np.zeros((n,n))
            x_averaging [1:4,1:4] = 1/9
            # gauss blur
            center, sigma = (2,2), 0.015
            x_gaussian_blur = np.ones((n,n))
            gauss_blur = lambda r : 1/(2*np.pi*sigma)**.5 * np.exp(-r**2/(2*sigma**2))
            for i in range(n):
                for j in range(n):
                    d = ((i-center[0])**2 + (j-center[1])**2)/1000
                    
                    x_gaussian_blur[i,j] *= gauss_blur(d)

            # Dilated (Atrous) Convolution
            x_dilated = np.zeros((n,n))
            for i in range(1, 7, 2):
                for j in range(1,7,2):
                    x_dilated[i,j] = 1

            x_dilated/=np.sum(x_dilated)

            # Dilated (Atrous) Convolution bigger
            x_dilated_big = np.zeros((n,n))
            for i in range(1, 13, 3):
                for j in range(1,13,3):
                    x_dilated_big[i,j] = 1

            x_dilated_big/=np.sum(x_dilated_big)

            fig, ax = plt.subplots(1,4, figsize=(15,3))
            ax[0].imshow(x_averaging) ; ax[0].set_title('averaging kernel')
            ax[1].imshow(x_gaussian_blur) ; ax[1].set_title('gaussian blur kernel')
            ax[2].imshow(x_dilated) ; ax[2].set_title('dilated kernel')
            ax[3].imshow(x_dilated_big) ; ax[3].set_title('dilated kernel bigger')
            for axi in ax:
                axi.set(xticks=[], yticks=[])

            st.pyplot(fig)

            plt.close()

        # lets convolve with this kernel
        img = plt.imread('assets/advanced_deep_learning/shiba.png')
        fig, ax = plt.subplots(1,6, figsize=(12,4))
        ax[0].imshow(img) ; ax[0].set_title('original image', fontsize=10)
        # grey scale
        img = np.mean(img, axis=2)

        # down sample
        def downscale(img, factor=5):
            # make sure divisible by factor
            img = img[:img.shape[0]//factor*factor, :img.shape[1]//factor*factor]
            # max pooling
            maxes = np.max(img.reshape(img.shape[0]//factor, factor, img.shape[1]//factor, factor), axis=(1,3))

            return maxes
        
        img = downscale(img)

        
        ax[1].set_title('grey and downsample', fontsize=10)
        ax[1].imshow(img)
        
        ax[2].set_title('averaging conv.', fontsize=10)
        ax[2].imshow(convolve2d(img, x_averaging[:5,:5]))
        ax[3].set_title('gaussian blur conv.', fontsize=10)
        ax[3].imshow(convolve2d(img, x_gaussian_blur[:5,:5]))
        ax[4].set_title('dilated conv.', fontsize=10)
        ax[4].imshow(convolve2d(img, x_dilated[1:7,1:7]))
        ax[5].set_title('dilated conv. big', fontsize=10)
        ax[5].imshow(convolve2d(img, x_dilated_big[:13,:13]))
        for axi in ax:
            axi.set(xticks=[], yticks=[])

        st.pyplot(fig)

    # conv2d
    with st.expander('conv2d', expanded=True):
        '''
        * We define a kernel to use to step through our matrix.
        * This makes the shape smaller, so we can pad it to keep the shape the same. (we typically pad with zeros)
        * the step size is called the stride.

        * 
        '''

    # load data
    transform = transforms.ToTensor()
    train_data = datasets.MNIST(root=filepath_assets+'data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root=filepath_assets+'data', train=False, download=True, transform=transform)

    # DataLoader
    torch.manual_seed(69)

    train_loader = DataLoader(train_data, batch_size=100, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=500, shuffle=False)

       
        
    # instantiate model
    model = ConvolutionalNetwork()
    
    # view model
    cols = st.columns((1,1))
    cols[0].markdown(text_dict['convolutional neural network model'], unsafe_allow_html=True)
    view_model(model, st=cols[1], input_sz=(1, 28,28)) # view graph of model

    #print number of parameters
    cols[1].markdown('**Number of parameters:** '+str(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    

    # define loss function
    criterion = nn.CrossEntropyLoss()

    # define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    '---'
    '### Training'
    if st.button('Train model'):
    # train model
        epochs = 2
        train_losses = []
        test_losses = []
        cols = st.columns(2)
        # center text in col
        cols[0].markdown("""<div style="text-align: center">**Loss**</div>""", unsafe_allow_html=True)
        cols[1].markdown("""<p style="text-align: center">**Accuracy**</p>""", unsafe_allow_html=True)
        loss_chart = cols[0].line_chart()
        acc_chart = cols[1].line_chart()
        for i in range(epochs):
            trn_corr = 0
            tst_corr = 0
            
            # Run the training batches
            for b, (X_train, y_train) in enumerate(train_loader):
                b+=1
                
                # Apply the model
                y_pred = model(X_train)
                loss = criterion(y_pred, y_train)

                # Tally the number of correct predictions
                predicted = torch.max(y_pred.data, 1)[1]
                batch_corr = (predicted == y_train).sum()
                trn_corr += batch_corr
                
                # Update parameters
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Print interim results
                if b%200 == 0:
                
                    print(f'epoch: {i:2}  batch: {b:4} [{100*b:6}/60000]  loss: {loss.item():10.8f}  \
                    accuracy: {trn_corr.item()*100/(100*b):7.3f}%')
                    loss_chart.add_rows([[loss.item()]])
                    acc_chart.add_rows([[trn_corr.item()*100/(100*b)]])
                
            train_losses.append(loss.item())

            # Run the testing batches
            with torch.no_grad():
                for b, (X_test, y_test) in enumerate(test_loader):

                    # Apply the model
                    y_val = model(X_test)

                    # Tally the number of correct predictions
                    predicted = torch.max(y_val.data, 1)[1] 
                    tst_corr += (predicted == y_test).sum()

            loss = criterion(y_val, y_test)
            test_losses.append(loss)
            print(f'TESTING:  loss: {loss.item():10.8f}  accuracy: {tst_corr.item()*100/10000:7.3f}%')

        '---'
        # evaluate model
        st.markdown(text_dict['evaluation'], unsafe_allow_html=True)

        # get one batch of test images
        for images,labels in test_loader:
            break     # grab first batch of images
        # how many to show =4
        n_images = 4
        y_pred = model(images[:n_images])

        # Tally the number of correct predictions
        predicted = torch.max(y_pred.data, 1)[1]
        visualize_from_dataloader(images[:n_images], predicted)

def U_net():
    # load text
    text_dict = getText_prep_new(filepath_assets+'unet.md')
    
    # title
    st.markdown('# U-Net', unsafe_allow_html=True)

    # intro text
    st.markdown(text_dict['intro'], unsafe_allow_html=True)
    # to center the image
    st.markdown('''
    ''', unsafe_allow_html=True)

    # load data
    transform = transforms.ToTensor()
    train_data = datasets.MNIST(root=filepath_assets+'data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root=filepath_assets+'data', train=False, download=True, transform=transform)

    # DataLoader
    torch.manual_seed(69)

    train_loader = DataLoader(train_data, batch_size=100, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=500, shuffle=False)

    # define model
    class U_Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 16, 3, 1, 1)
            self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)
            self.conv3 = nn.Conv2d(32, 64, 3, 1, 1)
            self.conv4 = nn.Conv2d(64, 128, 3, 1, 1)
            self.conv5 = nn.Conv2d(128, 64, 3, 1, 1)
            self.conv6 = nn.Conv2d(64, 32, 3, 1, 1)
            self.conv7 = nn.Conv2d(32, 16, 3, 1, 1)
            self.conv8 = nn.Conv2d(16, 10, 3, 1, 1)
            self.pool = nn.MaxPool2d(2, 2)
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.drop = nn.Dropout2d(0.2)
            self.relu = nn.ReLU()
            self.softmax = nn.Softmax(dim=1)
        
        def forward(self, x):
            # encoder
            x = self.relu(self.conv1(x))
            x = self.relu(self.conv2(x))
            x = self.pool(x)
            x = self.relu(self.conv3(x))
            x = self.relu(self.conv4(x))
            x = self.pool(x)
            # decoder
            x = self.relu(self.conv5(x))
            x = self.relu(self.conv6(x))
            x = self.up(x)
            x = self.relu(self.conv7(x))
            x = self.relu(self.conv8(x))
            x = self.up(x)
            x = self.softmax(x)

            return x
        

    # instantiate model
    model = U_Net()
    
    # view model
    cols = st.columns((1,1))
    cols[0].markdown(text_dict['U-Net model'], unsafe_allow_html=True)
    view_model(model, st=cols[1], input_sz=(1,1, 28,28)) # view graph of model

    #print number of parameters
    cols[1].markdown('**Number of parameters:** '+str(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    

    # define loss function
    criterion = nn.BCEWithLogitsLoss()#nn.CrossEntropyLoss()

    # define optimizer
    optimizer = torch.optim.RMSprop(model.parameters(),
                              lr=0.01, weight_decay=1e-8, momentum=0.999, foreach=True)
    '---'
    for images, labels in train_loader:
        break
    # view model output
    fig, ax = plt.subplots(4, 10, figsize=(10, 5))
    output = model(images[:4])
    output.shape
    for i in range(4):
        for j in range(10):
            ax[i][j].axis('off')
            ax[i][j].imshow(output[i][j].detach().numpy())
    st.pyplot(fig)



    '### Training'
    if st.button('Train model'):
    # train model
        epochs = 2
        train_losses = []
        test_losses = []
        cols = st.columns(2)
        # center text in col
        cols[0].markdown("""<div style="text-align: center">**Loss**</div>""", unsafe_allow_html=True)
        cols[1].markdown("""<p style="text-align: center">**Accuracy**</p >""", unsafe_allow_html=True)
        loss_chart = cols[0].line_chart()
        acc_chart = cols[1].line_chart()
        for i in range(epochs):
            train_loss = 0.0
            test_loss = 0.0
            accuracy = 0
            model.train()
            for images, labels in train_loader:
                optimizer.zero_grad()
                output = model(images)
                loss = criterion(output, images)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()*images.size(0)
                loss_chart.add_rows([[loss.item()]])
            model.eval()

def autoencoders():
    # load text
    text_dict = getText_prep_new(filepath_assets+'vae.md')

    # title
    st.markdown('# Autoencoders', unsafe_allow_html=True)

    # intro text
    st.markdown(text_dict['intro'], unsafe_allow_html=True)

    # load data -> fashion MNIST
    transform = transforms.ToTensor()
    train_data = datasets.MNIST(root=filepath_assets+'data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root=filepath_assets+'data', train=False, download=True, transform=transform)

     # dont use all data
    train_data, _ = torch.utils.data.random_split(train_data, [10000, 50000])
    test_data, _ = torch.utils.data.random_split(test_data, [10000, 0])

    # DataLoader
    torch.manual_seed(69)

    train_loader = DataLoader(train_data, batch_size=400, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=400, shuffle=False)

   
    # view data
    fig, ax = plt.subplots(1, 10, figsize=(10, 5))
    for i in range(10):
        ax[i].axis('off')
        ax[i].imshow(train_data[i][0].squeeze().numpy())
    st.pyplot(fig)
    plt.close()
    # define model
    class VAE(nn.Module):
        def __init__(self, latent_dim=2):
            super().__init__()
            self.fc1 = nn.Linear(784, 400)
            self.fc2 = nn.Linear(400, latent_dim)
            self.fc3 = nn.Linear(latent_dim, 400)
            self.fc4 = nn.Linear(400, 784)
            self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()

            self.fc__ = nn.Linear(400, 400)
        
        def encode(self, x):
            h1 = self.relu(self.fc1(x))
            h1 = self.relu(self.fc__(h1))
            return self.fc2(h1)
        
        def reparameterize(self, mu):
            std = torch.exp(0.5*mu)
            eps = torch.randn_like(std)
            return mu + eps*std
        
        def decode(self, z):
            h3 = self.relu(self.fc3(z))
            h3 = self.relu(self.fc__(h3))
            return self.sigmoid(self.fc4(h3))
        
        def forward(self, x):
            mu = self.encode(x.view(-1, 784))
            z = self.reparameterize(mu)
            return self.decode(z), mu
    
    class VAE_with_conv2d(nn.Module):
        def __init__(self, latent_dim=2):
            self.latent_dim = latent_dim
            super().__init__()
            self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.conv2_ = nn.Conv2d(32, 16, 3, padding=1)
            self.conv1_ = nn.Conv2d(16, 3, 3, padding=1)
            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(784*2*2, 256)
            self.fc2 = nn.Linear(256, 64)
            self.fc3 = nn.Linear(64, 16)
            self.fc4 = nn.Linear(16, latent_dim*2)
            self.fc4_ = nn.Linear(latent_dim, 16)
            self.fc3_ = nn.Linear(16, 64)
            self.fc2_ = nn.Linear(64, 256)
            self.fc1_ = nn.Linear(256, 784)
            self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()


        def encode(self, x):
            h1 = self.relu(self.conv1(x))
            h1 = self.relu(self.conv2(h1))
            h1 = self.relu(self.conv2_(h1))
            h1 = self.conv1_(h1)
            h1 = self.flatten(h1)
            h1 = torch.cat((h1, x.view(-1, 784)), dim=1)
            h1 = self.relu(self.fc1(h1))
            h1 = self.relu(self.fc2(h1))
            h1 = self.relu(self.fc3(h1))
            return self.sigmoid(self.fc4(h1))
        
        
        def reparameterize(self, mu_std):
            #st.write(mu_std.shape)
            std = mu_std[:, self.latent_dim:]
            mu = mu_std[:, :self.latent_dim]
            eps = torch.randn_like(std)
            out = mu + eps*std
            #st.write('out shape', out.shape)
            return self.sigmoid(out)
        
        def decode(self, z):
            h3 = self.relu(self.fc4_(z))
            h3 = self.relu(self.fc3_(h3))
            h3 = self.relu(self.fc2_(h3))
            h3 = self.fc1_(h3)
            h3 = h3.view(-1, 1, 28, 28)
            return self.sigmoid(h3)
        
        def forward(self, x):
            mu_std = self.encode(x)
            z = self.reparameterize(mu_std)
            return self.decode(z), mu_std
        
        

    # instantiate model
    st.radio('Choose model', ['Linear VAE', 'Convolutional VAE'], key='model_type')
    if st.session_state.model_type == 'Linear VAE':
        model = VAE()
        model_filename = 'vae.pt'
    else:
        model = VAE_with_conv2d()
        model_filename = 'vae_conv.pt'

    # view model

    cols = st.columns((1,1))
    if st.session_state.model_type == 'Linear VAE':
        cols[0].markdown(text_dict['VAE model'], unsafe_allow_html=True)
    else:
        cols[0].markdown(text_dict['VAE conv model'], unsafe_allow_html=True)
    view_model(model, st=cols[1], input_sz=(1,1, 28,28)) # view graph of model

    #print number of parameters
    cols[1].markdown('**Number of parameters:** '+str(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    # define loss function
    criterion = nn.BCELoss()
    
    # define optimizer
    lr = 0.01
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    cols = st.columns((1,1))
    with cols[0]:
        if st.button('Load model from file'):
            try:
                # load model
                model.load_state_dict(torch.load(filepath_assets+'models/'+model_filename))
                st.success('Loaded model from file')
            except:
                st.error('Could not load model from file')
    with cols[1]:
        latent_dim = st.slider('Latent dimension', 2, 10, 2)
        if st.session_state.model_type == 'Linear VAE':
            model = VAE(latent_dim=latent_dim)
            model_filename = 'vae.pt'
        else:
            model = VAE_with_conv2d(latent_dim=latent_dim)
            model_filename = 'vae_conv.pt'
        train_button = st.button('Train model')
    if train_button:

        # train model
        epochs = 3
        train_losses = []
        test_losses = []
        # center text in col
        
        st.markdown("""<div style="text-align: center, font-type: bold
    ">Train Loss</div>""", unsafe_allow_html=True)
        loss_chart = st.line_chart() 
        # add labels to chart


        
        for i in range(epochs):
            train_loss = 0.0
            test_loss = 0.0
            accuracy = 0
            
            for (images_train, labels_train), (images_test, labels_test) in zip(train_loader, test_loader):
                model.train()
                optimizer.zero_grad()
                output_train, mu_train = model(images_train)
                if st.session_state.model_type == 'Linear VAE':
                    loss_train = criterion(output_train, images_train.view(-1, 784)) + 0.0001*torch.sum(mu_train**2)
                else:
                    loss_train = criterion(output_train, images_train.view(-1,1, 28,28)) + 0.0001*torch.sum(mu_train**2)
                loss_train.backward()
                optimizer.step()
                train_loss += loss_train.item()*images_train.size(0)
                
                
                # eval
                model.eval()
                output_test, mu_test = model(images_test)
                if st.session_state.model_type == 'Linear VAE':
                    loss_test = criterion(output_test, images_test.view(-1, 784)) + 0.0001*torch.sum(mu_test**2)
                else:
                    
                    loss_test = criterion(output_test, images_test.view(-1,1, 28,28)) + 0.0001*torch.sum(mu_test**2)
                test_loss += loss_test.item()*images_test.size(0)
                loss_chart.add_rows([{'train loss' :loss_train.item(), 'test loss' :loss_test.item()}])

            train_loss = train_loss/len(train_loader.sampler)
            test_loss = test_loss/len(test_loader.sampler)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            print('Epoch: {} \tTraining Loss: {:.6f} \tTest Loss: {:.6f}'.format(
                i+1, train_loss, test_loss))
            
            lr *= .5
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        # save model
        torch.save(model.state_dict(), filepath_assets+'models/'+model_filename)
        
    
    
    '''---'''

    # view model output -> latent space
    st.markdown(text_dict['latent space'], unsafe_allow_html=True)
    cols = st.columns(2)
    # lets make the latent space
    # we will take all images, encode them 
    # USE PCA TO REDUCE DIMENSIONALITY
    # and plot them in 2D

    # encode all images
    for images, labels in train_loader:
        if st.session_state.model_type == 'Linear VAE':
            z = model.encode(images.view(-1, 784))
        else:
            z = model.encode(images.view(-1,1,28,28))
            z = model.reparameterize(z)

        break


    # reduce dimensionality
    pca = PCA(n_components=2)
    z = pca.fit_transform(z.detach().numpy())

    # plot
    fig = plt.figure(figsize=(10, 10))
    plt.scatter(z[:, 0], z[:, 1], c=labels, cmap='tab10')
    
    # PCs should be randomly distributed points in the span of z
    PCs = np.random.normal(np.mean(z[:,:2], axis=0), np.std(z[:,:2], axis=0), size=(4,2))

    # PCs = np.array([[-.5,1],
    #                 [1.1, .1],
    #                 [-.47,-.8],
    #                 [1.1, -.5],
    #                 ])

    plt.scatter(PCs[:,0], PCs[:,1], c='black', s=200, marker='x')
    cols[0].pyplot(fig)

    fig_latent, ax_latent = plt.subplots(2, 2, figsize=(10, 10))
    # transform PC1 and PC2 to latent space
    for i, (PC1, PC2) in enumerate(PCs):
        ax = ax_latent[i//2, i%2]
        z_trans = pca.inverse_transform([PC1, PC2])

        # z_trans to tensor
        z_trans = torch.Tensor(z_trans)

        ax.imshow(model.decode(z_trans).view(1, 28, 28).detach().numpy()[0], cmap='gray')
    cols[1].pyplot(fig_latent)
    plt.close()

def generative_adversarial_networks():
    # load text
    text_dict = getText_prep_new(filepath_assets+'gan.md')

    # Title
    st.markdown('# Generative Adversarial Networks')

    # intro text
    st.markdown(text_dict['intro'], unsafe_allow_html=True)

    # load data
    train_data = datasets.MNIST(filepath_assets+'data', train=True, download=True, transform=transforms.ToTensor())
    test_data = datasets.MNIST(filepath_assets+'data', train=False, download=True, transform=transforms.ToTensor())

    # define data loaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)

    # show some samples
    #st.markdown(text_dict['samples'], unsafe_allow_html=True)
    
    for images, labels in train_loader:
        break
    
    fig, ax = plt.subplots(1, 4, figsize=(10, 10))
    for i in range(4):
        ax[i].imshow(images[i].view(28, 28), cmap='gray')
        if i != 0:
            ax[i].axis('off')
    st.pyplot(fig)


    # define model
    class Discriminator(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 16, 3, stride=2, padding=1)
            self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
            self.conv3 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
            self.fc1 = nn.Linear(64*4*4, 1)
            self.dropout = nn.Dropout(0.4)
        def forward(self, x):
            x = F.leaky_relu(self.conv1(x), 0.2)
            x = self.dropout(x)
            x = F.leaky_relu(self.conv2(x), 0.2)
            x = self.dropout(x)
            x = F.leaky_relu(self.conv3(x), 0.2)
            x = self.dropout(x)
            x = x.view(-1, 64*4*4)
            x = self.fc1(x)
            return torch.sigmoid(x)
        
    class Generator(nn.Module):
        def __init__(self, z_dim=10, im_chan=1, hidden_dim=64):
            super().__init__()
            self.z_dim = z_dim
            self.gen = nn.Sequential(
                self.make_gen_block(z_dim, hidden_dim*4),
                self.make_gen_block(hidden_dim*4, hidden_dim*2, kernel_size=4, stride=1),
                self.make_gen_block(hidden_dim*2, hidden_dim),
                self.make_gen_block(hidden_dim, im_chan, kernel_size=4, final_layer=True),
            )
        def make_gen_block(self, input_channels, output_channels, kernel_size=3, stride=2, final_layer=False):
            if not final_layer:
                return nn.Sequential(
                    nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
                    nn.BatchNorm2d(output_channels),
                    nn.ReLU(inplace=True),
                )
            else:
                return nn.Sequential(
                    nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
                    nn.Tanh(),
                )
        def forward(self, noise):
            x = noise.view(len(noise), self.z_dim, 1, 1)
            return self.gen(x)
    

    # initialize models
    D = Discriminator()
    G = Generator()

    # view model
    cols = st.columns((1,1))

    view_model(G, st=cols[0]) # view graph of model
    #print number of parameters
    cols[1].markdown('**Number of parameters:** '+str(sum(p.numel() for p in G.parameters() if p.requires_grad)))

    view_model(D, st=cols[1], input_sz=(1, 28,28)) # view graph of model
    #print number of parameters
    cols[1].markdown('**Number of parameters:** '+str(sum(p.numel() for p in D.parameters() if p.requires_grad)))

    # define loss function
    criterion = nn.BCELoss()

    # define optimizers
    lr = 0.001
    d_optimizer = torch.optim.Adam(D.parameters(), lr)
    g_optimizer = torch.optim.Adam(G.parameters(), lr)

    # display_step is how often to display and record the loss and images
    
    display_step = 20

    def show_tensor_images(image_tensor, num_images=1, size=(1, 28, 28), st=st):
        fig, ax = plt.subplots(1, num_images)
        ax = [ax] if num_images == 1 else ax
        for i in range(num_images):
            ax[i].imshow(image_tensor[i].detach()[0])
            ax[i].axis('off')
        st.pyplot(fig)

    
    # define training function
    def train(D, G, criterion, d_optimizer, g_optimizer, train_loader, num_epochs):
        cols = st.columns(2)
        cols[0].markdown('#### fake')
        cols[1].markdown('#### real')
        c1 = cols[0].empty()  # fake
        c2 = cols[1].empty()  # real
        
        cols[0].markdown('#### discriminator loss')
        discriminator_loss = cols[0].line_chart([])
        cols[1].markdown('#### generator loss')
        generator_loss = cols[1].line_chart([])
        cur_step = 0
        mean_generator_loss = 0
        mean_discriminator_loss = 0
        test_noise = torch.randn(16, 10)
        for epoch in range(num_epochs):
            # Dataloader returns the batches
            for real, _ in train_loader:
                cur_batch_size = len(real)
                
                ## Update discriminator ##
                d_optimizer.zero_grad()
                # 1. Train with real images
                D_real = D(real).view(-1)
                d_real_loss = criterion(D_real, torch.ones_like(D_real))
                # 2. Train with fake images
                z = torch.randn(cur_batch_size, 10)
                fake = G(z)
                D_fake = D(fake.detach()).view(-1)
                d_fake_loss = criterion(D_fake, torch.zeros_like(D_fake))
                # 3. Add up loss and perform backprop
                d_loss = d_real_loss + d_fake_loss
                d_loss.backward(retain_graph=True)
                d_optimizer.step()

                ## Update generator ##
                g_optimizer.zero_grad()
                # 1. Train with fake images and flipped labels
                D_fake = D(fake).view(-1)
                g_loss = criterion(D_fake, torch.ones_like(D_fake))
                # 2. Perform backprop
                g_loss.backward()
                g_optimizer.step()

                # Keep track of the average discriminator loss
                mean_discriminator_loss += d_loss.item() / display_step
                # Keep track of the average generator loss
                mean_generator_loss += g_loss.item() / display_step
                
                
                ## Visualization code ##
                if cur_step % display_step == 0:
                    if cur_step > 0:
                        print(f"Step {cur_step}: Generator loss: {mean_generator_loss}, discriminator loss: {mean_discriminator_loss}")
                    show_tensor_images(fake, st=c1)
                    show_tensor_images(real, st=c2)
                    step_bins = 20
                    num_examples = (len(train_loader.dataset)//cur_batch_size)*cur_batch_size
                    num_steps = num_examples//cur_batch_size
                    
                    # show loss
                    discriminator_loss.add_rows([mean_discriminator_loss])
                    generator_loss.add_rows([mean_generator_loss])
                    
                    mean_generator_loss = 0
                    mean_discriminator_loss = 0


                    
                cur_step += 1

    if st.button('train'):
        st.write('cant train now...')
        if 8==0:
            # train model
            train(D, G, criterion, d_optimizer, g_optimizer, train_loader, num_epochs=10)

            # define function to generate images
            def get_generator_image(G, z_dim, index):
                z = torch.randn(1, z_dim)
                fake_image = G(z)
                img = fake_image[0].detach().numpy()
                img = np.transpose(img, (1, 2, 0))
                img = (img + 1) / 2
                return img
            
            # generate images
            z_dim = 10
            index = 0
            img = get_generator_image(G, z_dim, index)
            
            fig = plt.figure(figsize=(4, 4))
            plt.imshow(img)
            st.pyplot(fig)

def MLops():
    ''
    '# Machine Learning Operations (MLOps)'
    cols = st.columns(2)
    with cols[0]:
        r"""
        * we want to get insights into the behaviour of our models.
        * we want to be able to reproduce our results.
        * we want to be able to deploy our models.
        """

        """
        CI/CD : Continuous Integration / Continuous Deployment. Refers to the process of automating the building, testing and deployment of software.
        """


    with cols[1]:st.image('https://ml-ops.org/img/mlops-loop-en.jpg', width=400)
    '---'
    cols = st.columns((1,1))
    with cols[0]:
        st.markdown("""
            ### Weights and Biases    <img src=https://raw.githubusercontent.com/wandb/assets/04cfa58cc59fb7807e0423187a18db0c7430bab5/wandb-dots-logo.svg width=69, align='right'>
            In practice, we can use [weights and biases](wandb.com) to track our experiments.

            To use W&B, do the following:
            1. Install the library: `pip install wandb`
            2. Import the library: `import wandb`
            3. Login to your account: `wandb login` --> I did this in my terminal
            4. Initialize a new run: `wandb.init(project="my-project-name")`
            5. Log metrics and visualize them in the dashboard: `wandb.log({"Epoch":epoch, "loss": loss})`
            6. Visualize the model: `wandb.watch(model)`

            note: i had to run as super user, i.e. `sudo python3 main.py` to avoid permission errors.


        """, unsafe_allow_html=True)

        """
        ##### Sweeps
        A sweep is a set of runs that each contain different hyperparameters. We do this to find the best hyperparameters for our model.

        To implement a sweep in W&B, do the following:
        1. Define the sweep configuration: `sweep_config = {'method': 'grid', 'parameters': {'learning_rate': {'values': [0.01, 0.001]}}}`
        2. Initialize the sweep: `sweep_id = wandb.sweep(sweep_config, project="my-project-name")`
        3. Define the training function: `def train(): ...`
        4. Define the training loop: `wandb.init() for epoch in range(epochs): ...`
        5. Run the sweep agent: `wandb.agent(sweep_id, function=train)`

        The results are shown on the W&B dashboard.
        """
    with cols[1]:
        st.image('https://assets.website-files.com/5ac6b7f2924c656f2b13a88c/63c6b3b7218b038527171ad3_hero-app.jpg')
    
def Natural_Language_processing():
    ''

    """
    # Natural Language Processing (NLP)
    ## Word2Vec
    """
    cols = st.columns((1,1))
    with cols[0]:
        """
        Finds word embeddings, how are different words related to each other? We can pass these embeddings to a neural network (like a multiLayerPerceptron) to perform NLP tasks.

        The word are embedded as a high-dimensional vector. The resulting embeddings capture the semantic and syntactic relationships between words.

        A slightly better approach to interpreting the embeddings that a simple MLP is a CNN. The CNN can capture the local structure of the text. This is still not great...

        So we use a recurrent neural network (RNN). This is a neural network that has a memory. It can remember the previous words in the sentence and use that information to predict the next word. This is called a language model.

        The RNN has context of previously written words. An improvement is the bi-directional RNN. This is a RNN that has context of both the previous and the next words because it reads from both ends (use `torch.flip()` to get reverse order of words).   
        
        To deal with words outside the corpus provided to the embedding network, we can use a character-level embedding.
        """
    with cols[1]:
        st.markdown("""**Word2vec embbeding**""")
        st.image('https://www.researchgate.net/profile/Hakime-Oeztuerk/publication/339013257/figure/fig1/AS:857837307691008@1581535760669/The-illustration-of-the-Skip-Gram-architecture-of-the-Word2Vec-algorithm-For-a.ppm')
    '---'
    '### Applications of RNNs'
    """
    * classification
        * Sentiment classification: classify a sentence as positive or negative
        * Named entity recognition: classify words in a sentence as names, locations, organizations, etc.
    * Sequence encoding
        * Similarity between sentences: how similar are two sentences?
    * Sequence generation
        * Machine translation: translate a sentence from one language to another.
    """
    tabs = st.tabs(['BACKEND', 'RNN-based classification', 'similarity between sentences', 'machine translation'])
    with tabs[0]: # backend
        st.markdown("""
        ```python
        import torch
        import torch.nn as nn

        class RNNCell(nn.Module):
            def __init__(self, input_size, hidden_size, output_size):
                super(RNNCell, self).__init__()
                self.hidden_size = hidden_size
                self.input_to_hidden = nn.Linear(input_size + hidden_size, hidden_size)
                self.hidden_to_output = nn.Linear(hidden_size, output_size)
            def forward(self, input, hidden_state):
                combined = torch.cat((input, hidden_state), 1)
                hidden_state = nn.Tanh(self.input_to_hidden(combined))
                output = self.hidden_to_output(hidden_state)
                return output, hidden
            
            def init_hidden(self):
                return torch.zeros(1, self.hidden_size)

        class RNN(nn.Module):
            def __init__(self, input_size, hidden_size, output_size, n_layers=1):
                super(RNNCell, self).__init__()
                self.RNN_layers = [RNNCell(input_size, hidden_size, output_size) for _ in range(n_layers)]

            def forward(self, input, hidden=None):
                outputs = []
                if hidden is None:
                    hidden = self.init_hidden()
                for layer_idx in range(len(self.RNN_layers)):
                    layer_outputs = []
                    for input_idx in range(input.size(1)):
                        x = input[:, input_idx, :]
                        output, hidden = self.RNN_layers[layer_idx](x, hidden)
                        layer_outputs.append(output)
                    outputs.append(torch.stack(layer_outputs, dim=1))
                    input = outputs[-1]
                return torch.stack(outputs, dim=1)

            def init_hidden(self):
                return torch.zeros(1, self.hidden_size) 22

        class TextRNNEncoder(nn.Module):
            def __init__(self, vocab_size, input_size, hidden_size, output_size, n_layers=2):
                super(TextRNNEncoder, self).__init__()
                self.Emb = nn.Embedding(vocab_size, input_size)
                self.RNN_f = RNN(input_size, hidden_size, output_size, n_layers)
                self.RNN_b = RNN(input_size, hidden_size, output_size, n_layers)

            def forward(self, input):
                word_embeddings = self.Emb(input)
                outputs_f = self.RNN(word_embeddings)
                outputs_b = self.RNN(torch.flip(word_embeddings, dim=(1,)))
                outputs = torch.cat((outputs_f, torch.flip(outputs_b, dim=(1,)), dim=1)
                return outputs
            


        ```""",)
    with tabs[1]: # RNN-based classification
        st.markdown("""
        ```python
        class TextRNNClassifier(nn.Module):
            def __init__(self, vocab_size, input_size, hidden_size, output_size, n_layers=2, n_classes=2):
                super(TextRNNClassifier, self).__init__()
                self.Emb = nn.Embedding(vocab_size, input_size)
                self.RNN_f = RNN(input_size, hidden_size, output_size, n_layers)
                self.RNN_b = RNN(input_size, hidden_size, output_size, n_layers)
                self.classifier = nn.Linear(hidden_size, n_classes)
                
            def forward(self, input):
                word_embeddings = self.Emb(input)
                outputs_f = self.RNN(word_embeddings)
                outputs_b = self.RNN(torch.flip(word_embeddings, dim=(1,)))
                outputs = torch.cat((outputs_f, torch.flip(outputs_b, dim=(1,)), dim=1)
                logits = self.classifier(outputs[:, -1, -1, :].squeeze(1))
                return outputs

        ```""",)

    with tabs[2]: # similarity between sentences
        st.markdown("""
        ```python
        class TextRNNSimilarity(nn.Module):
            def __init__(self, vocab_size, input_size, hidden_size, output_size, n_layers=2):
                super(TextRNNSimilarity, self).__init__()
                self.rnn_encoder = TextRNNEncoder(vocab_size, input_size, hidden_size,
                output_size, n_layers)
                
            def forward(self, input_s, input_t):
                output_s = self.rnn_encoder(input_s)[:, -1, -1, :]
                output_t = self.rnn_encoder(input_t)[:, -1, -1, :]
                sim_score = F.cosine_similarity(input_s, input_t)
                return sim_score
            
        ```""",)

    with tabs[3]: # machine translation
        st.markdown("""
        ```python
        class TextRNNSGenerator(nn.Module):
            def __init__(self, vocab_size, input_size, hidden_size, output_size, n_layers=2):
                super(TextRNNSimilarity, self).__init__()
                self.rnn_encoder = TextRNNEncoder(vocab_size, input_size, hidden_size,
                output_size, n_layers)
                self.rnn_decoder = TextRNNEncoder(vocab_size, input_size, hidden_size,
                output_size, n_layers)
                self.word_decoder = nn.Linear(hidden_size, vocab_size)

            def forward(self, input):
                encoded_outputs = self.rnn_encoder(input)[:, -1, -1, :]
                decoder_inputs = torch.zeros(1, 1)
                for _ in range(self.max_gen_length):
                    decoder_outputs = self.rnn_encoder(decoder_inputs, hidden=encoded_outputs)[:, -1, -1, :]
                    predicted_logits = self.word_decoder(decoder_outputs)
                    predicted_words = predicted_logits.argmax(1).unsqueeze(0)
                    decoder_inputs = torch.cat((decoder_inputs, predicted_words), dim=1)
                return decoder_inputs 

        ```""",)





    "## Gated Recurrent Units (GRUs)"
    cols = st.columns((1,1))
    with cols[0]:
        """
        
        """

    with cols[1]:
        st.image('https://d2l.ai/_images/gru-3.svg')


    "## Long Short-Term Memory (LSTM)"
    cols = st.columns((1,1))
    with cols[0]:
        """
        Similar to the GRU, but with two hidden states. 
        """

    with cols[1]:
        st.image('https://d2l.ai/_images/lstm-3.svg')
    

    '## ELMo'
    cols = st.columns((1,1))
    with cols[0]:
        """
        Passes embeddings to a bi-directional LSTM network. 

        This embeds words given a context -> So "snake" isnt just a snake, its "snake" given the context: i.e., this man was a "snake" (a bad person).

        """
    with cols[1]:
        st.image('https://www.researchgate.net/publication/356967282/figure/fig1/AS:1099809205288962@1639226356684/The-architecture-of-ELMo.png')


if __name__ == '__main__':
    functions = [landing_page,
                 artificial_neural_networks,
                 convolutional_neural_networks,
                 U_net,
                 autoencoders,
                 generative_adversarial_networks,
                 MLops,
                 Natural_Language_processing]
    with streamlit_analytics.track():
        navigator(functions)
