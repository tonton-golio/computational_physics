
from utils.utils_ADL import *
#import PCA
from sklearn.decomposition import PCA

filepath_assets = 'assets/advanced_deep_learning/'

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


def variational_autoencoders():
    # load text
    text_dict = getText_prep_new(filepath_assets+'vae.md')

    # title
    st.markdown('# Variational Autoencoders', unsafe_allow_html=True)

    # intro text
    st.markdown(text_dict['intro'], unsafe_allow_html=True)

    # load data -> fashion MNIST
    transform = transforms.ToTensor()
    train_data = datasets.FashionMNIST(root=filepath_assets+'data', train=True, download=True, transform=transform)
    test_data = datasets.FashionMNIST(root=filepath_assets+'data', train=False, download=True, transform=transform)

    # DataLoader
    torch.manual_seed(69)

    train_loader = DataLoader(train_data, batch_size=300, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=500, shuffle=False)

    # define model
    class VAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(784, 400)
            self.fc2 = nn.Linear(400, 8)
            self.fc3 = nn.Linear(8, 400)
            self.fc4 = nn.Linear(400, 784)
            self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()
        
        def encode(self, x):
            h1 = self.relu(self.fc1(x))
            return self.fc2(h1)
        
        def reparameterize(self, mu):
            std = torch.exp(0.5*mu)
            eps = torch.randn_like(std)
            return mu + eps*std
        
        def decode(self, z):
            h3 = self.relu(self.fc3(z))
            return self.sigmoid(self.fc4(h3))
        
        def forward(self, x):
            mu = self.encode(x.view(-1, 784))
            z = self.reparameterize(mu)
            return self.decode(z), mu
        
    # instantiate model
    model = VAE()

    # view model
    cols = st.columns((1,1))
    cols[0].markdown(text_dict['VAE model'], unsafe_allow_html=True)
    view_model(model, st=cols[1], input_sz=(1,1, 28,28)) # view graph of model

    #print number of parameters
    cols[1].markdown('**Number of parameters:** '+str(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    # define loss function
    criterion = nn.BCELoss()
    
    # define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    if st.button('Train model'):
        # train model
        epochs = 1
        train_losses = []
        test_losses = []
        # center text in col
        st.markdown("""<div style="text-align: center, font-type: bold
        ">Loss</div>""", unsafe_allow_html=True)
        loss_chart = st.line_chart()
        for i in range(epochs):
            train_loss = 0.0
            test_loss = 0.0
            accuracy = 0
            model.train()
            for images, labels in train_loader:
                optimizer.zero_grad()
                output, mu = model(images)
                loss = criterion(output, images.view(-1, 784)) + 0.0001*torch.sum(mu**2)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()*images.size(0)
                loss_chart.add_rows([[loss.item()]])
            model.eval()
            for images, labels in test_loader:
                output, mu = model(images)
                loss = criterion(output, images.view(-1, 784)) + 0.0001*torch.sum(mu**2)
                test_loss += loss.item()*images.size(0)
            train_loss = train_loss/len(train_loader.sampler)
            test_loss = test_loss/len(test_loader.sampler)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            print('Epoch: {} \tTraining Loss: {:.6f} \tTest Loss: {:.6f}'.format(
                i+1, train_loss, test_loss))
            
        # save model
        torch.save(model.state_dict(), filepath_assets+'models/vae.pt')
            
        
    
    # load model
    model.load_state_dict(torch.load(filepath_assets+'models/vae.pt'))
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
        z = model.encode(images.view(-1, 784))

        break


    # reduce dimensionality
    pca = PCA(n_components=2)
    z = pca.fit_transform(z.detach().numpy())

    # plot
    fig = plt.figure(figsize=(10, 10))
    plt.scatter(z[:, 0], z[:, 1], c=labels, cmap='tab10')
    

    PCs = np.array([[-.5,1],
                    [1.1, .1],
                    [-.47,-.8],
                    [1.1, -.5],
                    ])

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
        ax[i].imshow(images[i].permute(1,2,0))
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


    



if __name__ == '__main__':
    functions = [artificial_neural_networks,
                 convolutional_neural_networks,
                 U_net,
                 variational_autoencoders,
                 generative_adversarial_networks]
    with streamlit_analytics.track():
        
        navigator(functions)


