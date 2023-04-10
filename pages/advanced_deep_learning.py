
from utils.utils_ADL import *

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
    
    '---'
    
    # train model
    st.markdown(text_dict['training'], unsafe_allow_html=True)
    # cache model
    #@st.cache(allow_output_mutation=True)
    def train_MNIST(model, train_loader, test_loader,optimizer,criterion,  epochs = 10, streamlit_view=True):
        train_losses, test_losses = [], []
        if streamlit_view:
            cols = st.columns(2)
            loss_chart = pd.DataFrame([], columns=['loss'])
            cols[0].markdown('**Loss**')
            loss_chart = cols[0].line_chart(loss_chart)
            # add title to chart
            
            cols[1].markdown('**Accuracy**')
            accu_chart = pd.DataFrame([], columns=['accuracy'])
            accu_chart = cols[1].line_chart(accu_chart)
        for i in range(epochs):
            trn_corr = 0
            tst_corr = 0
            
            # Run the training batches
            for b, (X_train, y_train) in enumerate(train_loader):
                b+=1
                
                # Apply the model
                y_pred = model(X_train.view(100, 784))
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
                    
                    if streamlit_view:
                        df2 = pd.DataFrame([loss.item()], columns=['loss'])
                        loss_chart.add_rows(df2)
                        
                        df2 = pd.DataFrame([trn_corr.item()*100/(100*b)], columns=['accuracy'])
                        accu_chart.add_rows(df2)

                    else:
                        print(f'epoch: {i:2}  batch: {b:4} [{100*b:6}/60000]  loss: {loss.item():10.8f}  \
                        accuracy: {trn_corr.item()*100/(100*b):7.3f}%')
                    
            train_losses.append(loss)

            # Run the testing batches
            with torch.no_grad():
                for b, (X_test, y_test) in enumerate(test_loader):

                    # Apply the model
                    y_val = model(X_test.view(500, 784))

                    # Tally the number of correct predictions
                    predicted = torch.max(y_val.data, 1)[1] 
                    tst_corr += (predicted == y_test).sum()

            loss = criterion(y_val, y_test)
            test_losses.append(loss)
            print(f'TESTING:  loss: {loss.item():10.8f}  accuracy: {tst_corr.item()*100/10000:7.3f}%')
        return train_losses, test_losses
    
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
    text_dict = getText_prep_new(filepath_assets+'cnn.md')
    st.markdown(text_dict['intro'], unsafe_allow_html=True)
    


if __name__ == '__main__':
    functions = [artificial_neural_networks,
                 convolutional_neural_networks]
    with streamlit_analytics.track():
        
        navigator(functions)
