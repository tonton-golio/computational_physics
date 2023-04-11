
from utils.utils_global import *
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from sklearn.metrics import confusion_matrix



from torchvision.utils import make_grid
from torchviz import make_dot


# Visualize data
def visualize_from_dataloader(images, labels ,st=st):
    n_to_print = len(images)
    im = make_grid(images[:n_to_print], nrow=n_to_print)
    fig = plt.figure(figsize=(10,10))
    plt.imshow(np.transpose(im.numpy(), (1,2,0)), cmap='viridis')
    for l in range(n_to_print):
        try:
            plt.text(l*30+14, -4, labels[l].item(), color='black', fontsize=20)
        except:
            plt.text(l*30+14, -4, labels[l], color='black', fontsize=20)
    plt.close()
    st.pyplot(fig)

# Define model
class MultilayerPerceptron(nn.Module):
    def __init__(self, in_sz=784, out_sz=10, layers=[120, 84]):
        super().__init__()
        self.fc1 = nn.Linear(in_sz, layers[0])
        self.fc2 = nn.Linear(layers[0], layers[1])
        self.out = nn.Linear(layers[1], out_sz)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return F.log_softmax(x, dim=1)

def view_model(model, input_sz = (1,784), st=st):
    dummy_input = torch.rand(*input_sz)
    dot = make_dot(model(dummy_input), params=dict(model.named_parameters()))
    dot.attr(margin='0', pad = '0')
    st.graphviz_chart(dot, use_container_width=True)



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

class ConvolutionalNetwork(nn.Module):
    '''
    We will use the following structure:
    - convolutional layer
    - max pooling layer
    - convolutional layer
    - max pooling layer
    - linear layer
    - linear layer

    The model should take in a 28x28 image and output a 10x1 vector of logits.
    '''
    def __init__(self, im_shape=(1, 28, 28), n_classes=10, batch_size=100):
        super().__init__()
        self.batch_size = batch_size
        self.im_shape = im_shape
        self.n_classes = n_classes

        # define layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(in_features=8*7*7, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(-1, 8*7*7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x