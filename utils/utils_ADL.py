
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

def view_model(model, st=st):
    dummy_input = torch.rand(1, 784)
    dot = make_dot(model(dummy_input), params=dict(model.named_parameters()))
    st.graphviz_chart(dot)

