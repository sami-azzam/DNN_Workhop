
import random
import os

import matplotlib.pyplot as plt
from PIL import Image

import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.init

import warnings
warnings.filterwarnings("ignore")

## structuring the NN (layers, connects, etc.)
class DNN(torch.nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.keep_prob = 0.5
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.BatchNorm2d(32))
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.BatchNorm2d(64))
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, padding=1),
            torch.nn.BatchNorm2d(128))
        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(4 * 4 * 128, 625),
        )
        self.layer4 = torch.nn.Sequential(
            self.fc1,
            torch.nn.ReLU(),
            torch.nn.Dropout(p = 1 - self.keep_prob))
        self.fc2 = torch.nn.Sequential(
            torch.nn.Linear(625, 10)
        )
        self.initialize_weights()

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.layer4(out)
        out = self.fc2(out)
        return out

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0.01)
            if isinstance(m, torch.nn.Conv2d):
                # torch.nn.init.kaiming_uniform_(m.weight)
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0.01)
            if isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.normal_(m.weight, 0, 0.02)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0.01)


mnist_train = dsets.MNIST(
    root='../../DB/MNIST', # download path
    train=True, # download as train data 
    transform=transforms.ToTensor(),
    download=True)

mnist_test = dsets.MNIST(
    root='../../DB/MNIST',
    train=False, # download as test data
    transform=transforms.ToTensor(),
    download=True)

def save_checkpoint(model, opt, epoch, path):
    torch.save({
        'model':model.state_dict(),
        'optimizer': opt.state_dict(),
        'epoch': epoch
    }, os.path.join(path, f'model_{epoch}.tar'))

def load_checkpoint(path):
    pass

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

# fix the seed
torch.manual_seed(777)

# fix the seed for the case of GPU
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

learning_rate = 0.001
training_epochs = 10
batch_size = 100

checkpoints_path = './checkpoints'
if not os.path.isdir(checkpoints_path):
    os.makedirs(checkpoints_path)

def load_image(path):
    image = Image.open(path)
    image = image.convert('L')
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])
    image_tensor = transform(image)
    return image_tensor

data_loader = torch.utils.data.DataLoader(
    dataset=mnist_train,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True)

# instanciate a DNN
model = DNN().to(device)
L = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
batches = len(data_loader)
print('total number of batches : {}'.format(batches))


## Now, the learning starts!

for epoch in range(training_epochs):
    total_epoch_loss = 0.0
    for X, Y_label in data_loader: # X is a batch, Y is a label batch
        # in MNIST, labels are one-hot encoded
        # so Y_label is a batch of 10D vectors
        X = X.to(device)
        Y_label = Y_label.to(device)
        optimizer.zero_grad()
        Y = model(X)
        batch_avg_loss = L(Y, Y_label)
        batch_avg_loss.backward()
        optimizer.step()
        total_epoch_loss += batch_avg_loss
    save_checkpoint(model, optimizer, epoch, checkpoints_path)
    print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, total_epoch_loss))

with torch.no_grad():
    X_test = mnist_test.test_data.view(len(mnist_test), 1, 28, 28).float().to(device)
    Y_label_test = mnist_test.test_labels.to(device)

    prediction = model(X_test)
    correct_prediction = torch.argmax(prediction, 1) == Y_label_test
    accuracy = correct_prediction.float().mean()
    print('Accuracy:', accuracy.item())

    # Pick a random sample from MNIST test data set
    r = random.randint(0, len(mnist_test) - 1)
    X_single_data = mnist_test.test_data[r:r + 1].view(1, 1, 28, 28).float().to(device)
    Y_single_data = mnist_test.test_labels[r:r + 1].to(device)

    print('Label: ', Y_single_data.item())
    single_prediction = model(X_single_data)
    print('Prediction: ', torch.argmax(single_prediction, 1).item())

    plt.imshow(mnist_test.test_data[r:r + 1].view(28, 28), cmap='Greys', interpolation='nearest')
    plt.show()
