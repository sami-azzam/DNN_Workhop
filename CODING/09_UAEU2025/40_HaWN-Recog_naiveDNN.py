
import random
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.init

import warnings
warnings.filterwarnings("ignore")

# structuring the NN (layers, connects, etc.)
class DNN(torch.nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(784, 10),
        )
        self.initialize_weights()

    def forward(self, x):
        out = x
        out = out.view(out.shape[0], -1)
        out = self.fc1(out)
        return out

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0.01)
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0.01)
            if isinstance(m, nn.BatchNorm2d):
                torch.nn.init.normal_(m.weight, 0, 0.02)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0.01)

# prepare the MNIST data
mnist_train = dsets.MNIST(
    root='../../DB/MNIST/',
    train=True, # download as train data 
    transform=transforms.ToTensor(),
    download=True)

mnist_test = dsets.MNIST(
    root='../../DB/MNIST/',
    train=False, # download as test data
    transform=transforms.ToTensor(),
    download=True)

# define a function that stores the learned result
def save_checkpoint(model, opt, epoch, path):
    torch.save({
        'model':model.state_dict(),
        'optimizer': opt.state_dict(),
        'epoch': epoch
    }, os.path.join(path, f'model_{epoch}.tar'))

# start of the learning part
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if device == 'cuda':
        torch.cuda.manual_seed_all(777)

    learning_rate = 0.001
    training_epochs = 10
    batch_size = 100

    # specify the folder to store the learned result
    checkpoints_path = './checkpoints'
    if not os.path.isdir(checkpoints_path):
        os.makedirs(checkpoints_path)

    # loads the train data as a number of batches
    training_batches = torch.utils.data.DataLoader(
        dataset=mnist_train,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True)

    model = DNN().to(device)
    L = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    num_batches = len(training_batches)
    print('total number of batches : {}'.format(num_batches))
    for epoch in range(training_epochs):
        total_epoch_loss = 0.0
        for X, Y_label in training_batches:
            X = X.to(device)
            Y_label = Y_label.to(device)
            optimizer.zero_grad()
            Y = model(X)
            batch_avg_loss = L(Y, Y_label)
            batch_avg_loss.backward()
            optimizer.step()
            total_epoch_loss += batch_avg_loss
        save_checkpoint(model, optimizer, epoch, checkpoints_path)
        print('[Epoch: {:>4}] total_epoch_loss = {:>.9}'.format(epoch + 1, total_epoch_loss))

# end of the training part
#------------------------------------------------------------------
# start of the test part

    with torch.no_grad():
        X_test = mnist_test.test_data.float().to(device)
          # loads the test data
        Y_label_test = mnist_test.test_labels.to(device)
        ans = model(X_test) # produces batch_size x 10 tensor
        correct_ans = torch.argmax(ans, 1) == Y_label_test
          # produces batch_size x 1 boolean tensor
        accuracy = correct_ans.float().mean()
        print('Accuracy:', accuracy.item())

        # randomly pick a datum from MNIST test set
        r = random.randint(0, len(mnist_test) - 1)
        X_single_data = mnist_test.test_data[r:r + 1].view(-1, 28 * 28).float().to(device)
        Y_single_data = mnist_test.test_labels[r:r + 1].to(device)
        print('Label: ', Y_single_data.item())
        single_ans = model(X_single_data)
        print('Prediction: ', torch.argmax(single_ans, 1).item())

        plt.imshow(mnist_test.test_data[r:r + 1].view(28, 28), cmap='Greys', interpolation='nearest')
        plt.show()
