
import os
import torch
import torch.nn  as nn
import torchvision
import torchvision.transforms as transforms

## structuring the NN (layers, connects, etc.)
class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3,6,5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(6),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(6,16,5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(16),
        )       
        self.fc1 = nn.Sequential(
            nn.Linear(16*5*5, 120),
            nn.ReLU(),
            nn.Linear(120, 1),
            nn.Sigmoid()
        )
        self.initialize_weights()

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
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

# instanciate the model and transfer to GPU
model = DNN().to(device) 

#------------------------------------------------------------------
# start of the training part

# define a function that stores the learned result
def save_checkpoint(model, opt, epoch, path):
    torch.save({
        'model':model.state_dict(),     # NN state related part
        'optimizer': opt.state_dict(),
        'epoch': epoch                  # current epoch
    }, os.path.join(path, f'model_{epoch}.tar'))

# start of the learning part
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(777)
    if device == 'cuda':
        torch.cuda.manual_seed_all(777)
        
    learning_rate = 0.001
    training_epochs = 10
    batch_size = 32

    # specify the folder to store the learned result
    checkpoints_path = './checkpoints'  
    if not os.path.isdir(checkpoints_path):
        os.makedirs(checkpoints_path)

    trans_func = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    trainset = torchvision.datasets.ImageFolder(root='../../DB/train_data_lowRes/', transform=trans_func)
    training_batches = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    testset = torchvision.datasets.ImageFolder(root='../../DB/test_data_lowRes/', transform=trans_func)
    test_batches = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)


    L = nn.MSELoss()    # use mean square error as the loss func
                        # other choices are CrossEntropyLoss(), BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    num_batches = len(training_batches)
    for epoch in range(training_epochs):
        total_epoch_loss = 0.0
        for X, Y_label in training_batches:
            X = X.to(device)
            Y_label = Y_label.to(device) # Y_label is a 100D vector
            optimizer.zero_grad() # remove the trace of the previous batch
            Y = model(X)    # compute Y, which is a 100x1 tensor
            batch_avg_loss = L(Y.squeeze(), Y_label.float())    # computes batch's avg loss
              # squeeze() and float(): for Y and Y_label to have compatable shape & type for calculation
            batch_avg_loss.backward() # compute gradient values and take the average
            optimizer.step()    # perform back-propagation based on the avg gradient
            total_epoch_loss += batch_avg_loss
        save_checkpoint(model, optimizer, epoch, checkpoints_path)
          # an epoch is finished, so save the learned result
        print('[Epoch: {:>4}] total_epoch_loss = {:>.9}'.format(epoch + 1, total_epoch_loss))

# end of the learning part
#------------------------------------------------------------------
# start of the batch test

    N_correct = 0
    total = 0
    with torch.no_grad(): # do not compute gradient, since this is test
        for X_forTest, Y_label_forTest in test_batches:
            X_forTest = X_forTest.to(device)
            Y_label_forTest = Y_label_forTest.to(device)
            ans = model(X_forTest).squeeze()
              # model(X_forTest) produces a 100x1 tensor.
              # Y_label_forTest is a 100D vector.
              # squeeze() converts it to a 100D vector
              # such that ans can be compatible with Y_label_forTest
            total += Y_label_forTest.size(0)
              # size(0) gives the size of the first dim, thus the batch size.
            ans[ans >= 0.5] = 1  # round off the ans
            ans[ans < 0.5] = 0   # round off the ans
            N_correct += (ans == Y_label_forTest).sum().item()

    print('Performance after executing %d batches of %d samples: %d %%' %
          (len(test_batches), batch_size, (100 * N_correct / total)))

# end of the batch test



