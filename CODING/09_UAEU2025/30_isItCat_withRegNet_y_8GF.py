import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# ------------------------------------------------------------------
# start of the DNN creation part


# define the structure of DNN (layers, connects, etc.)
class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.layers = torchvision.models.regnet_y_8gf(
            weights=torchvision.models.regnet.RegNet_Y_8GF_Weights.IMAGENET1K_V2
        )
        num_of_features = self.layers.fc.in_features  # num_of_featurs = 2048 always
        self.layers.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_of_features, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.Dropout(0.1),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        out = self.layers(x)
        return out


if __name__ == "__main__":
    # check if GPU is available
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    torch.manual_seed(777)
    if device == "cuda":
        torch.cuda.manual_seed_all(777)
    elif device == "mps":
        torch.mps.manual_seed(777)

    ######################################################################################
    # the following line creates a DNN, names it as 'model', then transfers it to GPU    #
    ######################################################################################
    model = DNN().to(device)


# end of the DNN creation part
# ------------------------------------------------------------------
# start of the training part


# define a function that stores the learned result
def save_checkpoint(model, opt, epoch, path):
    torch.save(
        {"model": model.state_dict(), "optimizer": opt.state_dict(), "epoch": epoch},
        os.path.join(path, f"model_{epoch}.tar"),
    )


if __name__ == "__main__":

    # specify the folder to store the learned result
    checkpoints_path = "./checkpoints"
    if not os.path.isdir(checkpoints_path):
        os.makedirs(checkpoints_path)

    # set the learning parameters (Part 1)
    learning_rate = 0.001
    training_epochs = 15
    batch_size = 100

    trans_func = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    # load the training data then split them to batches
    trainset = torchvision.datasets.ImageFolder(
        root="../../DB/train_data_lowRes/", transform=trans_func
    )
    training_batches = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True
    )
    # load the test data then split them to batches
    testset = torchvision.datasets.ImageFolder(
        root="../../DB/test_data_lowRes/", transform=trans_func
    )
    test_batches = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False
    )

    # set the learning parameters (Part 2)
    L = nn.MSELoss()  # use mean square error as the loss func
    # other choices are CrossEntropyLoss(), BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # start of the training loop
    num_batches = len(training_batches)
    model.train()  # announcement to use model for training
    # need to announce when using a pre-trained model
    for epoch in range(training_epochs):
        total_epoch_loss = 0.0
        for X, Y_label in training_batches:
            X = X.to(device)
            Y_label = Y_label.to(device)  # Y_label is a 32D vector
            optimizer.zero_grad()  # remove the trace of the previous batch
            Y = model(X)  # compute Y, which is a 32x1 tensor
            batch_avg_loss = L(
                Y.squeeze(), Y_label.float()
            )  # computes batch's avg loss
            batch_avg_loss.backward()  # compute gradient values and take the average
            optimizer.step()  # perform back-propagation based on the avg gradient
            total_epoch_loss += batch_avg_loss
        save_checkpoint(model, optimizer, epoch, checkpoints_path)
        # an epoch is complete, so save the learned result
        print(
            "[Epoch:{:>3}] total_epoch_loss = {:>.9}".format(
                epoch + 1, total_epoch_loss
            )
        )
    # end of the training loop

    # end of the training part
    # ------------------------------------------------------------------
    # start of the test part

    N_correct = 0
    total = 0
    with torch.no_grad():
        model.eval()  # announcement to use model for evaluation
        # need to announce when using a pre-trained model
        for X_forTest, Y_label_forTest in test_batches:
            X_forTest = X_forTest.to(device)
            Y_label_forTest = Y_label_forTest.to(device)
            ans = model(X_forTest).squeeze()
            # model(X_forTest) produces a 32x1 tensor.
            # Y_label_forTest is a 32D vector.
            # squeeze() converts it to a 32D vector
            # such that ans can be compatible with Y_label_forTest
            total += Y_label_forTest.size(0)
            # size(0) gives the size of the first dimension, thus the batch size
            ans[ans >= 0.5] = 1  # round off the ans
            ans[ans < 0.5] = 0  # round off the ans
            N_correct += (ans == Y_label_forTest).sum().item()

    print(
        "Performance after executing %d batches of %d samples: %d %%"
        % (len(test_batches), batch_size, (100 * N_correct / total))
    )


# end of the test part
