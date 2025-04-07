import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# ------------------------------------------------------------------
# start of the DNN creation part


# define the structure of DNN (layers, connects, etc.)
class DNN(nn.Module):
    def __init__(self, num_classes=2):
        super(DNN, self).__init__()

        self.layers = torchvision.models.regnet_y_8gf(
            weights=torchvision.models.regnet.RegNet_Y_8GF_Weights.IMAGENET1K_V2
        )

        # --- Original conv1 and maxpool (for 224x224 images):
        # self.layers.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # self.layers.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # --- New modifications for 32x32 images:
        # Replace the first convolution with a smaller one suitable for 32x32 images.
        self.layers.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        # Remove the maxpool layer since pooling on 32x32 images would shrink feature maps too aggressively.
        self.layers.maxpool = nn.Identity()

        self.num_classes = num_classes

        # --- Original prototype initialization (for 224x224 images):
        # self.prototypes = nn.Parameter(torch.randn(num_classes, 1000))
        # --- New: Keep the same since we still use the original fc output of ResNet50 (1000-d).
        self.prototypes = nn.Parameter(torch.randn(num_classes, 1000))

    def forward(self, x):
        out = self.layers(x)
        out_norm = nn.functional.normalize(out, p=2, dim=1)
        prototypes_norm = nn.functional.normalize(self.prototypes, p=2, dim=1)
        # Compute cosine similarity between the normalized features and prototypes.
        cosine_sim = torch.matmul(out_norm, prototypes_norm.t())
        return cosine_sim


if __name__ == "__main__":
    done = False
    if not done:
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


def save_checkpoint(model, opt, epoch, path):
    torch.save(
        {"model": model.state_dict(), "optimizer": opt.state_dict(), "epoch": epoch},
        os.path.join(path, f"model_{epoch}.tar"),
    )


if __name__ == "__main__" and not done:
    # Specify folder for saving checkpoints.
    checkpoints_path = "./checkpoints"
    if not os.path.isdir(checkpoints_path):
        os.makedirs(checkpoints_path)

    # --- Original data transformation for larger images:
    # trans_func = transforms.Compose(
    #     [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    # )
    # --- New: Simplified augmentations for 32x32 images.
    trans_func = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),  # Random crop with padding
            transforms.RandomHorizontalFlip(),  # Random horizontal flip
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    trainset = torchvision.datasets.ImageFolder(
        root="../../DB/train_data_lowRes/", transform=trans_func
    )
    training_batches = torch.utils.data.DataLoader(
        trainset, batch_size=100, shuffle=True
    )
    testset = torchvision.datasets.ImageFolder(
        root="../../DB/test_data_lowRes/", transform=trans_func
    )
    test_batches = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)

    # --- Original loss and optimizer (for CrossEntropyLoss on 224x224 images):
    # L = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # --- New: Use differential learning rates and add a cosine margin modification.
    L = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        [
            {
                "params": model.layers.parameters(),
                "lr": 0.0005,
            },  # Lower learning rate for backbone
            {
                "params": model.prototypes,
                "lr": 0.001,
            },  # Higher learning rate for prototypes
        ]
    )

    # New: Add a learning rate scheduler (decays LR every 5 epochs).
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # New: Define a cosine margin for improved class separation (ArcFace-inspired).
    margin = 0.35

    training_epochs = 15
    for epoch in range(training_epochs):
        total_epoch_loss = 0.0
        model.train()
        for X, Y_label in training_batches:
            X = X.to(device)
            Y_label = Y_label.to(device).long()
            optimizer.zero_grad()
            logits = model(X)  # logits shape: [batch_size, num_classes]

            # --- Original logits usage:
            # loss = L(logits, Y_label)
            # --- New: Apply cosine margin modification:
            # Create a one-hot mask for the true labels.
            mask = torch.zeros_like(logits)
            mask.scatter_(1, Y_label.unsqueeze(1), 1.0)
            # Subtract the margin from the logits corresponding to the true class.
            logits_margin = logits - margin * mask
            loss = L(logits_margin, Y_label)

            loss.backward()
            optimizer.step()
            total_epoch_loss += loss.item()
        scheduler.step()
        save_checkpoint(model, optimizer, epoch, checkpoints_path)
        print(
            "[Epoch:{:>3}] total_epoch_loss = {:>.9f}".format(
                epoch + 1, total_epoch_loss
            )
        )
    # end of the training loop

    N_correct = 0
    total = 0
    with torch.no_grad():
        model.eval()
        for X_forTest, Y_label_forTest in test_batches:
            X_forTest = X_forTest.to(device)
            Y_label_forTest = Y_label_forTest.to(device).long()
            logits = model(X_forTest)
            predicted = torch.argmax(logits, dim=1)
            total += Y_label_forTest.size(0)
            N_correct += (predicted == Y_label_forTest).sum().item()

    print(
        "Performance after executing %d batches of 100 samples: %.2f %%"
        % (len(test_batches), (100 * N_correct / total))
    )
# end of the test part
