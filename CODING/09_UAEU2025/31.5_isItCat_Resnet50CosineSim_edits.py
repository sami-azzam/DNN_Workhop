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

        self.layers = torchvision.models.resnet50(
            weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2
        )
        self.num_classes = num_classes
        # Create learnable prototypes for each class.
        # nn.Parameter ensures that the tensor is treated as a model parameter.
        # torch.randn(num_classes, 1000) initializes the prototypes with random values.
        self.prototypes = nn.Parameter(torch.randn(num_classes, 1000))

        # --- Original code that replaced the output layer (commented out):
        # num_of_features = self.layers.fc.in_features  # num_of_featurs = 2048 always
        # self.layers.fc = nn.Sequential(
        #     nn.Dropout(0.5),
        #     nn.Linear(num_of_features, 1024),
        #     nn.Dropout(0.2),
        #     nn.Linear(1024, 512),
        #     nn.Dropout(0.1),
        #     nn.Linear(512, 1),
        #     nn.Sigmoid(),
        # )

    def forward(self, x):
        out = self.layers(x)
        out_norm = nn.functional.normalize(out, p=2, dim=1)
        prototypes_norm = nn.functional.normalize(self.prototypes, p=2, dim=1)
        # Compute cosine similarity between features and prototypes.
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


# define a function that stores the learned result
def save_checkpoint(model, opt, epoch, path):
    torch.save(
        {"model": model.state_dict(), "optimizer": opt.state_dict(), "epoch": epoch},
        os.path.join(path, f"model_{epoch}.tar"),
    )


if __name__ == "__main__" and not done:
    # Specify the folder to store the learned result.
    checkpoints_path = "./checkpoints"
    if not os.path.isdir(checkpoints_path):
        os.makedirs(checkpoints_path)

    # --- Original data transformation:
    # trans_func = transforms.Compose(
    #     [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    # )
    # --- New enhanced data augmentation and normalization using ImageNet stats:
    trans_func = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),  # Crop & resize to 224x224
            transforms.RandomHorizontalFlip(),  # Random horizontal flip
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
            ),  # Color jitter
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Load training and test datasets.
    trainset = torchvision.datasets.ImageFolder(
        root="../../DB/train_data_lowRes/", transform=trans_func
    )
    training_batches = torch.utils.data.DataLoader(
        trainset, batch_size=100, shuffle=True
    )
    testset = torchvision.datasets.ImageFolder(
        root="../../DB/test_data_lowRes/",
        transform=trans_func,  # Use same normalization for test set
    )
    test_batches = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)

    # --- Original loss and optimizer (for CrossEntropyLoss):
    # L = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # --- New: Differential learning rates (lower for backbone, higher for prototypes) and using CrossEntropyLoss:
    L = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        [
            {
                "params": model.layers.parameters(),
                "lr": 0.0001,
            },  # Lower LR for pre-trained backbone
            {"params": model.prototypes, "lr": 0.001},  # Higher LR for new prototypes
        ]
    )

    # New Learning Rate Scheduler: StepLR decays LR every 5 epochs by a factor of 0.5.
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # New: Margin for cosine similarity (ArcFace-inspired)
    margin = 0.35

    # --- Original training loop for CrossEntropyLoss:
    # for epoch in range(training_epochs):
    #     total_epoch_loss = 0.0
    #     for X, Y_label in training_batches:
    #         X = X.to(device)
    #         Y_label = Y_label.to(device).long()
    #         optimizer.zero_grad()
    #         Y = model(X)
    #         batch_avg_loss = L(Y, Y_label)
    #         batch_avg_loss.backward()
    #         optimizer.step()
    #         total_epoch_loss += batch_avg_loss.item()
    #         ...
    # --- New training loop with cosine margin modification:
    training_epochs = 15
    for epoch in range(training_epochs):
        total_epoch_loss = 0.0
        model.train()  # Ensure model is in training mode
        for X, Y_label in training_batches:
            X = X.to(device)
            Y_label = Y_label.to(device).long()
            optimizer.zero_grad()
            logits = model(X)  # Output shape: [batch_size, num_classes]

            # Apply cosine margin to the logits for the true class
            # Create one-hot mask for labels.
            mask = torch.zeros_like(logits)
            mask.scatter_(1, Y_label.unsqueeze(1), 1.0)
            # Modify logits: subtract margin from the true class logits.
            logits_margin = logits - margin * mask

            loss = L(logits_margin, Y_label)
            loss.backward()
            optimizer.step()
            total_epoch_loss += loss.item()
        scheduler.step()  # Step the LR scheduler after each epoch
        save_checkpoint(model, optimizer, epoch, checkpoints_path)
        print(
            "[Epoch:{:>3}] total_epoch_loss = {:>.9f}".format(
                epoch + 1, total_epoch_loss
            )
        )
    # end of the training loop

    # --- Original test segment for CrossEntropyLoss:
    # with torch.no_grad():
    #     model.eval()
    #     for X_forTest, Y_label_forTest in test_batches:
    #         X_forTest = X_forTest.to(device)
    #         Y_label_forTest = Y_label_forTest.to(device).long()
    #         logits = model(X_forTest)
    #         predicted = torch.argmax(logits, dim=1)
    #         total += Y_label_forTest.size(0)
    #         N_correct += (predicted == Y_label_forTest).sum().item()
    # --- New test segment remains largely unchanged:
    N_correct = 0
    total = 0
    with torch.no_grad():
        model.eval()  # Set model to evaluation mode
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
