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

        # --- Original conv1 and maxpool modifications for 32x32 images:
        # self.layers.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # self.layers.maxpool = nn.Identity()
        # (These changes were kept from the previous version for 32x32 images)
        self.layers.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.layers.maxpool = nn.Identity()

        self.num_classes = num_classes

        # --- Original prototype initialization:
        # self.prototypes = nn.Parameter(torch.randn(num_classes, 1000))
        # (Kept as-is since we still use the 1000-d output of ResNet50)
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

    # --- Original data transformation for 32x32 images:
    # trans_func = transforms.Compose([
    #     transforms.RandomCrop(32, padding=4),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    # ])
    # --- New: Add a small random rotation (±10°) to further augment the data.
    trans_func = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),  # New: slight rotation
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
        root="../../DB/test_data_lowRes/",
        transform=trans_func,  # Use same transformation for test
    )
    test_batches = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)

    # --- Original loss and optimizer settings from the previous version:
    # L = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam([
    #     {"params": model.layers.parameters(), "lr": 0.0005},
    #     {"params": model.prototypes, "lr": 0.001}
    # ])
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    # margin = 0.35
    # --- New: Use label smoothing in CrossEntropyLoss, AdamW with weight decay, and CosineAnnealingLR.
    L = nn.CrossEntropyLoss(label_smoothing=0.1)  # New: label smoothing added
    optimizer = torch.optim.AdamW(
        [
            {"params": model.layers.parameters(), "lr": 0.0005},
            {"params": model.prototypes, "lr": 0.001},
        ],
        weight_decay=1e-4,
    )  # New: weight decay added
    training_epochs = 30  # New: Increase number of epochs for better convergence
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=training_epochs
    )  # New: smoother LR decay
    margin = 0.35  # Keeping the same cosine margin

    # --- Original training loop with StepLR and margin:
    # for epoch in range(training_epochs):
    #     total_epoch_loss = 0.0
    #     model.train()
    #     for X, Y_label in training_batches:
    #         X = X.to(device)
    #         Y_label = Y_label.to(device).long()
    #         optimizer.zero_grad()
    #         logits = model(X)
    #         mask = torch.zeros_like(logits)
    #         mask.scatter_(1, Y_label.unsqueeze(1), 1.0)
    #         logits_margin = logits - margin * mask
    #         loss = L(logits_margin, Y_label)
    #         loss.backward()
    #         optimizer.step()
    #         total_epoch_loss += loss.item()
    #     scheduler.step()
    #     save_checkpoint(model, optimizer, epoch, checkpoints_path)
    #     print("[Epoch:{:>3}] total_epoch_loss = {:>.9f}".format(epoch + 1, total_epoch_loss))
    # --- New training loop: (Same margin modification applied, but with new optimizer/scheduler and more epochs)
    for epoch in range(training_epochs):
        total_epoch_loss = 0.0
        model.train()
        for X, Y_label in training_batches:
            X = X.to(device)
            Y_label = Y_label.to(device).long()
            optimizer.zero_grad()
            logits = model(X)  # logits shape: [batch_size, num_classes]

            # Apply cosine margin modification.
            mask = torch.zeros_like(logits)
            mask.scatter_(1, Y_label.unsqueeze(1), 1.0)
            logits_margin = logits - margin * mask
            loss = L(logits_margin, Y_label)

            loss.backward()
            optimizer.step()
            total_epoch_loss += loss.item()
        scheduler.step()  # CosineAnnealingLR updates at the end of each epoch
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
