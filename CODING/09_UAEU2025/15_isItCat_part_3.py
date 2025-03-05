import os
import sys
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn  as nn
import torchvision
import torchvision.transforms as transforms


# define the structure of DNN (layers, connects, etc.)
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

#--------------------------------------------------------

# func to load the input image
def load_image(path):
    image = Image.open(path)
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    image_tensor = transform(image)
    return image_tensor


#------------------------------------------------------------------
# start of the test part


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = DNN().to(device)    # create a DNN

    # reinstate the learned result
    checkpoints_path = './checkpoints'
    if not os.path.isdir(checkpoints_path):
        print("There is no learned result...\n")
        exit()
    checkpoint = torch.load('./checkpoints/model_4.tar')
    model.load_state_dict(checkpoint['model'])

    # load the image
    input_tensor = load_image(f'./{sys.argv[1]}')

    # let the image pass through the DNN
    with torch.no_grad():
        x = input_tensor.view(-1, 3, 32, 32).to(device)
        y = model(x)
        decision = (y.item() >= 0.5)
        print(decision)
