import torch as T
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import sys
import numpy as np
import copy as cp

device = T.device("cuda" if T.cuda.is_available() else "cpu")

def acc(log_ps, labels):
    ps = T.exp(log_ps)
    _, top_class = ps.topk(1, dim=1)
    equals = top_class == labels.view(*top_class.shape)
    accuracy = equals.to(T.float).mean()

    return accuracy

def data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((.5,), (.5,))
    ])

    trainset = datasets.MNIST(".pytorch/MNIST", download=True, train=True, transform=transform)
    valset = datasets.MNIST(".pytorch/MNIST", download=True, train=False, transform=transform)
    trainloader = T.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
    valloader = T.utils.data.DataLoader(valset, batch_size=128, shuffle=True)

    return trainloader, valloader

def train(mdl, data_fn=data, epochs=2, alpha=1e-3):
    optim = T.optim.Adam(mdl.parameters(), lr=alpha)
    criterion = nn.NLLLoss()
    trainloader, valloader = data_fn()

    best_val_so_far = np.inf
    best_model = None
    accs = []
    losses = []
    val_accs = []
    val_losses = []

    print("Started parameter optimization")
    for e in range(1, epochs + 1):
        running_acc = 0
        running_loss = 0
        val_running_acc = 0
        val_running_loss = 0

        # Training
        mdl.train()
        for imgs, lbls in trainloader:
            imgs = imgs.to(device)
            lbls = lbls.to(device)

            log_y_hat = mdl(imgs)
            loss = criterion(log_y_hat, lbls)

            # Parameter optimization
            optim.zero_grad()
            loss.backward()
            optim.step()

            running_acc += acc(log_y_hat, lbls)
            running_loss += loss.item()
        else:
            # Validation
            mdl.eval()
            with T.no_grad():
                for imgs, lbls in valloader:
                    imgs = imgs.to(device)
                    lbls = lbls.to(device)

                    val_log_y_hat = mdl(imgs)
                    val_loss = criterion(val_log_y_hat, lbls)

                    val_running_acc += acc(val_log_y_hat, lbls)
                    val_running_loss += val_loss

        accs.append(running_acc/len(trainloader))
        losses.append(running_loss/len(trainloader))
        val_accs.append(val_running_acc/len(valloader))
        val_losses.append(val_running_loss/len(valloader))

        print("Epoch {:2} | acc: {:5.4f} loss: {:5.4f} | val_acc: {:5.4f} val_loss: {:5.4f}".format(
            e,
            accs[-1],
            losses[-1],
            val_accs[-1],
            val_losses[-1]
        ), end="")

        # Best Model
        if val_running_loss/len(valloader) < best_val_so_far:
            best_model = cp.deepcopy(mdl)
            best_val_so_far = val_running_loss/len(valloader)
            print(" *")
        else:
            print("")

    best_model.train()
    return best_model, (accs, losses, val_accs, val_losses)

class NN(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.out = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x.view(-1, 784)))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.log_softmax(self.out(x), dim=1)

        return x.view(-1, 10)

print("training begins..")
model = NN()
model = model.to(device)
model, metrics = train(model)
