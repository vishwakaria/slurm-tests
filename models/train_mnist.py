import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import signal
import sys
import os
import time

# Define global variables for the model, optimizer, and other checkpoint-related data
net = None
optimizer = None
checkpoint_path = 'fsx/checkpoints/model.pt'
epoch = 0
sig_received = False

# Define a custom neural network
class CustomNN(nn.Module):
    def __init__(self):
        super(CustomNN, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Define the training function
def train(trainloader, criterion, device, epochs):
    global net
    global optimizer
    global epoch

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
    else:
        start_epoch = 0

    net.to(device)
    net.train()

    for e in range(start_epoch, epochs):
        running_loss = 0.0
        epoch = e
        if sig_received == True:
            checkpoint_and_exit()
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()            
        print(f'Epoch {epoch + 1}, Loss: {running_loss / (i + 1)}')

    print('Finished Training')

"""
# Define a function to ckpt
def checkpoint_and_exit():
    global checkpoint_path
    global net
    global optimizer
    print(f"epoch: {epoch}")
    print(f"Checkpoint Path: {checkpoint_path}")
    print(f"net: {net}") if net is not None else None
    print(f"optimizer: {optimizer}") if optimizer is not None else None
    if os.path.exists(checkpoint_path) and net is not None and optimizer is not None:
        print("inside ckpt")
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        try:
            torch.save(checkpoint, checkpoint_path)
            print("Done saving ckpt")
        except Exception as e:
            print(f"Error saving ckpt: {e}")
    sys.exit(0)

"""
def sigterm_handler(signum, frame):
    global sig_received
    print(f"{signum} caught, setting flag")
    sig_received = True

# Mock checkpoint method to measure time allowed for ckpt
def checkpoint_and_exit():
    try:
        print(" saving checkpoint now")
        start = time.time()
        end = start + 120
        i = 0
        while time.time() < end:
            time.sleep(2)
            print(f"Counter: {i}")
            i += 1
        print("done")
        sys.exit(0)
    except Exception as e:
        print(f"Error saving ckpt: {e}")


# Main script
if __name__ == "__main__":
    # Set device to GPU if available, else CPU
    print('start training')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define data transformations and load the MNIST dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    # Create the custom neural network
    net = CustomNN()

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    epochs = 100
    
    # Set the SIGTERM signal handler
    signal.signal(signal.SIGUSR1, sigterm_handler)
    signal.signal(signal.SIGTERM, sigterm_handler)

    # Train the model, optionally starting from a checkpoint
    train(trainloader, criterion, device, epochs)
    print('done training')
