import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
import sys
import os
import matplotlib.pyplot as plt
import datetime


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # The input size is 1×28×28
        # The fully-connected layer uses ReLU for activation and has 128 nodes
        # (784+1)×128 = 100480 parameters
        # Linear(in_features=784, out_features=128, bias=True)
        self.fc1 = nn.Linear(784, 128)

        # The fully-connected layer uses ReLU for activation and has 64 nodes
        # (128+1)×64 = 8256 parameters
        # Linear(in_features=128, out_features=64, bias=True)
        self.fc2 = nn.Linear(128, 64)

        # The fully-connected layer uses ReLU for activation and has 10 nodes
        # (64+1)×10 = 650 parameters
        # Linear(in_features=64, out_features=10, bias=True)
        self.fc3 = nn.Linear(64, 10)

        # This convolutional neural network has a total of 100480 + 8256 + 650 = 109386 parameters

    def forward(self, x):
        # Flatten the layer to give it to the fully connected layer (1×28×28 -> 1×784)
        # tensor.view(-1 means an appropriate number of rows, 784 columns)
        x = x.view(-1, 1*28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


def train(args, model, device, train_loader, optimizer, epoch):
    print('\nTraining epoch: {}/{}'.format(epoch, args.epochs))
    model.train()  # Set the module in training mode
    loss_sum = 0
    for batch_id, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        # Zero the parameter gradients
        optimizer.zero_grad() # clear the gradients of all optimized variables
        output = model(data)
        loss = F.nll_loss(output, target, reduction='mean')  # sum up batch loss
        loss.backward()
        loss_sum += loss.item()*data.size(0)  # sum up total loss
        optimizer.step()
        # Display
        if batch_id % args.log_interval == 0:
            print('[{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_id * len(data), len(train_loader.dataset),
                100. * batch_id / len(train_loader), loss.item()))
    average_loss = loss_sum / len(train_loader.dataset)
    # Display
    print('\nAverage loss of training set: {:.6f}'.format(average_loss))
    return average_loss

def test(model, device, test_loader):
    model.eval()  # Set the module in evaluation mode
    loss_sum = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.nll_loss(output, target, reduction='mean')  # sum up batch loss
            loss_sum += loss.item()*data.size(0)  # sum up total loss
            predicted = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += predicted.eq(target.view_as(predicted)).sum().item()
        average_loss = loss_sum / len(test_loader.dataset)
        # Display
        print('\nAverage loss of testing set: {:.6f}'.format(average_loss))
        print('Accuracy: {}/{} ({:.2f}%)'.format(
            correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
        return average_loss


def main():

    model_folder = "./model"
    model_name = "mnist_nn.pt"
    model_path = model_folder + "/" + model_name
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
            # Convert a PIL Image (H x W x C) [0,255] to a torch.FloatTensor (C x H x W) [0.0,1.0]
            transforms.ToTensor(),
            # Scale range from [0.0,1.0] to [-1.0,1.0]
            # input = (input - mean) / std
            transforms.Normalize((0.5,), (0.5,))
            # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ]
    )

    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        # Testing
        if os.path.exists(sys.argv[2]) & os.path.isfile(sys.argv[2]):
            image = plt.imread(sys.argv[2])
            model = Net().to(device)
            model.load_state_dict(torch.load(model_path))
            image = transform(image).float()
            image = image.unsqueeze(0)
            image = image.to(device)
            output = model(image)
            predicted = output.max(1)
            print('[{}] [{}] [Prediction: {}]'.format(device, sys.argv[2], classes[predicted[1]]))

    else:
        # Training settings
        parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
        parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                            help='input batch size for training (default: 64)')
        parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                            help='input batch size for testing (default: 1000)')
        parser.add_argument('--epochs', type=int, default=3, metavar='N',
                            help='number of epochs to train (default: 3)')
        parser.add_argument('--lr', type=float, default=0.002, metavar='LR',
                            help='learning rate (default: 0.002)')
        parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                            help='Learning rate momentum (default: 0.9)')
        parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                            help='how many batches to wait before logging training status')
        parser.add_argument('--save-model', action='store_true', default=False,
                            help='For Saving the current Model')

        args = parser.parse_args()

        # Prepare 60,000 training data
        train_loader = torch.utils.data.DataLoader(
            dataset=torchvision.datasets.MNIST(
                root='./data',
                train=True,  # This is training data
                transform=transform,
                download=True  # Automatically download if there is no dataset
            ),
            batch_size=args.batch_size,
            shuffle=True
        )

        # Prepare 10,000 testing data
        test_loader = torch.utils.data.DataLoader(
            dataset=torchvision.datasets.MNIST(
                root='./data',
                train=False,  # This is testing data
                transform=transform,
                download=True
            ),
            batch_size=args.test_batch_size,
            shuffle=False
        )

        print('Device:', device)
        model = Net().to(device)
        print(model)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

        start_time = datetime.datetime.now()
        for epoch in range(1, args.epochs + 1):
            train_loss = train(args, model, device, train_loader, optimizer, epoch)
            valid_loss = test(model, device, test_loader)
            optimizer.step()
        end_time = datetime.datetime.now()
        print('\nStart time: {}'.format(start_time))
        print('  End time: {}'.format(end_time))

        if args.save_model:
            if not os.path.exists(model_folder):
                os.makedirs(model_folder)
            torch.save(model.state_dict(), model_path)


if __name__ == '__main__':
    main()
