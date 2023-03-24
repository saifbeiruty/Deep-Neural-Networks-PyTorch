import torch
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim



# We have two datasets: Training and testing
train = datasets.MNIST("", train=True, download=True, transform = transforms.Compose([transforms.ToTensor()]))
test = datasets.MNIST("", train=False, download=True, transform = transforms.Compose([transforms.ToTensor()]))

trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # 28 x 28 image connected to 64 neurons (hidden layer 1)
        # Linear is fully connected (Dense)
        # 4 layers
        self.fc1 = nn.Linear(784, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        # 10 output neurons because we have 10 outputs 0,1,2,3,...,9
        self.fc4 = nn.Linear(64, 10)
    
    # We are using the ReLu activation function
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return F.softmax(x, dim=1)


net = Net()
# print(net)

# Building the optimizer. Using Adam algorithm
# First parameters: Which parameters to update? We told it both w and b
# Second is the learning rate
optimizer = optim.Adam(net.parameters(), lr=0.001)

epochs = 3

for epoch in range(epochs):
    for data in trainset:
        X, y = data
        # Zero the gradient after every batch instead of every epoch
        # This is what is usually done in ML.
        net.zero_grad()
        output = net.forward(X.view(-1, 28*28))

        # We use nll loss because our data is scalar. We can't just use MSE
        # because the output will be 2 for example and the expected is 3.
        # 3 - 2 = 1 has no meaning at all
        # Negative log lik elihood
        loss = F.nll_loss(output, y)

        #Propagates the loss backward
        loss.backward()

        #Uses the propagated loss to update the parameters (w and b)
        optimizer.step()
    
    print(loss)

correct = 0
total = 0

with torch.no_grad():
    for data in trainset:
        X, y = data
        output = net.forward(X.view(-1, 784))
        for idx, i in enumerate(output):
            if torch.argmax(i) == y[idx]:
                correct += 1
            total += 1

print("Accuracy: ", round(correct/total, 3)*100, "%")