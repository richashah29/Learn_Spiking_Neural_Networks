import torch
import torch.nn as nn
import torch.nn.functional as F
x = torch.rand(5, 3) # creates random 5x3 tensor 
print(x)

# manually creating a tensor:
tensor = torch.tensor([1, 2, 3, 4, 5]) # 1x5
print(tensor)

# various operations
# eg. add two 1x3 tensors
y = torch.tensor([1,2,3])
z = torch.tensor([4,5,6])
sum_yz = y + z
product_yz = y * z  #element-wise multiplication

print(sum_yz, product_yz)

# backpropagation is how you can efficiently calculate gradients of loss function concerning weights of network; similar to graidents in seam carving

# gradient descent: optimization algorithm that minimizes loss; gradient calculated by backpropagation to update weights
# learning rate: size of steps taken towards minimum loss; smaller rate might have slow convergence while larger overshoots the minimum

# classify handwritten digits using MNIST dataset
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(784, 64) # 784 input features, 64 output features
        self.fc2 = nn.Linear(64, 10) # 64 input features, 10 output features (digits) 

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x

# 784 input features because 28x28 pixels; simplify to 64 features and match to 10 possible featuresc
# self has two fully connected layers; the first one is a the flattened image that outputs 64 features
# the second layer takes the 64 values and outputs 10 values, one for each digit class

# forward() defines the forward pass (how the input moves through network); PyTorch calls it when I do model(x)
## x is pass through fc1 (first layer), the ReLU activiation func. (all negative values become 0) and adds non-linearity to the model
## output from first layer goes through second layer (fc2) and gives raw scores (logits) for each 10 digit classes (0-9) then returns the logits

## ^ model has already been trained on the MNIST dataset so it is able to classify


## nn.Module is the base class for all neural networks (provides key infrastructure to store layer, track parameters, etc.)

## super() gives access to parent class's methods
## thus, super().__init__() calls constructor of nn.Moudle (parent class)



# training loop (process data, calculate loss, update weights)
## example of supervised learning:

net = SimpleNet() # neural network ready to train
optimizer = optim.SGD(net.parameters(), lr = 0.001, momentum = 0.9) ## create an optimizer (Stochastic Gradient Descent)
criterion = nn.CrossEntropyLoss()  #loss func; CrossEntropyLoss() used for classification problems (compares predicted class scores, outputs, to actual labels)

# epochs: loops through the dataset multiple times

for epoch in range(num_epochs):
    for images, labels in dataloader: # loop through batches of data from dataloader
        images = images.view(images.size(0), -1)  #flatten the images
        optimizer.zero_grad()  #clear gradients for training step
        outputs = net(images)  #forward pass; run model to get predicted outputs
        loss = criterion(outputs, labels)  #calculate loss; compare outputs (predictions) to labels (answers) and determine loss (the error)
        loss.backward()  #backward pass
        optimizer.step()  #update weights using loss gradients computed

## net.parameters() tells optimizer which weights to update; lr is the learning rate (how big a step to take during updates); momentum smooths updates by including previous updates' direction

## images: batch of input images
## labels: batch of correct answers

## the flattening takes the images from shape (batch_sz, 1, 28, 28) to (batch_sz, 784) because SimpleNet expects 784-length vector (not 2D image)

# important to clear gradients since PyTorch accumulates


## backward() uses backpropagation where PyTorch calculates the gradients of loss wrt each weight using chain rule
