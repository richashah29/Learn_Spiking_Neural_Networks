## focusing on Pima Diabetes Dataset (binary classification)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# load the dataset
df = pd.read_csv('pima_diabetes.csv')  # adjust the path as necessary
x = df.drop('Outcome', axis=1).values # data without outputs; just inputs
y = df['Outcome'].values # target column; outputs

# split the dataset into training (80%) and testing (20%)
## random state = 0 ensures reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# standardize the data; subtract mean and divide by standard deviation 
# data like glucose, BMI can hurt training
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

## ^ prepare the dataset for training; load dataset and then separate features from target variables; split data into training and testing sets and standardize features

# define the network architecture:
import torch
import torch.nn as nn
import torch.nn.function as F

class DiabetesPredictor(nn.Module):
    def __init__(self):
        super(DiabetesPredictor, self).__init__()
        self.fc1 = nn.Linear(8, 16) # 8 features; true for this dataset
        self.fc2 = nn.Linear(16, 16)
        self.output = nn.Linear(16, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.output(x)) # apply sigmoid probabilistic func.
        return x


# train the model:
model = DiabetesPredictor()
criterion = nn.BCELoss() # binary cross entropy loss; expects output in [0, 1]
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 100
for epoch in range(epochs):
    # Convert arrays to tensors
    inputs = torch.tensor(X_train, dtype=torch.float32) # the different training data converted to tensors
    labels = torch.tensor(y_train, dtype=torch.float32)

    # Foward pass
    outputs = model(inputs)
    loss = criterion(outputs, labels.unsqueeze(1))

    ## unsqueeze adds new dimension at position 1 as BCELoss expects outputs and loss to be same shape

    ## in this case: outputs = (batch_sz, 1) eg. [ [0.73], [0.22], ...] <-- tensor thus (batch_sz, 1) means rows, and 1 column; 2D
    ## labels = (batch_sz) eg. [1, 0, 1, 0, ...] <-- a 1D list

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')



# evaluate the model using test set:
with torch.no_grad(): # don't want to track gradients
    y_predicted = model(torch.tensor(X_test, dtype=torch.float32)) # put test data into model and determine the predicted
    y_predicted_cls = y_predicted.round()
    acc = y_predicted_cls.eq(torch.tensor(y_test).unsqueeze(1)).sum() / float(y_test.shape[0]) #compares test predictions to actual answer going through each element & returns tensor of 1 or 0 where 1 is correct and 0 is incorrect
    print(f'Accuracy: {acc:.4f}')