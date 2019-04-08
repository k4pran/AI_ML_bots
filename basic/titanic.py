import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns


# Extract data
inc_columns = [1, 2, 4, 5, 6, 7, 9]
dataset = pd.read_csv("../datasets/titanic.csv", usecols=inc_columns)
dataset.fillna(dataset.mean(), inplace=True)

X = dataset.iloc[:, 1:]
y = dataset.iloc[:, 0]

# Survivors by sex
male_survived = dataset.query('Survived == 1 and Sex == "male"')
female_survived = dataset.query('Survived == 1 and Sex == "female"')
total_survived = male_survived + female_survived
perc_male_survived = male_survived.shape[0] / total_survived.shape[0]
perc_female_survived = female_survived.shape[0] / total_survived.shape[0]
plt.pie([perc_male_survived, perc_female_survived], labels=["Male", "Female"],
        autopct=lambda p : '{:.2f}%  ({:,.0f})'.format(p, p * sum([male_survived.shape[0], female_survived.shape[0]]) / 100))

# Preprocessing
scaler = MinMaxScaler()
X[['Age', 'Fare']] = scaler.fit_transform(X[['Age', 'Fare']])

# Encode categorical columns
X = pd.get_dummies(X, columns=['Pclass', 'Sex'], drop_first=True)

# Split training and test sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
x_train = torch.tensor(x_train.values, dtype=torch.float)
x_test = torch.tensor(x_test.values, dtype=torch.float)
y_train = torch.tensor(y_train.values, dtype=torch.float)
y_test = torch.tensor(y_test.values, dtype=torch.float)
plt.show()

# Create model
model = torch.nn.Sequential(
    torch.nn.Linear(X.shape[1], 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, 1),
    torch.nn.Sigmoid()
)

loss_fn = torch.nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# Training model

steps = 500
losses = np.zeros(500)
for i in range(steps):
    # Make prediction
    y_pred = model(x_train)

    # Calculate loss
    loss = loss_fn(y_pred, y_train)
    losses[i] = loss
    loss.backward()
    print("Loss at step {} - {:.4f}".format(i, loss.item()))

    # Optimize parameters
    optimizer.step()
    optimizer.zero_grad()

# Graph loss
sns.set_style("darkgrid")
loss_df = pd.DataFrame({'Step': range(500), 'Loss': losses})
ax = sns.lineplot(x='Step', y='Loss', data=loss_df)
plt.show()
print("\n\n")

# Testing model
y_pred = model(x_test)
loss = loss_fn(y_pred, y_test)
print("Test loss - {:.4f}".format(loss.item()))


