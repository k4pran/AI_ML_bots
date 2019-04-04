import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def split(X, y):
    x_1, x_t2, y_1, y_2 = train_test_split(X, y, test_size=0.2)
    return torch.Tensor(x_1), torch.Tensor(x_t2), torch.Tensor(y_1), torch.Tensor(y_2)


dataset = pd.read_csv("../datasets/iris.csv")

X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

lb = LabelBinarizer()
y_encoded = lb.fit_transform(y)
x_train, x_test, y_train, y_test = split(X.values, y_encoded)

model = torch.nn.Sequential(
    torch.nn.Linear(x_train.shape[1], 32),
    torch.nn.ReLU(),
    torch.nn.Linear(32, y_train.shape[1]),
    torch.nn.Softmax()
)

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.03)


# Train
losses = np.zeros(500)
for i in range(500):
    y_pred = model(x_train)

    loss = loss_fn(y_pred, y_train)
    losses[i] = loss
    print("Loss at time step {}: {:.4f}".format(i + 1, loss.item()))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
print("\n\n")

# Graph loss
sns.set_style("darkgrid")
loss_df = pd.DataFrame({'Step': range(500), 'Loss': losses})
ax = sns.lineplot(x='Step', y='Loss', data=loss_df)
plt.show()

# Test
y_pred = model(x_test)
loss = loss_fn(y_pred, y_test)
print("Test Loss: {:.4f}".format(loss))

# Accuracy
pred_labels = torch.argmax(y_pred, dim=1)
actual_labels = torch.argmax(y_test, dim=1)
results = pred_labels == actual_labels
correct_preds = len([i for i in results if i == 1])
incorrect_preds = len(y_test) - correct_preds
accuracy = correct_preds / len(results)
print("Accuracy: {:.4f}".format(accuracy))

# Visualise
conf_matrix = confusion_matrix(actual_labels, pred_labels)
labels = list(set(y))
df_conf_matrix = pd.DataFrame(conf_matrix, index=labels, columns=labels)
plt.figure(figsize=(10, 7))
sns.set(font_scale=1.4)
sns.heatmap(df_conf_matrix, annot=True, annot_kws={"size": 16})
plt.show()
