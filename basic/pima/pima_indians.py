import tensorflow as tf
from tensorflow import keras

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix

# Get data
dataset = pd.read_csv("../../datasets/pima_indians_diabetes.txt")
x_data, y_data = dataset.iloc[:, :8], dataset.iloc[:, 8]

# Feature scaling
scalar = MinMaxScaler()
scalar.fit(x_data)
x_data = scalar.transform(x_data)

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2)

# Create model
model = keras.Sequential([
    keras.layers.Dense(input_shape=(x_train.shape[1],), units=32, activation=keras.activations.relu),
    keras.layers.Dense(units=2, activation=keras.activations.softmax)
])

model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Training model
model.fit(x_train, y_train, epochs=100)

# Evaluating model
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)

predictions = tf.argmax(model.predict(x_test), axis=1)
sess = tf.InteractiveSession()
predictions = predictions.eval(session=sess)
confusion_mat = confusion_matrix(labels=(0, 1), y_pred=predictions, y_true=y_test)
print(confusion_mat)