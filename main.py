from openpyxl import load_workbook
import itertools
import os
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.config.list_physical_devices('GPU')


class Data:
    P205: list[float]
    K20: list[float]
    hydrolytic_acid: list[float]
    PH_water: list[float]
    PH_salt: list[float]
    humus: list[float]


def functoarray(arr, func):
    for i in range(len(arr)):
        arr[i] = func(arr[i])

    return arr


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x*(1-x)


def Loss(y, y_exp):
    loss = 0
    for i in range(len(y)):
        loss += (y[i]-y_exp[i])**2

    return loss


class NeuralNetwork:
    def __init__(self, x, y):
        self.input = x
        self.weights1 = np.random.rand(self.input.shape[1], 70)
        self.weights2 = np.random.rand(70, 1)
        self.y = y
        self.output = np.zeros(self.y.shape)

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = np.dot(self.layer1, self.weights2)

    def backprop(self):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1

        d_weights2 = np.dot(self.layer1.T, (2 * (self.y - self.output) * sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T, (np.dot(2 * (self.y - self.output) * sigmoid_derivative(self.output),
                                                  self.weights2.T) * sigmoid_derivative(self.layer1)))

        # update the weights with the derivative (slope) of the loss function
        self.weights1 += d_weights1
        self.weights2 += d_weights2


# Load in the workbook
wb = load_workbook("./i_kanalov-1.xlsx")
sheet = wb[wb.sheetnames[0]]
count = sum(1 for _ in itertools.takewhile(lambda y: sheet[f"A{y}"].value is not None, itertools.count(2)))
data = Data()
data.P205 = [sheet[f"C{x}"].value for x in range(2, 2 + count)]
data.K20 = [sheet[f"D{x}"].value for x in range(2, 2 + count)]
data.hydrolytic_acid = [sheet[f"E{x}"].value for x in range(2, 2 + count)]
data.PH_water = [sheet[f"F{x}"].value for x in range(2, 2 + count)]
data.humus = [sheet[f"G{x}"].value for x in range(2, 2 + count)]
data.PH_salt = [sheet[f"H{x}"].value for x in range(2, 2 + count)]

InputArr = []
OutputArr = []
for i in range(count):
    InputArr.append([data.P205[i], data.K20[i], data.PH_salt[i], data.PH_water[i], data.hydrolytic_acid[i]])
    OutputArr.append([data.humus[i]])

InputArr = np.array(InputArr, dtype=float)
OutputArr = np.array(OutputArr, dtype=float)
#
# CurModel = NeuralNetwork(InputArr, OutputArr)
# CurModel.feedforward()
# CurModel.backprop()
#
# for i in range(1, count*200):
#     CurModel.feedforward()
#     CurModel.backprop()
#     print(Loss(CurModel.output, CurModel.y))

model = Sequential()
model.add(Dense(count, input_dim=5, activation='relu'))
model.add(Dense(count, activation='relu'))


model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(InputArr, OutputArr, epochs=500)

loss, accuracy = model.evaluate(InputArr, OutputArr)
print(f"Точность модели: {accuracy * 100:.2f}%")

predictions = model.predict(InputArr)
print(predictions)