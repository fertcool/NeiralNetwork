from openpyxl import load_workbook
import itertools
import os
import numpy as np
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# import tensorflow as tf
# tf.config.list_physical_devices('GPU')


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
        loss += (y-y_exp)**2

    return loss

class NeuralNetwork:
    def __init__(self, x, y):
        self.input = x
        self.weights1 = np.random.rand(self.input.shape[1],5)
        self.weights2 = np.random.rand(5,1)
        self.y = y
        self.output = np.zeros(self.y.shape)

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))

    def backprop(self):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1

        d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))

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


CurModel = NeuralNetwork(np.array([[data.P205[0], data.K20[0], data.PH_salt[0], data.PH_water[0], data.hydrolytic_acid[0]]]), np.array(data.humus[0]))
CurModel.feedforward()
CurModel.backprop()
print(Loss(CurModel.output, data.humus[0]))
for i in range(1, count*2):
    CurModel.input = np.array([[data.P205[i % count], data.K20[i % count], data.PH_salt[i % count], data.PH_water[i % count], data.hydrolytic_acid[i % count]]])
    CurModel.y = np.array(data.humus[i % count])
    CurModel.feedforward()
    CurModel.backprop()
    print(Loss(CurModel.output, CurModel.y))











# class DanseNN(tf.Module):
#     def __init__(self, outputs):
#         super().__init__()
#         self.outputs = outputs
#         self.fl_init = False
#     def __call__(self, x):
#         if not self.fl_init:
#             self.w = tf.random.truncated_normal((x.shape[-1], self.outputs), stddev=0.1, name="w")
#             self.b = tf.zeros([self.outputs], dtype=tf.float32, name = "b")
#
#             self.w = tf.Variable(self.w)
#             self.b = tf.Variable(self.b)
#
#             self.fl_init = True
#
#         y = x@self.w + self.b
#         return y
#
#
#
# model = DanseNN(1)
# print(model(tf.constant([[1.0, 2.0]])))