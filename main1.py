import pandas as pd
import numpy as np
import math
from scipy.stats import mode
pd.set_option('display.max_columns', None)
data = pd.read_csv("homework.csv")
data['Цвет глаз'] = data['Цвет глаз'].fillna(data['Цвет глаз'].mode()[0])
data = pd.get_dummies(data, columns=['Цвет глаз', 'Высшая школа', 'Округ'])
data['Во сколько встаете'] = (data['Во сколько встаете']-np.min(data['Во сколько встаете']))/(np.max(data['Во сколько встаете'])-np.min(data['Во сколько встаете']))

x = np.array(data.drop(["К/ч"], 1))
y = np.array(data["К/ч"])
n_train = math.floor(0.7 * x.shape[0])
n_test = math.ceil((1-0.7) * x.shape[0])
x_train = x[:n_train]
y_train = y[:n_train]
x_test = x[n_train:]
y_test = y[n_train:]

def euclidean_distance(row1, row2):
    distance = 0
    for i in range(len(row1)-1):
        distance += (row1[i] - row2[i])**2
    return np.sqrt(distance)

def prediction(x_train, x_test, k):
    predictions = []

    for i in x_test:

        distances = []

        for j in range(len(x_train)):
            distance = euclidean_distance(np.array(x_train[j, :]), i)
            distances.append(distance)
        distances = np.array(distances)
        dist = np.argsort(distances)[:k]

        classes = y[dist]

        mdclass = mode(classes)
        mdclass = mdclass.mode[0]
        predictions.append(mdclass)

    return predictions

y_pred = prediction(x_train, x_test, 7)
print(y_pred)

sumaccurate = 0
for p in range(len(y_pred)):
    if y_test[p] == y_pred[p]:
        sumaccurate += 1

print('Сошлось:', sumaccurate/len(y_pred), '%')
