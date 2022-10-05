import pandas as pd
import numpy as np
import math
from scipy.stats import mode
# чтобы все колонки высвечивались
pd.set_option('display.max_columns', None)
# открываю файл
data = pd.read_csv(r"C:\Users\Лиза\Downloads\homework.csv")
# закрываю пробел
data['Цвет глаз'] = data['Цвет глаз'].fillna(data['Цвет глаз'].mode()[0])
# перевожу нечисловые данные в числовые с помощью дамми переменных
data = pd.get_dummies(data, columns=['Цвет глаз', 'Высшая школа', 'Округ'])
# перевожу столбик во сколько встаете в шкалу от 0 до 1
data['Во сколько встаете'] = (data['Во сколько встаете']-np.min(data['Во сколько встаете']))/(np.max(data['Во сколько встаете'])-np.min(data['Во сколько встаете']))


# делим данные на тестовую и тренировочную выборки
x = np.array(data.drop(["К/ч"], 1))
y = np.array(data["К/ч"])
n_train = math.floor(0.7 * x.shape[0])
n_test = math.ceil((1-0.7) * x.shape[0])
x_train = x[:n_train]
y_train = y[:n_train]
x_test = x[n_train:]
y_test = y[n_train:]

# евклидово расстояние (сходство между двумя экземплярами данных)
def euclidean_distance(row1, row2):
    distance = 0
    for i in range(len(row1)-1):
        distance += (row1[i] - row2[i])**2
    return np.sqrt(distance)

# функция для нахождения прогноза по knn
def prediction(x_train, y_train, x_test, k):
    predictions = []
# цикл через данные, которые нужно классифицировать
    for i in x_test:

        # список для расстояний
        distances = []

        for j in range(len(x_train)):
            distance = euclidean_distance(np.array(x_train[j, :]), i)
            # расчитываем расстояния с помощью нашей функции евклидова расстояния
            distances.append(distance)
        distances = np.array(distances)
        # сортируем расстояния и выбираем k первых
        dist = np.argsort(distances)[:k]

        # классы отобранных расстояний
        classes = y[dist]

        # выбираем наиболее распространенный класс из k ближайших элементов
        mdclass = mode(classes)
        mdclass = mdclass.mode[0]
        predictions.append(mdclass)

    return predictions

# запускаем процесс на наших данных
y_pred = prediction(x_train, y_train, x_test, 7)
print(y_pred)

# смотрим какая часть сошлась с реальными данными
sumaccurate = 0
for p in range(len(y_pred)):
    if y_test[p] == y_pred[p]:
        sumaccurate += 1

print('Сошлось:', sumaccurate/len(y_pred), '%')
