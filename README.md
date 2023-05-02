# -DA-in-GameDev-lab1
# АНАЛИЗ ДАННЫХ И ИСКУССТВЕННЫЙ ИНТЕЛЛЕКТ [in GameDev]
Отчет по лабораторной работе #1 выполнил(а):
- Шмаков Никита Владимирович
- ФО210005
Отметка о выполнении заданий (заполняется студентом):

| Задание | Выполнение | Баллы |
| ------ | ------ | ------ |
| Задание 1 | * | 60 |
| Задание 2 | * | 20 |
| Задание 3 | * | 20 |

знак "*" - задание выполнено; знак "#" - задание не выполнено;

Работу проверили:
- к.т.н., доцент Денисов Д.В.
- к.э.н., доцент Панов М.А.
- ст. преп., Фадеев В.О.

[![N|Solid](https://cldup.com/dTxpPi9lDf.thumb.png)](https://nodesource.com/products/nsolid)

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

Структура отчета

- Данные о работе: название работы, фио, группа, выполненные задания.
- Цель работы.
- Задание 1.
- Код реализации выполнения задания. Визуализация результатов выполнения (если применимо).
- Задание 2.
- Код реализации выполнения задания. Визуализация результатов выполнения (если применимо).
- Задание 3.
- Код реализации выполнения задания. Визуализация результатов выполнения (если применимо).
- Выводы.
- ✨Magic ✨

## Цель работы
Ознакомиться с основными операторами зыка Python на примере реализации линейной регрессии.

## Задание
### Запустить Hello World на Python и Unity

Для Python использовал Jupiter Notebook:
![image_2023-01-26_20-04-46](https://user-images.githubusercontent.com/113372135/215065211-2ed05ec8-b370-4eaf-84e1-0e1d826b8150.png)
![image_2023-01-26_20-11-54](https://user-images.githubusercontent.com/113372135/215065217-1422da4d-1183-4cd6-87a3-74556e6a7853.png)

Unity:
![image_2023-01-27_13-24-41](https://user-images.githubusercontent.com/113372135/215065239-e8666a4a-b088-4daa-8939-392251e38f76.png)


## Задание 1
### Пошагово выполнить каждый пункт раздела "ход работы" с описанием и примерами реализации задач.

Ход работы:
- Произвести подготовку данных для работы с алгоритмом линейной регрессии. 10 видов данных были установлены случайным образом, и данные находились в линейной зависимости. Данные преобразуются в формат массива, чтобы их можно было вычислить напрямую при использовании умножения и сложения.

```py

In [ ]:
#Import the required modules, numpy for calculation, and Matplotlib for drawing
import numpy as np
import matplotlib.pyplot as plt

# define data, and change list to array
x = [3,21,22,34,54,34,55,67,89,99]
x = np.array(x)
y = [2,22,24,65,79,82,55,130,150,199]
y = np.array(y)

#Show the effect of a scatter plot
plt.scatter(x,y)

```
![image_2023-01-27_13-49-55](https://user-images.githubusercontent.com/113372135/215067692-16aa81b1-db92-4099-8fb8-50eb8d751228.png)

- Определите связанные функции. Функция модели: определяет модель линейной регрессии wx+b. Функция потерь: функция потерь среднеквадратичной ошибки. Функция оптимизации: метод градиентного спуска для нахождения частных производных w и b.
```
def model (a, b, x): # F Модели
    return a*x + b


def loss_function(a, b, x, y): # F потерь
    num = len(x)
    prediction = model (a,b,x)
    return (0.5/num) * (np.square(prediction-y)).sum()

def optimize(a,b,x,y):
    num=len(x)
    prediction = model(a,b,x)
    da = (1.0/num) * ( (prediction -y)*x).sum()
    db = (1.0/num) * ((prediction -y).sum())
    a = a - Lr*da
    b = b = Lr*db
    return a, b

def iterate(a, b, x, y, times) :
    for i in range(times):
        a,b = optimize(a,b,x,y)
    return a, b
```
![image_2023-01-27_14-33-09](https://user-images.githubusercontent.com/113372135/215065615-b4011654-a292-4052-9633-430943391eb2.png)

```
a = np.random.rand (1)
print(a)
b = np.random.rand(1)
print (b)
Lr = 0.00001

a,b = iterate(a,b,x,y,1)
prediction=model (a, b, x)
loss = loss_function(a, b, x, y)
print (a, b, loss)
plt.scatter(x, y)
plt.plot(x,prediction)
```
![image_2023-01-27_14-52-17](https://user-images.githubusercontent.com/113372135/215065787-493a0042-bbdb-4c33-ac24-0b608a83fb64.png)


## Задание 2
### Должна ли величина loss стремиться к нулю при изменении исходных данных? Ответьте на вопрос, приведите пример выполнения кода, который подтверждает ваш ответ.

При изменении исходных данных, loss не меняется.
![image_2023-01-27_14-52-17](https://user-images.githubusercontent.com/113372135/215066053-90dd6364-d89c-4560-b4af-7bf0c40d7dee.png)

## Задание 3
### Какова роль параметра Lr? Ответьте на вопрос, приведите пример выполнения кода, который подтверждает ваш ответ. В качестве эксперимента можете изменить значение параметра.

Параметр Lr настраивает кривую, изменяя x и y. Так при меньшем Lr, точность кривой упадёт.
![image_2023-01-27_15-09-43](https://user-images.githubusercontent.com/113372135/215066289-a473d4ed-bd03-4a17-9f62-a19738aafb01.png)
![image_2023-01-27_15-10-27](https://user-images.githubusercontent.com/113372135/215066297-1b829fa2-9e9c-4732-b7ce-c619e58cfebd.png)

## Выводы

Немного разобрался с использованием Unity, через муки и нервы настроил использование Unity и Jupiter. Ознакомился с линейной регрессией в python. 

| Plugin | README |
| ------ | ------ |
| Dropbox | [plugins/dropbox/README.md][PlDb] |
| GitHub | [plugins/github/README.md][PlGh] |
| Google Drive | [plugins/googledrive/README.md][PlGd] |
| OneDrive | [plugins/onedrive/README.md][PlOd] |
| Medium | [plugins/medium/README.md][PlMe] |
| Google Analytics | [plugins/googleanalytics/README.md][PlGa] |

## Powered by

**BigDigital Team: Denisov | Fadeev | Panov**
