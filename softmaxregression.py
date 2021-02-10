import csv
import numpy as np
from matplotlib import pyplot

classNames = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

def loadData():
    designMatrix = np.zeros((150, 6))

    with open('iris.csv', newline='') as csvfile:
        f = csv.DictReader(csvfile)

        index = 0
        for row in f:
            typ = 0

            if row['type'] == classNames[1]:
                typ = 1
            elif row['type'] == classNames[2]:
                typ = 2

            designMatrix[index] = [float(row['sepal_length']), float(row['sepal_width']), float(row['petal_length']), float(row['petal_width']), float(1.0), typ]
            index += 1

    return designMatrix


class SoftmaxRegression:

    def __init__(self):
        self.b0 = np.random.rand(5) * 10.0
        self.b1 = np.random.rand(5) * 10.0
        self.b2 = np.random.rand(5) * 10.0

    def softmax(self, betaUp, x):
        up = np.exp(np.dot(betaUp, x))
        down = np.exp(np.dot(self.b0, x)) + np.exp(np.dot(self.b1, x)) + np.exp(np.dot(self.b2, x))

        a = float(up/down)

        if a == 0.0:
            a = 0.1

        return a


    def predict(self, designMatrix):
        arr = np.zeros((150, 3))

        for index in range(0, 150):
            arr[index] = [self.softmax(self.b0, designMatrix[index][0:5]),
             self.softmax(self.b1, designMatrix[index][0:5]), self.softmax(self.b2, designMatrix[index][0:5])]
        
        return arr

    def cost(self, targets, predictions):
        loss = 0

        for a in range(0, 150):
            for m in range(0, 3):
                loss += targets[a][m]*np.log(predictions[a][m])

        loss *= -1
        loss /= 150

        return loss


    def genTargets(self, designMatrix):
        arr = np.zeros((150, 3))

        for x in range(0, 150):
            s = [0.0, 0.0, 0.0]

            if int(designMatrix[x][5]) == 0:
                s[0] = float(1)
            elif int(designMatrix[x][5]) == 1:
                s[1] = float(1)
            else:
                s[2] = float(1)

            arr[x] = s

        return arr

    def delta(self, a, b):
        if a == b:
            return float(1)
        else:
            return float(0)


    def fit(self, designMatrix):

        rate = 1e-1

        predictions = self.predict(designMatrix)
        targets = self.genTargets(designMatrix)

        cost = self.cost(targets, predictions)
        print (cost)

        while cost > 0.05:
            # Calculate derivatives
            del_b0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
            del_b1 = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
            del_b2 = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

            if cost < 0.1:
                rate = 4e-1
            elif cost < 0.06:
                rate = 6e-1

            for a in range(0, 150):
                for m in range(0, 3):
                    del_b0 += targets[a][m]*designMatrix[a][0:5] * (self.delta(m, 0) - predictions[a][0])
                    del_b1 += targets[a][m]*designMatrix[a][0:5] * (self.delta(m, 1) - predictions[a][1])
                    del_b2 += targets[a][m]*designMatrix[a][0:5] * (self.delta(m, 2) - predictions[a][2])

            del_b0 *= -1
            del_b1 *= -1
            del_b2 *= -1

            del_b0 /= 150
            del_b1 /= 150
            del_b2 /= 150

            # Update weights
            self.b0 = self.b0 - rate*del_b0
            self.b1 = self.b1 - rate*del_b1
            self.b2 = self.b2 - rate*del_b2

            predictions = self.predict(designMatrix)
            cost = self.cost(targets, predictions)
            print (cost)


    def test(self, designMatrix):
        result = 0
        predictions = self.predict(designMatrix)

        for x in range(0, 150):
           m = np.argmax(predictions[x])
           if m == designMatrix[x][5]:
               result += 1
        
        print('Correct Classifications: ', result)
        print(self.b0, self.b1, self.b2)


            
        
    
dM = loadData()

regressor = SoftmaxRegression()

regressor.fit(dM)

regressor.test(dM)