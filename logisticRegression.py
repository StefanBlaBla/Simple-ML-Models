from matplotlib import pyplot as plt
import csv
import numpy as np
import random
import math

class LogisticRegression():
    
    def __init__(self):
        self.b0 = np.random.uniform(-10.0, 10.0)
        self.b1 = np.random.uniform(-10.0, 10.0)
        self.b2 = np.random.uniform(-10.0, 10.0)
        self.rate = 1e-2

    def predict(self, petalLength, sepalLength):
        preds = []

        for pl, sl in zip(petalLength, sepalLength):
            res = self.b0 + self.b1*pl + self.b2*sl
            preds.append(self.logistic(res))

        return preds

    def classify(self, petalLength, sepalLength, targets):
        correct = 0

        for pl, sl, t in zip(petalLength, sepalLength, targets):
            res = self.b0 + self.b1*pl + self.b2*sl
            if res >= 0 and t == 1:
                correct += 1
            if res < 0 and t == 0:
                correct += 1

        return correct
                
    
    def error(self, targets, predictions):
        err = 0

        for h, hHat in zip(targets, predictions):
            # There is a problem where hHat often ends up being 1.0 that causes 1-hHat to be 0
            # which results in an *error*
            # I don't think there is a better way to solve this problem (other than adding riciculous amout of precision to the floats)
            # because the logistic f-on is very very very close to 1.0 for big negative values of z (ex. -100 or -30) or to 0 for big positive vals
            if hHat == 1.0:
                hHat = 0.99
            elif hHat == 0.0:
                hHat = 0.01
            err += h*np.log(hHat) + (1-h)*np.log(1-hHat)

        err *= -1
        err /= len(predictions)

        return err

    def logistic(self, z):
        return float(1.0/(1.0+math.exp(-z)))


    def fit(self, petalLength, sepalLength, typ):
        predictions = self.predict(petalLength, sepalLength)

        loss = self.error(typ, predictions)


        while loss >= 0.005:
            predictions = self.predict(petalLength, sepalLength)
            
            loss = self.error(typ, predictions)
            print (loss)

            if loss <= 0.005:
                break

            del_b0 = 0
            del_b1 = 0
            del_b2 = 0

            for pr, tar, pl, sl in zip(predictions, typ, petalLength, sepalLength):
                #del_b0 += tar*(1-pr) - pr*(1-tar)
                #del_b1 += pl*tar*(1-pr) - pl*pr*(1-tar)
                #del_b2 += sl*tar*(1-pr) - sl*pr*(1-tar)

                # These are the same as the above commented ones if you expand the commented ones
                # so that they simplify by some of the terms cancelling each other
                del_b0 += tar - pr
                del_b1 += pl*(tar-pr)
                del_b2 += sl*(tar-pr)

            del_b0 *= -1
            del_b0 /= len(predictions)

            del_b1 *= -1
            del_b1 /= len(predictions)

            del_b2 *= -1
            del_b2 /= len(predictions)

            self.b0 = self.b0 - self.rate*del_b0
            self.b1 = self.b1 - self.rate*del_b1
            self.b2 = self.b2 - self.rate*del_b2

    

def loadData():
    pl = []
    sl = []
    t = []
    with open('iris-modified.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            pl.append(float(row['petal_length']))
            sl.append(float(row['sepal_length']))
            t.append(int(row['type']))

    return pl, sl, t


petalLengthOrig = []
sepalLengthOrig = []
typ = []

petalLengthOrig, sepalLengthOrig, typ = loadData()        

regressor = LogisticRegression()

regressor.fit(petalLengthOrig, sepalLengthOrig, typ)

print(regressor.b0, regressor.b1, regressor.b2)

print (regressor.classify(petalLengthOrig, sepalLengthOrig, typ))


sp = np.linspace(0, 10)
plt.plot(petalLengthOrig, sepalLengthOrig, 'ro')
plt.plot(sp, -(regressor.b1/regressor.b2)*sp - (regressor.b0/regressor.b2))
plt.xlabel('Petal Length')
plt.ylabel('Sepal Length')
plt.show()