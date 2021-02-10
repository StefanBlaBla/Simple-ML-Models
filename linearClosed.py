import numpy as np
import csv
import matplotlib.pyplot as plt
import decimal

decimal.getcontext().prec = 50

class LinearRegression:

    def __init__(self):
        self.a = float(np.random.uniform(2.0, 3.0))
        self.b = float(np.random.uniform(2.0, 3.0))

    def predict(self, x):
        return self.a*x + self.b

    def fit_closedform(self, dataX, dataY):
        A = np.zeros((699, 2))
        for x in range(0, 699):
            A[x, 0] = dataX[x]
            A[x, 1] = 1

        b = np.zeros((699, 1))
        for x in range(0, 699):
            b[x, 0] = dataY[x]

        inverse = np.linalg.inv(np.matmul(A.T, A))
        preb = np.matmul(inverse, A.T)
        coefs = np.matmul(preb, b)

        self.a = coefs[0]
        self.b = coefs[1]

    def mean_error(self, original, prediction):
        error = decimal.Decimal(0)
        for o, p in zip(original, prediction):
            od = decimal.Decimal(o)
            pd = decimal.Decimal(p)
            error += (od - pd)**2

        error /= decimal.Decimal(len(prediction))

        return float(error)

    def least_squares(self, original, prediction):
        error = decimal.Decimal(0)
        for o, p in zip(original, prediction):
            od = decimal.Decimal(o)
            pd = decimal.Decimal(p)
            error += (od - pd)**2
        
        return error

    def fit_gradientdescent(self, dataX, dataY):
        
        learningRate = 4.2e-7#9e-8

        # Propagate forward for the first time
        predictions = []
        for x in dataX:
            predictions.append(self.predict(x))

        # Calculate the error using the cost function for the first time
        error_mean = self.mean_error(dataY, predictions)
        error_leastsquares = self.least_squares(dataY, predictions)
        
        iterations = 0

        while error_leastsquares > 9:

            predictions.clear()

            # Propagate forward
            for x in dataX:
                predictions.append(self.predict(x))

            # Calculate the error using the cost function
            error_mean = self.mean_error(dataY, predictions)
            error_leastsquares = self.least_squares(dataY, predictions)

            if error_leastsquares <= 10:
                break
            if iterations > 10000:
                learningRate = 4.25e-7
            if iterations > 20000:
                break
            
            #if error < 11:
            #    learningRate = 29e-5

            # Calculate the partial derivatives
            del_a = 0
            del_b = 0

            iterations += 1

            for d, o, p in zip(dataX, dataY, predictions):
                del_a += (p - o)*d
                del_b += p - o

            del_a *= 2
            del_b *= 2
            #del_a /= len(predictions)
            #del_b /= len(predictions)

            #print (del_a, del_b)

            # Update the parameters(weights)
            self.a = self.a - del_a*learningRate
            self.b = self.b - del_b*learningRate

            print ("m: ", error_mean)
            print ("ls: ", error_leastsquares)

        return iterations


        
def loadData():
    dataX = []
    dataY = []
    with open('train.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            dataX.append(float(row['x']))
            dataY.append(float(row['y']))
        
    return dataX, dataY





# Load data
plotDataX, plotDataY = loadData()

regression = LinearRegression()

# Fit the model using closed form
#regression.fit_closedform(plotDataX, plotDataY)

# Fit the model using gradient descent
iters = regression.fit_gradientdescent(plotDataX, plotDataY)

print (regression.a, regression.b, iters)

# Plotting stuff
sp = np.linspace(0, 100)
plt.plot(plotDataX[0:600], plotDataY[0:600], 'ro')
plt.plot(sp, sp*regression.a + regression.b, '--', linewidth=2)
plt.show()