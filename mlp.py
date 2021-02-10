import csv
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import copy

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoidVec(x):
	out = np.zeros(len(x))

	for i in range(len(x)):
		out[i] = sigmoid(x[i])

	return out.reshape(-1, 1)

def softmax(vector, j):
	return np.exp(vector[j]) / np.sum(np.exp(vector))

def softmaxVec(vector):
	out = np.zeros(len(vector))

	for i in range (len(vector)):
		out[i] = softmax(vector, i)

	return out.reshape(-1, 1)


# NOTE: oneHot outputs a row vector not a column vector!!!
def oneHot(vector):
	v = np.zeros(len(vector))
	v[np.argmax(vector)] = 1.0

	return v

def classFromRaw(vector):
	return np.argmax(vector)

def softmaxJacobian(vector):
	out = np.zeros((len(vector), len(vector)))

	for i in range(0, len(vector)):
		for j in range(0, len(vector)):
			if i == j:
				out[i, j] = softmax(vector, i)*(1 - softmax(vector, j))
			else:
				out[i, j] = -softmax(vector, i)*softmax(vector, j)

	return out

def sigmoidJacobian(vector):
	v = np.zeros(len(vector))

	for i in range(0, len(vector)):
		sigi = sigmoid(vector[i])

		v[i] = sigi*(1 - sigi)

	return np.diag(v)

def diag_oneoverola(ola):
	v = np.ones_like(ola).flatten()

	for i in range(0, len(ola)):
		v[i] /= ola[i]

	return np.diag(v)


class MLP():

	def __init__(self, designMatrix, shape):
		self.data = designMatrix
		self.numLayers = len(shape)
		self.numHiddenLayers = self.numLayers - 2

		# Weights at [0] is W^{(12)}
		# Weights at [i] is W^{(i+1, i+2)}
		self.weights = []
		self.biases = []


		# NOTE: First time building this I banged my head for 5 hours trying to figure out why my
		# network stopped improving at a specific error. It was due to the fact that my weights were
		# too large, making the sigmoid output always 1 and at that point the derivative was very very
		# small so it rounded to zero

		for i in range(0, self.numLayers - 1):
			self.weights.append(np.random.randn(shape[i+1], shape[i]))
			#self.weights[-1] /= 10.0
			self.biases.append(np.random.randn(shape[i+1], 1))
			#self.biases[-1] /= 10.0

		self.weightsBefore = self.weights[:]
		self.biasesBefore = self.biases[:]


		self.shape = shape


	def rawOutput(self, dataPoint):
		out = []

		# Convert it into a column vector (technically a matrix)
		inputData = np.array(dataPoint[:-1]).reshape(-1, 1)

		if len(inputData) != self.shape[0]:
			print ('Incorrect data dimensions:', inputData, ' for the given shape', self.shape)

		out.append(inputData)

		for i in range(0, self.numLayers - 1):
			temp = np.matmul(self.weights[i], out[i])
			temp += self.biases[i]

			if i == self.numLayers - 2:
				out.append(softmaxVec(temp))
			else:
				out.append(sigmoidVec(temp))

		return np.array(out[-1])


	def rawAll(self, dataPoint):
		o = []
		z = []

		# Convert it into a column vector (technically a matrix)
		inputData = np.array(dataPoint[:-1]).reshape(-1, 1)

		if len(inputData) != self.shape[0]:
			print ('Incorrect data dimensions:', inputData, ' for the given shape', self.shape)

		o.append(inputData)

		for i in range(0, self.numLayers - 1):
			zi = np.matmul(self.weights[i], o[i])
			zi += self.biases[i]

			z.append(zi)

			if i == self.numLayers - 2:
				o.append(softmaxVec(zi))
			else:
				o.append(sigmoidVec(zi))

		return np.array(o), np.array(z)


	def predictOne(self, dataPoint):
		rawOut = self.rawOutput(dataPoint)

		prediction = classFromRaw(rawOut)

		return prediction

	def checkCorrectOne(self, dataPoint, prediction):
		if prediction == dataPoint[-1]:
			return 1

		return 0

	def predictData(self, designMatrix):
		numCorrect = 0

		for counter, dataPoint in enumerate(designMatrix):
			prediction = self.predictOne(dataPoint)
			numCorrect += self.checkCorrectOne(dataPoint, prediction)

			print ('Correct: ', dataPoint[-1], ' Predicted: ', prediction)

		print ('Correct: ', numCorrect, ' / ', len(designMatrix))
		print ('Accuracy: ', numCorrect / float(len(designMatrix)))


	def cost(self, designMatrix):
		c = 0

		for counter, dataPoint in enumerate(designMatrix):
			y = np.zeros(self.shape[-1])
			y[int(dataPoint[-1])] = 1.0

			yHat = self.rawOutput(dataPoint)

			#print (np.sum(yHat))

			#print('Yhat', yHat)
			yHat = np.log(yHat)

			#print (dataPoint)

			c += np.dot(y, yHat)

		c *= -1.0
		c /= len(designMatrix)

		return c

	def train(self, designMatrix):
		learningRate = 1e-1

		error = self.cost(designMatrix)
		print (error)

		while error >= 0.01:
			print (error)
			# Do weights update
			weightsDerivs = [np.zeros_like(self.weights[i]) for i in range(0, len(self.weights))]
			biasesDerivs = [np.zeros_like(self.biases[i]) for i in range(0, len(self.biases))] 
			
			for numDataPoint, dataPoint in enumerate(designMatrix):
				os, zs = self.rawAll(dataPoint)

				alphas = []
				
				# Calculate the alphas
				for counter in range(len(self.weights)):
					if counter == 0:
						temp = np.matmul(softmaxJacobian(zs[-1]).T, diag_oneoverola(os[-1]))
						ya = np.zeros(self.shape[-1])
						ya[int(dataPoint[-1])] = 1.0

						# Make it a column vector
						ya = ya.reshape(-1, 1)

						# Prepend to the list
						alphas.insert(0, np.matmul(temp, ya))
					else:
						temp = np.matmul(sigmoidJacobian(zs[-counter - 1]), self.weights[-counter].T)

						# The previous alpha is at 0 because we prepended
						alphas.insert(0, np.matmul(temp, alphas[0]))

				# Calculate part(for this example) of deriv of loss wrt weights
				for count, weightDeriv in enumerate(weightsDerivs):
					#print ('Count: ', count)
					#print (alphas)
					#print (os[count].T)
					temp = np.matmul(alphas[count], os[count].T)
					weightsDerivs[count] = np.add(weightDeriv, temp)

				# Calculate part(for this example) of deriv of loss wrt biases
				for count, biasDeriv in enumerate(biasesDerivs):
					biasesDerivs[count] = np.add(biasDeriv, alphas[count])


			for i in range(len(weightsDerivs)):
				weightsDerivs[i] *= -1.0
				weightsDerivs[i] /= float(len(designMatrix))

				biasesDerivs[i] *= -1.0
				biasesDerivs[i] /= float(len(designMatrix))


			# Apply the derivatives
			for counter, weighti in enumerate(self.weights):
				self.weights[counter] = self.weights[counter] - learningRate * weightsDerivs[counter]

			for counter, biasi in enumerate(self.biases):
				self.biases[counter] = self.biases[counter] - learningRate * biasesDerivs[counter]

			error = self.cost(designMatrix)

	def printBeforeAfter(self):
		print('Before: ')
		print ('Weights:')
		print (self.weightsBefore)
		print ('\nBiases:')
		print (self.biasesBefore)
		print ('\n\nAfter:')
		print ('Weights:')
		print (self.weights)
		print ('Biases:')
		print (self.biases)

	def printRaws(self, designMatrix):
		for counter, example in enumerate(designMatrix):
			print ('Ex #', counter, ': ', example, '\n', self.rawOutput(example))





def loadData():
	data = []

	with open('iris-modified-noduplicates.csv', newline='') as csvfile:
		reader = csv.DictReader(csvfile)
		for row in reader:
			rowData = []
			rowData.append(float(row['petal_length']))
			rowData.append(float(row['sepal_length']))
			rowData.append(int(row['type']))

			data.append(rowData)

	return data


rawData = loadData()


# spicing things up

for i in range (0, 13):
	rawData.append([np.random.normal(6.0, 0.7), np.random.normal(2.5, 0.4), 2])
	rawData.append([np.random.normal(2.0, 0.4), np.random.normal(2.5, 0.4), 2])
	rawData.append([np.random.normal(6.0, 0.7), np.random.normal(9.1, 0.15), 2])

# end experimentation

designMat = np.array(rawData)


network = MLP(designMat, [2, 6, 5, 3])
network.train(designMat)
network.predictData(designMat)
network.printBeforeAfter()
#network.printRaws(designMat)


x = np.arange(0, 10.0, 0.2)
y = np.arange(0, 10.0, 0.2)

xx, yy = np.meshgrid(x, y)

xxr = xx.ravel()
yyr = yy.ravel()
Z = np.zeros(len(xxr))

for i in range(0, len(xxr)):
    Z[i] = network.predictOne(np.array([xxr[i], yyr[i], 1.0]))

Z = Z.reshape(xx.shape)

# Plot the boundary
plt.contourf(xx, yy, Z, alpha=0.8)


petalLen = designMat[:, 0]
sepalLen = designMat[:, 1]

plt.xlim(0, 10)
plt.ylim(0, 10)
plt.plot(petalLen, sepalLen, 'bo')

plt.xlabel('Petal Length')
plt.ylabel('Sepal Length')
plt.show()