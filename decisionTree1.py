import csv
import math

atts = { 'buying' : 0, 'maintenance' : 1, 'doors' : 2, 'persons' : 3, 'lug_boot' : 4, 'safety' : 5 }
attVals = { 
    'buying' : ['vhigh', 'high', 'med', 'low'],
    'maintenance' : ['vhigh', 'high', 'med', 'low'],
    'doors' : ['2', '3', '4', '5more'],
    'persons' : ['2', '4', 'more'],
    'lug_boot' : ['small', 'med', 'big'],
    'safety' : ['low', 'med', 'high']
}
classes = ['unacc', 'acc', 'good', 'vgood']

attsList = ['buying', 'maintenance', 'doors', 'persons', 'lug_boot', 'safety' ]

class Node:
 
    def __init__(self, data, remainingAtts, branch):
        self.leaf = ''
        self.branches = []
        self.splitAttribute = ''
        self.data = data
        self.remainingAttributes = remainingAtts
        self.branchValue = branch
        self.path = ''


class DecisionTree:

    def __init__(self, data, remainingAtts):
        self.root = Node(data, remainingAtts, '')

    def build(self):
        # Build the tree here ID3
        openSet = []
        openSet.append(self.root)

        while openSet:      # this means while openSet is not empty
            currNode = openSet.pop()

            numExPerClass = [0] * len(classes)
            for row in currNode.data:
                numExPerClass[ classes.index(row[6]) ] += 1

            numZeros = 0
            # Check if all the data is homogenious
            for num in numExPerClass:
                if num == 0:
                    numZeros += 1

            if numZeros == len(numExPerClass) - 1:
                # We know the data is homogenious
                currNode.leaf = classes[numExPerClass.index(max(numExPerClass))]
                continue
            
            if not currNode.remainingAttributes:
                # Find the most common class and use that as a leaf node
                currNode.leaf = classes[numExPerClass.index(max(numExPerClass))]
                continue

            # ID3
            # Find best split
            maxIG = 0
            bestSplits = []
            bestAttribute = ''

            currentEntropy = self.calcEntropy(currNode.data)
            newEntropy = 0

            for a in currNode.remainingAttributes:
                # Find the splits this produces
                splits = self.getSplits(currNode.data, a)

                for split in splits:
                    splitEntr = self.calcEntropy(split)
                    normFactor = float(len(split))/float(len(currNode.data))
                    newEntropy += normFactor * splitEntr

                informationGain = currentEntropy - newEntropy

                #print('IG: ', informationGain)

                if informationGain > maxIG:
                    maxIG = informationGain
                    bestSplits = splits
                    bestAttribute = a


            if bestAttribute == '':         # If no attribute leads to information gain
                currNode.leaf = classes[numExPerClass.index(max(numExPerClass))]
                continue

            # At this point, we have found the best attribute
            #if currNode.branchValue == '':
                #print('Best Attr: ', bestAttribute)
                #print(currNode.remainingAttributes)

            currNode.splitAttribute = bestAttribute

            for val in attVals[bestAttribute]:
                
                ra = [attrib for attrib in currNode.remainingAttributes if attrib != bestAttribute]
                currNode.branches.append( Node( bestSplits[attVals[bestAttribute].index(val)], ra, val ) )

                if not bestSplits[attVals[bestAttribute].index(val)]:
                    # If there are no examples for this value, create a leaf node labelled
                    # with the most common class in the parent, i.e. in currNode
                    currNode.branches[-1].leaf = classes[numExPerClass.index(max(numExPerClass))]
                    continue

                openSet.append(currNode.branches[-1])



    def calcEntropy(self, data):
        d = {}
        for row in data:
            if row[6] in d:
                d[row[6]] = d[row[6]] + 1
            else:
                d[row[6]] = 1

        entropy = 0

        for key, val in d.items():
            psubi = float(val)/float(len(data))
            entropy += (psubi * math.log2(psubi)) # p_sub{i}*log(p_sub{i})

        entropy *= -1

        return entropy

    def getSplits(self, data, attribute):

        splits = [ [] for val in attVals[attribute] ]

        for row in data:
            numArray = attVals[attribute].index(row[atts[attribute]]) # Basically finding in which of the subsets should this datapoint go
            splits[numArray].append(row)

        return splits

    def prune(self, validationSet):
        # Get baseline error
        baselineError = self.test(validationSet, 'noprint')
        nodesErased = 0

        openSet = [self.root]
        while openSet:
            currNode = openSet.pop()

            for br in currNode.branches:
                if br.leaf != '':
                    continue

                # Find the most common class in this node
                numExPerClass = [0] * len(classes)
                for row in br.data:
                    numExPerClass[ classes.index(row[6]) ] += 1

                # Make br a leaf node labelled with the most common class
                br.leaf = classes[numExPerClass.index(max(numExPerClass))]

                # Now test the tree
                newError = self.test(validationSet, 'noprint')

                if newError - baselineError < 0.05:
                    # We keep the changes
                    br.splitAttribute = ''
                    nodesErased += 1

                    baselineError = newError

                    print(numExPerClass)
                else:
                    # We don't keep the changes, reset
                    br.leaf = ''
                    openSet.append(br)

        print('Nodes erased via pruning: ', nodesErased)
                



    def eval(self, dataPoint):
        currNode = self.root

        while currNode:

            if currNode.leaf != '':
                return currNode.leaf

            valDataPoint = dataPoint[atts[currNode.splitAttribute]]

            for node in currNode.branches:
                if node.branchValue == valDataPoint:
                    currNode = node


    def test(self, data, p):
        
        correct = 0
        wrong = 0

        for row in data:
            result = self.eval(row)

            if result == row[6]:
                correct += 1
            else:
                wrong += 1

        errorPercent = float(wrong)/float(correct+wrong)

        if p == 'print':
            print('Correct: ', correct)
            print('Wrong: ', wrong)
            print('Error Percent: ', errorPercent)

        return errorPercent

    def printStructure(self):

        openSet = []
        openSet.append(self.root)
        pathNum = 1

        while openSet:
            currNode = openSet.pop()

            if currNode == self.root:
                currNode.path += currNode.splitAttribute
            else:
                currNode.path += '== ' + currNode.branchValue + ' and ' + currNode.splitAttribute
                
                if currNode.leaf != '':
                    currNode.path += ' - ' + currNode.leaf
                    print(f'Path #{pathNum} : ', currNode.path)
                    pathNum += 1
                    continue

            for branch in currNode.branches:
                branch.path = currNode.path
                openSet.append(branch)



def loadData():

    d = []

    with open('trainCars.csv', newline='') as csvfile:
        f = csv.DictReader(csvfile)

        for row in f:
            d.append( [ row['buying'], row['maintenance'], row['doors'], row['persons'], row['lug_boot'], row['safety'], row['class'] ] )

    return d


data = loadData()

# Split data into training set and validation set to do pruning
trainingSet = data[:1500]
validationSet = data[1500:]

tree = DecisionTree(trainingSet, attsList)
tree.build()

n = [0] * len(classes)
for row in trainingSet:
    n[ classes.index(row[6]) ] += 1

#print ('TS: ', n)

n = [0] * len(classes)
for row in validationSet:
    n[ classes.index(row[6]) ] += 1

#print ('VS:', n)


print('Tested on whole data: ')
tree.test(data, 'print')

print ('\nTested on training data: ')
tree.test(trainingSet, 'print')

print ('\nTested on validation data: ')
tree.test(validationSet, 'print')

tree.prune(validationSet)

print('Tested on whole data after pruning: ')
tree.test(data, 'print')

print ('\nTested on training data after pruning: ')
tree.test(trainingSet, 'print')

print ('\nTested on validation data after pruning: ')
tree.test(validationSet, 'print')

tree.printStructure()