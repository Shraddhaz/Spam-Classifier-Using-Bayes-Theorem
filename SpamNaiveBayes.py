#Spambase Dataset
import numpy as np
import csv
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

#Global data declaration
fileLocation = "spambase/spambase.data"
dataArr, train, test = [],[],[]
featuresTrain, labelsTrain, featuresTest, labelsTest = [],[],[],[]
prior0=prior1=0
mean0, mean1, std0, std1, prob0, prob1, finalprob = [], [], [], [], [], [], []
s, t = 0,0

#Read from file - spambase.data
def readFile():
    with open(fileLocation) as file:
        reader = csv.reader(file)
        for data in reader:
            dataArr.append(data)
    file.close()
    np.random.shuffle(dataArr)

#Spliting the dataset in two parts of : Spam and not spam
def Split_SpamNotSpam():
    global train, test, featuresTrain, labelsTrain, featuresTest, labelsTest

    Spam, NotSpam = [], []
    for x in dataArr:
        if x[-1] == '1':
            Spam.append(x)
        else:
            NotSpam.append(x)

    NotSpamHalf, SpamHalf = np.array_split(NotSpam,2), np.array_split(Spam,2)
    train = np.concatenate((NotSpamHalf[0], SpamHalf[0]),axis=0)
    test = np.concatenate((NotSpamHalf[1], SpamHalf[1]),axis=0)

#Shuffling the training set and testing set
def shuffle():
    global featuresTrain, labelsTrain, featuresTest, labelsTest
    for i in [train, test]:
        np.random.shuffle(i)

    #Currently, train and test are shuffled
    featuresTrain = np.array([x[:-1] for x in train]).astype(np.float)
    labelsTrain = np.array([x[-1] for x in train]).astype(np.float)
    featuresTest = np.array([x[:-1] for x in test]).astype(np.float)
    labelsTest = np.array([x[-1] for x in test]).astype(np.float)

# Find Prior Probability
def computeProbability():
    global prior0, prior1
    prior0 = (np.count_nonzero(labelsTrain == 0.0)) / len(labelsTrain)
    prior1 = (np.count_nonzero(labelsTrain == 1.0)) / len(labelsTrain)
    print(prior0)
    print(prior1)

#Checking if standard deviation is positive or not.
# If not, put the standard deviation as 0.0001
def putStd(a):
    return a if (a > 0) else 0.0001

# Compute the mean and standard deviation for 57 features for class0 and class1
def TrainNaiveBayes():
    for col in featuresTrain.T:
        NotSpam, Spam = [], []
        for row in range(len(featuresTrain)):
            if(labelsTrain[row]==0):
                NotSpam.append(col[row])
            else:
                Spam.append(col[row])

        mean0.append(np.mean(NotSpam))
        mean1.append(np.mean(Spam))
        std0.append(putStd(np.std(NotSpam)))
        std1.append(putStd(np.std(Spam)))

#Writing a gauss function for testing
def gaussFunction(x, mean, std):
    return (1 / (np.sqrt(2*np.pi) * std)) * np.exp((-1) * (((x-mean)**2) / (2*(std**2))))


#3. Use the Gaussian Naïve Bayes algorithm to classify the instances in your test set, using the Gauss function
def TestNaiveBayes():
    vectorGauss = np.vectorize(gaussFunction)
    for i in featuresTest:
        prob0.append(vectorGauss(i, mean0, std0))
        prob1.append(vectorGauss(i, mean1, std1))

    #Finding the probability of class0 and class1 for all features
    for rowSpam, rowNonSpam in zip(prob0, prob1):
        class0 = np.log(prior0) + np.sum(np.log(rowSpam))
        class1 = np.log(prior1) + np.sum(np.log(rowNonSpam))

        #Finding  argmax
        finalprob.append(float(np.argmax([class0, class1])))

#Printing accuracy, precision, recall and confusion matrix
def printScores():
    print("Accuracy Score: ", accuracy_score(labelsTest, finalprob))
    print("Precision Score: ", precision_score(labelsTest, finalprob))
    print("Recall: ", recall_score(labelsTest, finalprob))
    print("Confusion Matrix: \n", confusion_matrix(labelsTest, finalprob))


#Main function
def main():
    readFile()
    Split_SpamNotSpam()
    shuffle()
    computeProbability()
    TrainNaiveBayes()
    TestNaiveBayes()
    printScores()

#Calling main
if __name__ == "__main__":
    main()
