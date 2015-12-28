import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import operator

__author__ = 'Prem Sai Kumar Reddy Gangana (psreddy@unm.edu)'

# Vocabulary Length and No of Classes are defined.
# Can be changed accordingly
vocablength=61188
nooflabels=20
np.set_printoptions(linewidth=150,precision=6) # Environment options to ease the output viewing

def mlecalculation():
    """ Function to read the training data and their class labels.
        Also to calculate the MLE P(Y_k)for the train data"""
    # Gets the train.data file and store it in an numpy ndarray
    with open('train.data', 'r') as file:
        ncols = len(file.readline().split(' '))
        file.seek(0,0)
        iterable= (int(v) for line in file for v in line.split(' '))
        data = np.fromiter(iterable, int, count=-1)
        cols = data.reshape(data.size / ncols, ncols).transpose()
    # Gets the train.label file and store it in an numpy ndarray
    trainlabel=np.fromfile('train.label',dtype=int,count=-1,sep=" ")
    trainlist=np.array([trainlabel[cols[0]-1]])

    # Appends both the train.data and train.label and put it in an array
    totalarray=np.append(cols,trainlist,axis=0)

    # Calculation of MLE
    mleprob=np.zeros((nooflabels,1))
    labelcount=np.zeros((nooflabels,1))
    for index in np.arange(0,nooflabels):
        for each in trainlabel:
            if(each==(index+1)):
                labelcount[index,0]=labelcount[index,0]+1

    numtrainrecord=trainlabel.size
    mleprob=labelcount/numtrainrecord
    logmleprob=np.log2(mleprob)
    # Transforms countarray to get the data in required format
    countarray=np.zeros((nooflabels,vocablength))
    for each in totalarray.transpose():
        countarray[each[3]-1,each[1]-1]=countarray[each[3]-1,each[1]-1]+each[2]
        
    return(countarray,logmleprob)

def mapcalculation(countarray,alpha):
    """ Function to calulcate and return the MAP for different passed
        values of Alpha"""
    # MAP Calculation
    mapprob=np.zeros((nooflabels,vocablength))
    for rownum,eachrow in enumerate(countarray):
        mapprob[rownum]=(countarray[rownum]+(alpha-1))/((countarray[rownum].sum())+(alpha-1)*vocablength)
       
    logmapprob=np.log2(mapprob)

    return(logmapprob)

def gettestdata():
    """ Function to get the test data and their class labels"""
    # Gets the test.data file and store it in an numpy ndarray
    with open('test.data','r') as tfile:
        ntcols=len(tfile.readline().split(' '))
        tfile.seek(0,0)
        iterable= (int(v) for line in tfile for v in line.split(' '))
        tdata = np.fromiter(iterable, int, count=-1)
        tcols = tdata.reshape(tdata.size / ntcols, ntcols).transpose()
    # Gets the test.label file and store it in an numpy ndarray
    testlabel=np.fromfile('test.label',dtype=int,count=-1,sep=' ')
    testlabel=testlabel.reshape(testlabel.size,1)
    # Appends both the test.data and test.label and put it in an array
    testarray=np.zeros((testlabel.size,vocablength))
    testarray[tcols[0]-1,tcols[1]-1]=tcols[2]

    return(testarray,testlabel)

def prediction(testarray,testlabel,logmleprob,logmapprob):
    """ Method to predict the class label of test data and calculate the
        accuracy of the classifier for a particular alpha value"""
    # Calculates the Ynew and then predict the labels of test data
    predictedlabel=np.zeros((testlabel.size,1))
    correctprediction,incorrrectprediction=0,0
    for rowindex,eachtrow in enumerate(testarray):
        predictedlabel[rowindex,0]=(np.argmax(logmleprob+(np.dot(logmapprob,eachtrow.transpose())).reshape(nooflabels,1)))+1
        if(predictedlabel[rowindex] == testlabel[rowindex]):
            correctprediction=correctprediction+1
        else:
            incorrrectprediction=incorrrectprediction+1

    # Accuracy Calculation
    accuracy=(correctprediction/(correctprediction+incorrrectprediction))*100

    return (predictedlabel,accuracy)

def getconfusionmatrix(testlabel,predictedlabel):
    """ Method to print the confusion matrix and confusion percentages
        of each Newsgroup label with others"""
    # Prints the confusion matrix
    confusionmatrix=confusion_matrix(testlabel,predictedlabel)
    print("Confusion Matrix ( X->Predicted label, Y->Actual label ) \n\n",confusionmatrix,"\n\n")

    # Calculates the confusion percentages per News Group
    confusedvalues=np.zeros((nooflabels,2),dtype=int)
    confusedvalues[:,0]=confusionmatrix.diagonal()
    confusedvalues[:,1]=confusionmatrix.sum(axis=1)-confusedvalues[:,0]
    print("Question 3:\n\nConfusion Percentages per label")
    conf_percent_dict={}
    for labelnum,each in enumerate(confusedvalues[:,1]/confusedvalues.sum(axis=1)):
            conf_percent_dict[labelnum+1]=each
    conf_percent_dict.values()
    print(sorted(conf_percent_dict.items(),key=operator.itemgetter(1),reverse=True),"\n")
    print("Above percentages are explained in detail in the report. \n")

def toprankwords(logmleprob,logmapprob):
    """ Method to rank the words and print them """

    # Obtains the Top Ranked 100 words
    toprankindices=(np.argsort(np.max(logmapprob/np.sum(logmapprob*logmleprob,axis=0),axis=0)))[-100:][::-1]
    vocabtxt=np.loadtxt('vocabulary.txt',dtype=bytes).astype(str)    
    print("\nQuestion 6:\n\nTop Ranked Words\n\n",vocabtxt[toprankindices])
          
if __name__ == '__main__':
    
    countarray,logmleprob=mlecalculation() # Gets the MLE Probability and the Array with count of Xi in Yk with size (20x61188) 
    testarray,testlabel=gettestdata() # Gets the testdata in form (7505x61188) and testlabels(7505x1)
    print("Question 2:\n")
    print("Accuracy when we take beta as 1/|V| \n")
    beta=1/vocablength
    logmapprob=mapcalculation(countarray,1+beta) # Gets the MAP for the traindata with beta as 1/|V|
    predictedlabel,Accuracy=prediction(testarray,testlabel,logmleprob,logmapprob)
    print("--> Beta Val:",beta,", Accuracy:",Accuracy,"% \n")
    # Prints the Confusion matrix for beta as 1/|V|
    getconfusionmatrix(testlabel,predictedlabel)
    
    # Gets the accuracy of the classifier for different Beta/Alpha Values
    print("Question 4:\n\nAccuracy when we take range of beta values from 0.00001 to 0.01 with step length 0.0001 and from 0.01 to 1 with step of 0.01\n")
    lowbetavalues=np.arange(0.00001,0.01,0.0001)
    highbetavalues=np.arange(0.00001,1,0.5)
    Betavalues=np.append(lowbetavalues,highbetavalues,axis=0)
    alphaAccuracy=np.zeros(Betavalues.size)
    for index,betaval in enumerate(Betavalues):
        logmapprob=mapcalculation(countarray,1+betaval)
        predictedlabel,alphaAccuracy[index]=prediction(testarray,testlabel,logmleprob,logmapprob)
        print("--> Beta Val:",betaval,", Accuracy:",alphaAccuracy[index],"%")

    # Genertaes a semilogx plot of logarithimic beta values (X-Axis) and Accuracy (Y-Axis)
    plt.semilogx(Betavalues, alphaAccuracy)
    plt.title('Beta vs Accuracy')
    plt.grid(True)
    plt.xlabel('Beta')
    plt.ylabel('Accuracy')
    plt.show(block=False)

    beta=1/vocablength
    logmapprob=mapcalculation(countarray,1+beta) # Gets the MAP for the traindata with beta as 1/|V|
    toprankwords(logmleprob,logmapprob) #toprankwords need the non-logarithimic probabilites, hence they are raised to the power 2

