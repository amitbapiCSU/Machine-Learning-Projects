import pandas
import numpy as np
import random as rn
import matplotlib.pyplot as plt
import warnings

# lambdaw = 0.4; #np.random.random_integers(10)/10
nSamples = 50;
def makeStandardize(X):
    means = X.mean(axis=0)
    stds = X.std(axis=0)
    def standardize(origX):
        return (origX - means) / stds
    def unStandardize(stdX):
        return stds * stdX + means
    return (standardize, unStandardize)


def makeLLS(X,T,lambdaw):
    nRows = 0;
    nCols = 0;
    (standardizeF, unstandardizeF) = makeStandardize(X)
    X = standardizeF(X)
    (nRows,nCols) = X.shape
    X = np.hstack((np.ones((X.shape[0],1)), X))    
    penalty = lambdaw * np.eye(nCols+1)
    penalty[0,0]  = 0  # don't penalize the bias weight
    w = np.linalg.lstsq(np.dot(X.T,X)+penalty, np.dot(X.T,T))[0]
    return (w, standardizeF, unstandardizeF)

def useLLS(model,X):
    w, standardizeF, _ = model
    X = standardizeF(X)
    X = np.hstack((np.ones((X.shape[0],1)), X))
    return np.dot(X,w)

X = np.hstack((np.linspace(10, 20, num=nSamples), np.linspace(6, 10, num=nSamples))).reshape((2*nSamples,1))
T = -3 + 4 * X + 0.6*np.random.normal(size=(2*nSamples,1))

j = 0;
nrows = X.shape[0]
plt.figure(figsize=(20,20))
plt.subplots_adjust(hspace=.5)


diff_partitions = rn.sample(range(1000,9999), 1000); # rn.sample(range(10,40), 10)
lambs = np.linspace(0,20,8)*nSamples  # ; [0.0,0.1,0.2,0.3,0.4]; #
result = []
for lambdaw in lambs:
    rmstrn = [];
    rmstst = [];
    i=0;
    for diff in diff_partitions:
        nTrain = int(round(nrows*(diff/10000)))
        nTest = nrows - nTrain
        rows = np.arange(nrows)
        np.random.shuffle(rows)
        trainIndices = rows[:nTrain]
        testIndices = rows[nTrain:]

        Xtrain = X[trainIndices,:]
        Ttrain = T[trainIndices,:]
        Xtest = X[testIndices,:]
        Ttest = T[testIndices,:]

        (standardizeF, unstandardizeF) = makeStandardize(Xtrain)
        Xtrain = standardizeF(Xtrain)
        lambdaI = lambdaw * np.eye(Xtrain.shape[1])
        lambdaI[0,0] = 0
        w = np.linalg.lstsq(np.dot(Xtrain.T,Xtrain) + lambdaI,np.dot(Xtrain.T, Ttrain))[0]
        model = makeLLS(Xtrain,Ttrain,lambdaw)
        predTrain = useLLS(model, Xtrain)
        predTest = useLLS(model, Xtest)
        rmstrn.append(np.sqrt(np.mean((predTrain-Ttrain)**2)))
        rmstst.append(np.sqrt(np.mean((predTest-Ttest)**2)))
        if np.isnan(rmstst[i]):
            rmstst.pop()
            rmstst.append(0)
        i+=1;   
        print('rmse tst ',rmstst)
    result.append([lambdaw, np.mean(rmstrn), np.mean(rmstst)])  # , list(w.flatten())

warnings.simplefilter("error")   
print('res ',result[2:3])
lambdas = [res[0] for res in result]
rmsestrain = np.array([res[1:2] for res in result])
rmsestest = np.array([res[2:3] for res in result])
    # ws = np.array( [res[3] for res in result] )
    
    
plt.subplot(2,2,1)
plt.plot(lambdas,rmsestrain,'o-')
plt.ylabel('RMSE Train')
plt.xlabel('$\lambda$')

    
plt.subplot(2,2,2)
plt.plot(lambdas,rmsestest,'o-')
plt.ylabel('RMSE Test')
plt.xlabel('$\lambda$')

plt.show()
