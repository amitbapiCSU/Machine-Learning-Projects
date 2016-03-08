import pandas
import numpy as np
import random as rn
import matplotlib.pyplot as plt
import warnings


def makeStandardize(X):
    means = X.mean(axis=0)
    stds = X.std(axis=0)
    def standardize(origX):
        return (origX - means) / stds
    def unStandardize(stdX):
        return stds * stdX + means
    return (standardize, unStandardize)

def normald(X, mu=None, sigma=None):
    """ normald:
       X contains samples, one per row, NxD. 
       mu is mean vector, Dx1.
       sigma is covariance matrix, DxD.  """
    d = X.shape[1]
    if np.any(mu == None):
        mu = np.zeros((d,1))
    if np.any(sigma == None):
        sigma = np.eye(d)
    detSigma = sigma if d == 1 else np.linalg.det(sigma)
    if detSigma == 0:
        raise np.linalg.LinAlgError('normald(): Singular covariance matrix')
    sigmaI = 1.0/sigma if d == 1 else np.linalg.inv(sigma)
    normConstant = 1.0 / np.sqrt((2*np.pi)**d * detSigma)
    diffv = X - mu.T # change column vector mu to be row vector
    return normConstant * np.exp(-0.5 * np.sum(np.dot(diffv, sigmaI) * diffv, axis=1))[:,np.newaxis]


def makeLDA(X,T):
    standardize,_ = makeStandardize(X)
    Xs = standardize(X)

    class1rows,_ = np.where(T == 1)
    class2rows,_ = np.where(T == 2)
    class3rows,_ = np.where(T == 3)

    mu1 = np.mean(Xs[class1rows,:],axis=0)
    mu2 = np.mean(Xs[class2rows,:],axis=0)
    mu3 = np.mean(Xs[class3rows,:],axis=0)

    Sigma1 = np.cov(Xs[class1rows,:].T)
    Sigma2 = np.cov(Xs[class2rows,:].T)
    Sigma3 = np.cov(Xs[class3rows,:].T)

    Sigma = Sigma1 + Sigma2 + Sigma3 #np.average(np.vstack((Sigma1, Sigma2, Sigma3)), axis = 0)
    print('sgm ',Sigma)
    N1 = class1rows.shape[0]
    N2 = class2rows.shape[0]
    N3 = class3rows.shape[0]
    N = len(T)
    prior1 = N1 / float(N)
    prior2 = N2 / float(N)
    prior3 = N3 / float(N)

    mu = np.vstack((mu1, mu2, mu3))
    sigma = np.sum(np.vstack((Sigma1 * prior1, Sigma2 * prior2, Sigma3 * prior3)), axis = 0) #np.array(Sigma * np.eye(X.shape[1]))
    prior = np.hstack((prior1, prior2, prior3))
    
    print('mu1 ',mu1)
    print('m ',mu[0,:])
    print('mu ',mu)
    print('sigma ',sigma)
    print('prior ',prior)
    return (mu, Sigma, prior)

def discLDA(X, standardize, mu, Sigma, prior):
    Xc = standardize(X)
    if mu.size == 1:
        mu = np.asarray(mu).reshape((1,1))
    if Sigma.size == 1:
        Sigma = np.asarray(Sigma).reshape((1,1))
    det = np.linalg.det(Sigma)        
    if det == 0:
        raise np.linalg.LinAlgError('discQDA(): Singular covariance matrix')
    SigmaInv = np.linalg.inv(Sigma)     # pinv in case Sigma is singular
    
    """
    return -0.5 * np.log(det) \
           - 0.5 * np.sum(np.dot(Xc,SigmaInv) * Xc, axis=1) \
           + np.log(prior)
    """
    print('X shp ',Xc.shape)
    print('sgma shape ',SigmaInv.shape)
    print('sum 1 ',np.dot(Xc,SigmaInv).shape)
    print('mu shape ',mu.T.shape)
    KK = np.dot(mu.T,SigmaInv) * mu
    print('mmkk ',KK)
    print('sum 2 ',np.sum((np.dot(mu.T,SigmaInv) * mu).reshape((2,1)), axis=1))
    
    return  np.sum((np.dot(Xc,SigmaInv) * mu), axis = 1) - 0.5 * np.sum(np.dot(mu.T,SigmaInv) * mu, axis = 1) + np.log(prior)
    #return np.dot(np.dot(Xc , SigmaInv) , mu) - 0.5 * np.dot (np.dot (mu.T, SigmaInv) , mu) + np.log(prior)

def useLDA(model,X):
    mu, sigma, prior =  model
    itr_indx = mu.shape[0]
    (standardizeF, unstandardizeF) = makeStandardize(X)
    discriminantvalues = []
    
    print('prior i ',np.array([[prior[0],prior[1],prior[2]]]))
    print('prior ',prior)
    print('sgma ',sigma)
    """
    for i in range(itr_indx):
        discrimntval = discLDA(X,standardizeF,mu[i,0],sigma,prior[i])
        discriminantvalues.append(discrimntval)
    """
    d1 =  discLDA(X,standardizeF,mu[0,:],sigma,prior[0])
    d2 =  discLDA(X,standardizeF,mu[1,:],sigma,prior[1])
    d3 =  discLDA(X,standardizeF,mu[2,:],sigma,prior[2])
    #discriminantvalues = np.array(discriminantvalues) 
    discriminantvalues = np.vstack((d1,d2,d3)).T
    print('disc values ',discriminantvalues.shape)  # [0,:,:] 
    #classprobs = np.exp(discriminantvalues.T - 0.5*D*np.log(2*np.pi) - np.log(np.array([[prior[0],prior[1],prior[2]]])))
    classprobs = np.exp(np.vstack((d1,d2,d3)).T - 0.5*D*np.log(2*np.pi) - np.log(np.array([[prior[0],prior[1],prior[2]]])))
    #predclasses = np.argmax(discriminantvalues,axis=0) 
    predclasses = np.argmax(np.vstack((d1,d2,d3)),axis=0) 
    prob_x = classprobs * np.array([[prior[0],prior[1],prior[2]]])
    print('shape ',classprobs.shape)
    return (predclasses, classprobs, discriminantvalues, np.array(prob_x))

def percentCorrect(p,t):
    return np.sum(p.ravel()==t.ravel()) / float(len(t)) * 100   

D = 2  # number of components in each sample
N = 10  # number of samples in each class
X1 = np.random.normal(1.0,0.1,(N,D))
T1 = np.array([1]*N).reshape((N,1))
X2 = np.random.normal(2.0,0.1,(N,D))  
T2 = np.array([2]*N).reshape((N,1))
X3 = np.random.normal(3.0,0.1,(N,D)) 
T3 = np.array([3]*N).reshape((N,1))

data = np.hstack(( np.vstack((X1,X2,X3)), np.vstack((T1,T2,T3))))
X = data[:,0:D]
T = data[:,-1].reshape(-1,1)
print('data ',data)
print('X ',X)
print('T ',T)
nrows = X.shape[0]
nTrain = int(round(nrows*0.8))
nTest = nrows - nTrain
rows = np.arange(nrows)
np.random.shuffle(rows)
trainIndices = rows[:nTrain]
testIndices = rows[nTrain:]
print('trn indices ',trainIndices.shape, 'rows ',nrows)
Xtrain = X[trainIndices,:]
Ttrain = T[trainIndices,:]
Xtest = X[testIndices,:]
Ttest = T[testIndices,:]

xtst = np.linspace(0,4,100).repeat(D).reshape((100,D))


lda_model = makeLDA(Xtrain,Ttrain)

mu, sigma, prior = lda_model
(standardizeF, unstandardizeF) = makeStandardize(X)
xtsts = standardizeF(xtst)
predictedClass,classProbabilities,discriminantValues,_ = useLDA(lda_model,Xtrain)

predictedClasstst,classProbabilitiestst,discriminantValuestst,_ = useLDA(lda_model,Xtest)

predictedClasstest,classProbabilitiestest,discriminantValuestest, probalts = useLDA(lda_model,xtsts)

print('xtst shape ',xtst.shape)
print('prob shape ',classProbabilitiestest.shape)


prob_class_1 = (normald(xtsts,mu[0,0],sigma))
prob_class_2 = (normald(xtsts,mu[1,0],sigma))
prob_class_3 = (normald(xtsts,mu[2,0],sigma))
px = ((prob_class_1 * (prior[0]))+ (prob_class_2 * (prior[1])) + (prob_class_3 * (prior[2])))

probs = np.hstack(((prob_class_1/px),(prob_class_2/px),(prob_class_3/px)))

probs_cls = np.hstack((prob_class_1, prob_class_2, prob_class_3))

print('Percent correct: Train',percentCorrect(predictedClass,Ttrain),'Test',percentCorrect(predictedClasstest,xtst))
      
plt.figure(figsize=(20,20))
plt.subplots_adjust(hspace=.6)
plt.subplot(6,1,1)
plt.ylim(0,4)
plt.plot(X,T,'o')
#plt.plot(xtst[:,0], classProbabilitiestest)
# plt.plot(Xtrain,predictedClass+1.0, 'o')
plt.xlabel('Train Data set')
plt.ylabel(' Class')

plt.subplot(6,1,2)
plt.plot(xtst[:,0], px)
#plt.ylim(-1,10)
plt.xlabel('Test Data set')
plt.ylabel('P(X)')

plt.subplot(6,1,3)
plt.plot(xtst[:,0], probs_cls)
#plt.ylim(-1,10)
plt.xlabel('Test Data set')
plt.ylabel(" P(x/Class=k)", multialignment="center")

plt.subplot(6,1,4)
plt.plot(xtst[:,0], discriminantValuestest)  # [0,:,:]
plt.xlabel('Test Data set')
plt.ylabel("Discriminent Values", multialignment="center")

plt.subplot(6,1,5)
plt.plot(xtst, predictedClasstest+1.0,'o-')
plt.ylim(0,4)
plt.xlabel('Test Data set')
plt.ylabel("Predicted Class", multialignment="center")

plt.subplot(6,1,6)
plt.plot(xtst[:,0], probs)
plt.ylim(-1,4)
plt.xlabel('Test Data set')
plt.ylabel("P(Class=k/x)", multialignment="center")

plt.show()
