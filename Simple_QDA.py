from scipy.io import arff
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

def discQDA(X, standardize, mu, Sigma, prior):
    Xc = standardize(X) - mu
    #Sigma = np.asarray(Sigma).reshape((-1,1))
    print('shp ',Sigma)
    if Sigma.size == 1:
        Sigma = np.asarray(Sigma).reshape((1,1))
    det = np.linalg.det(Sigma)        
    if det == 0:
        raise np.linalg.LinAlgError('discQDA(): Singular covariance matrix')
    SigmaInv = np.linalg.inv(Sigma)     # pinv in case Sigma is singular
    
    #XX = -0.5 * np.log(det) 
    #YY = - 0.5 * np.sum(np.dot(Xc,SigmaInv) * Xc, axis=1)
    print('shp 1 ',Xc.shape)
    print('shp 2 ',(np.log(prior)).shape)
    print('shp 3 ',SigmaInv.shape)
    return -0.5 * np.log(det) \
           - 0.5 * np.sum(np.dot(Xc,SigmaInv) * Xc, axis=1) \
           + np.log(prior)
           
def makeQDA(X,T):
    standardize,_ = makeStandardize(X)
    Xs = standardize(X)

    class1rows,_ = np.where(T == 1)
    print('class 1 ',class1rows.shape)
    #print('X shape ',X.shape)
    print('T shape ',T.shape)
    class2rows,_ = np.where(T == 2)
    class3rows,_ = np.where(T == 3)

    mu1 = np.mean(Xs[class1rows,:],axis=0)
    mu2 = np.mean(Xs[class2rows,:],axis=0)
    mu3 = np.mean(Xs[class3rows,:],axis=0)

    Sigma1 = np.cov(Xs[class1rows,:].T)
    Sigma2 = np.cov(Xs[class2rows,:].T)
    Sigma3 = np.cov(Xs[class3rows,:].T)

    N1 = class1rows.shape[0]
    N2 = class2rows.shape[0]
    N3 = class3rows.shape[0]
    N = len(T)
    #print('N s ',N1,N2,N3,N)
    prior1 = N1 / float(N)
    prior2 = N2 / float(N)
    prior3 = N3 / float(N)

    mu = np.vstack((mu1, mu2, mu3))
    sigma = np.vstack((Sigma1, Sigma2, Sigma3))
    prior = np.hstack((prior1, prior2, prior3))
    
    print('mu 1 ',mu1)
    print('mu 0',mu[0,:])
    print('mu ',mu)
    print('sigma 1 ',Sigma1)
    print('sigma 0',sigma[0:2,:])
    print('sigma ',sigma)
    #print('sigma ',sigma)
    print('prior ',prior)
    print('prior 0 ',prior[0])
    return (mu, sigma, prior)

def makeIndicatorVars(T):
    """ Assumes argument is N x 1, N samples each being integer class label """
    return (T == np.unique(T)).astype(int)

def makeQDA_real(X,T):
    standardize,_ = makeStandardize(X)
    Xs = standardize(X)

    diff_classes = makeIndicatorVars(T)
    
    print('classes ',diff_classes)
    #return (mu, sigma, prior)

def useQDA(model,X):
    mu, sigma, prior =  model
    itr_indx = mu.shape[0]
    (standardizeF, unstandardizeF) = makeStandardize(X)
    discriminantvalues = []
    
    #print('prior i ',np.array([[prior[0],prior[1],prior[2]]]))
    #print('prior ',prior)
    for i in range(itr_indx):
        #print('sgma ',sigma[i,:])
        discrimntval = discQDA(X,standardizeF,mu[i,:],sigma[2*i:2*i+2,:],prior[i])
        discriminantvalues.append(discrimntval)

    discriminantvalues = np.array(discriminantvalues)   
    classprobs = np.exp(discriminantvalues.T - 0.5*D*np.log(2*np.pi) - np.log(np.array([[prior[0],prior[1],prior[2]]])))
    predclasses = np.argmax(discriminantvalues,axis=0) 
    #prob_x = classprobs * np.array([[prior[0],prior[1],prior[2]]])
    #print('disc values ',discriminantvalues)
    #print('shape ',np.array(prob_x).T.shape)
    #print('class probs shape ',classprobs.shape)
    #print('prior shape ',prior.shape)
    return (predclasses, classprobs, discriminantvalues)


D = 2  # number of components in each sample
N = 10  # number of samples in each class
X1 = np.random.normal(1.0,0.1,(N,D))
T1 = np.array([1]*N).reshape((N,1))
X2 = np.random.normal(2.0,0.1,(N,D))  
T2 = np.array([2]*N).reshape((N,1))
X3 = np.random.normal(3.0,0.1,(N,D)) 
T3 = np.array([3]*N).reshape((N,1))

data = np.hstack((np.vstack((X1,X2,X3)), np.vstack((T1,T2,T3))))
X = data[:,0:D]
T = data[:,-1].reshape(-1,1)
#print('data ',data)
#print('X ',X)
#print('T ',T)
nrows = X.shape[0]
nTrain = int(round(nrows*0.8))
nTest = nrows - nTrain
rows = np.arange(nrows)
np.random.shuffle(rows)
trainIndices = rows[:nTrain]
testIndices = rows[nTrain:]
#print('trn indices ',trainIndices.shape, 'rows ',nrows)
Xtrain = X[trainIndices,:]
Ttrain = T[trainIndices,:]
Xtest = X[testIndices,:]
Ttest = T[testIndices,:]

xtst = np.linspace(0,4,100).repeat(D).reshape((100,D))

qda_model = makeQDA(Xtrain,Ttrain)
mu,sigma,prior = qda_model
"""
print('------')
#print('mu 1 ',mu1)
print('mu 0',mu[0,:])
print('mu ',mu)
#print('sigma 1 ',Sigma1)
print('sigma 0',sigma[0,:])
print('sigma ',sigma)
    #print('sigma ',sigma)
print('prior ',prior)
print('prior 0 ',prior[0])
"""
predictedClass,classProbabilities,discriminantValues = useQDA(qda_model,Xtrain)

predictedClasstst,classProbabilitiestst,discriminantValuestst = useQDA(qda_model,Xtest)

predictedClasstest,classProbabilitiestest,discriminantValuestest = useQDA(qda_model,xtst)

def percentCorrect(p,t):
    return np.sum(p.ravel()==t.ravel()) / float(len(t)) * 100   

print('Percent correct: Train',percentCorrect(predictedClass,Ttrain),'Test',percentCorrect(predictedClasstest,xtst))
#print('probs ',probalts.shape)
#print('xtest shape ',xtst.shape)
#print('class probs ',classProbabilitiestest[:,0].shape)
#print('disc val shape ',discriminantValuestest.shape)

(standardizeF, unstandardizeF) = makeStandardize(X)
xtsts = standardizeF(xtst)

prob_class_1 = (normald(xtsts,mu[0,:],sigma[0:2,:]))
prob_class_2 = (normald(xtsts,mu[1,:],sigma[2:4,:]))
prob_class_3 = (normald(xtsts,mu[2,:],sigma[4:6,:]))
px = ((prob_class_1 * (prior[0]))+ (prob_class_2 * (prior[1])) + (prob_class_3 * (prior[2])))

probs = np.hstack(((prob_class_1/px),(prob_class_2/px),(prob_class_3/px)))

print('shape ',probs.shape)
print('prec class ', predictedClass+1.0)


plt.figure(figsize=(20,20))
plt.subplots_adjust(hspace=.6)
plt.subplot(6,1,1)
plt.ylim(0,4)
plt.plot(Xtrain,Ttrain, 'o')
plt.xlabel('Train Data set')
plt.ylabel('Class')

plt.subplot(6,1,2)
plt.plot(xtst[:,0],px)
#plt.plot(xtst[:,0], probalts,'b')
plt.xlabel('Test Data set')
plt.ylabel('P(X)')

plt.subplot(6,1,3)
plt.plot(xtst[:,0], classProbabilitiestest)
plt.xlabel('Test Data set')
plt.ylabel(" P(x/Class=k)", multialignment="center")

plt.subplot(6,1,4)
plt.plot(xtst, discriminantValuestest.T)
plt.xlabel('Test Data set')
plt.ylabel("Discriminent Values", multialignment="center")

plt.subplot(6,1,5)
plt.plot(xtst, predictedClasstest+1.0,'o-')
plt.ylim(0,4)
plt.xlabel('Test Data set')
plt.ylabel("Predicted Class", multialignment="center")

plt.subplot(6,1,6)
plt.plot(xtst[:,0],probs)
plt.ylim(-1,4)
#plt.plot(xtst, (classProbabilitiestest[:,0].reshape(100,1) * np.array([[prior[0]]])/probalts[:,0].reshape(100,1)))
#plt.plot(xtst, (classProbabilitiestest[:,1].reshape(100,1) * np.array([[prior[1]]])/probalts[:,1].reshape(100,1)))
#plt.plot(xtst, (classProbabilitiestest[:,2].reshape(100,1) * np.array([[prior[2]]])/probalts[:,2].reshape(100,1)))
#plt.ylim(-1,1)
plt.xlabel('Test Data set')
plt.ylabel("P(Class=k/x)", multialignment="center")

plt.show()