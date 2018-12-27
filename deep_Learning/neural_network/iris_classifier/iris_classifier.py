import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#gather data
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
#cat={'Iris-setosa':0,'Iris-versicolor':1,'Iris-virginica':2}
a=pd.read_csv("data/iris.csv", names=names)
a = shuffle(a)
#a['class']=[cat[i] for i in a['class']]
dummies=pd.get_dummies(a)

array=dummies.values
x=array[:,:4]
y=array[:,4:]
validation_size=0.05
seed=7
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=validation_size, random_state=seed)

#Class of ANN
class Neural_Network(object):
  def __init__(self):
  #parameters
    self.inputSize = 4
    self.hl1Size = 10
    self.hl2Size = 10
    self.outputSize = 3

  #weights
    self.W1 = np.random.randn(self.inputSize, self.hl1Size) # (4*10) weight matrix from input to hidden layer 1
    self.W2 = np.random.randn(self.hl1Size,self.hl2Size) #(10*10) weight matrix for hidden layer 1 to hidden layer 2
    self.W3 = np.random.randn(self.hl2Size, self.outputSize) # (3x1) weight matrix from hidden layer 2 to output layer
    self.b1=np.random.randn(self.hl1Size)
    self.b2=np.random.randn(self.hl2Size)
    self.b3=np.random.randn(self.outputSize)

  def forward(self, X):
    #forward propagation through our network
    self.z = np.dot(X, self.W1)+self.b1 # dot product of X (input) and first set of weights and adding bias
    self.hl1 = self.sigmoid(self.z) # activation function
    
    self.z2 = np.dot(self.hl1, self.W2)+self.b2 # dot product of hidden layer1 and second set of weights and adding bias
    self.hl2 = self.sigmoid(self.z2) # activation function
    
    self.z3 = np.dot(self.hl2, self.W3)+self.b3 # dot product of hidden layer2 and third set of weights and adding bias
    o = self.sigmoid(self.z3) # final activation function
    
    return o

  def sigmoid(self, s):
    # activation function
    return 1/(1+np.exp(-s))

  def sigmoidPrime(self, s):
    #derivative of sigmoid
    return s * (1 - s)

  def backward(self, X, y, o):
    # backward propagate through the network
    self.o_error = y - o # error in output
    self.o_delta = self.o_error*self.sigmoidPrime(o)

    self.z3_error = self.o_delta.dot(self.W3.T)
    self.z3_delta = self.z3_error*self.sigmoidPrime(self.hl2) 
    
    self.z2_error = self.z3_delta.dot(self.W2.T) 
    self.z2_delta = self.z2_error*self.sigmoidPrime(self.hl1) 

    self.W1 += X.reshape(4,1).dot(self.z2_delta.reshape(1,10)) # adjusting first set (input --> hidden) weights
    self.W2 += self.hl1.T.dot(self.z3_delta) # adjusting second set (hidden --> output) weights
    self.W3 += self.hl2.reshape(10,1).dot(self.o_delta.reshape(1,3)) # adjusting second set (hidden --> output) weights

  def train(self, X, y):
    min,hw1,hw2,hw3,hb1,hb2,hb3=1,0,0,0,0,0,0
    print("Before Training loss: "+str(np.mean(np.square(y - self.forward(X)))))
    for j in range(50):
        avg_loss=[]
        for i in range(len(X)):
            o = self.forward(X[i])
            self.backward(X[i], y[i], o)
            l=np.mean(np.square(y - self.forward(X)))
            avg_loss.append(l)
            if min>l:
                min=l
                hw1,hb1=self.W1,self.b1
                hw2,hb2=self.W2,self.b2
                hw3,hb3=self.W3,self.b3
        print("avg loss of "+str(j)+" loop is " + str(np.mean(avg_loss)))
              
    print("min loss: "+str(min))
    self.W1,self.b1=hw1,hb1
    self.W2,self.b2=hw2,hb2
    self.W3,self.b3=hw3,hb3
    print("After Training loss: "+str(np.mean(np.square(y - self.forward(X)))))
    

  def predict(self,X):
    return self.forward(X).round()


#Start ANN
NN = Neural_Network()
NN.train(X_train,Y_train)
predictions=NN.predict(X_test)
print("accuracy is "+str(accuracy_score(Y_test, predictions)))


'''
[[-1.21284303  0.7653366   5.77585742 -0.16053269  4.0495684  -0.13457282
   0.99892737  0.62621542  0.15294304  0.35647138]
 [ 0.22419831  1.06739433  3.43644224 -1.36918124  3.58169431  0.31933846
   1.84551772  0.92438713 -0.78019575  2.49779688]
 [ 0.23338267  1.15804424 -7.44372746  0.2801827  -5.94053238 -1.51788186
  -1.92999582  0.65553977 -1.01559269 -3.21672418]
 [ 0.27253057 -0.28118594 -6.45123228 -0.79563445 -5.57933593 -0.66324054
  -2.01112581 -0.84070067 -1.50466992 -1.28277637]]

 [[-0.12145995  0.43023916 -1.76487375 -1.30988552  0.35545277 -0.08423902
  -0.48759456 -0.70791022  0.61053111  0.15946695]
 [ 0.85686799  0.98019868 -0.15986791  1.31468138  1.12615483  1.77639523
  -0.36546245 -0.63232247 -1.94200585 -0.67142239]
 [ 2.42298461 -0.04982327  0.13140764 -1.26759248  0.5951415  -1.32835331
  -0.44188917  1.46060251  0.4284857  -0.64956266]
 [-0.16036707  1.77427082 -0.62006542 -1.36715649  1.65575083  1.0428224
  -0.00600549  0.96001744 -0.04255657  0.82845573]
 [-0.43866881 -0.45401518 -0.19852405  0.65167426 -1.65456549  1.94501248
   1.80882052  0.3590564  -0.65666607  0.22809715]
 [-3.05851298  0.39010674  0.44704053 -0.30589431  1.00085876 -1.32219597
   0.15678434  0.71188144 -1.54847903 -0.36358935]
 [-1.62921365 -1.29691963  0.03843595 -0.32887826 -0.77545667 -0.57848129
  -0.06742884  0.21540451 -1.5283491   0.31886378]
 [-0.7791316   0.08527688 -0.94335229  0.67297575  0.03818223  0.29921923
   0.40383194 -0.83946665 -0.08612235  0.10733067]
 [-1.43390965  0.92119682  1.07619973  1.10749043  1.10573215 -0.80618357
  -1.62470427 -0.58065227 -0.99688302  0.03447581]
 [-0.08536634 -2.38842714  1.71726908 -0.44836812 -1.50715846 -0.81202011
  -0.33887174  0.2578967  -0.58364623  1.41422457]]

 [[-0.98962562  1.78566209 -1.36536307]
 [-4.72754436  0.23307942  2.54865515]
 [ 2.34955814 -6.44013275 -2.28222553]
 [-3.09828865 -4.42364947  3.71426704]
 [-3.02266528 -1.30790778  2.75271302]
 [-2.00326775  1.67901532 -0.90478127]
 [ 0.2048567   3.26744501 -3.50565646]
 [ 1.77941211  2.03026509 -4.43636393]
 [-0.85256819 -2.35921198  1.40241504]
 [ 2.67825694 -4.10523683  0.55030531]]

 [-0.71401113  0.79780284  1.6363363   0.57391267  3.02603411  2.77615793
  0.15069836 -0.31333038 -0.26293425 -0.20763434] 

 [-0.68968664  0.14034037  0.47739356 -1.99819019  0.00437731  0.26739473
  0.22633582 -1.23699613  0.82862533  0.0026157 ] 

 [1.94044302 1.44748941 0.09994495]
'''