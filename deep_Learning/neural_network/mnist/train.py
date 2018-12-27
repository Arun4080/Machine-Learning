import pandas as pd
import numpy as np

train=pd.read_csv("data/train.csv")
train_X=(train.drop(["label"],axis=1)/255).values
train_Y=pd.get_dummies([str(i) for i in train.label]).values
del(train)

class Neural_Network(object):
  def __init__(self):
  #parameters
    self.inputSize = 784
    self.hl1Size = 20
    self.hl2Size = 20
    self.outputSize = 10

  #weights
    self.W1 = np.random.randn(self.inputSize, self.hl1Size) 
    self.W2 = np.random.randn(self.hl1Size,self.hl2Size) 
    self.W3 = np.random.randn(self.hl2Size, self.outputSize)
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

    self.W1 += X.reshape(self.inputSize,1).dot(self.z2_delta.reshape(1,self.hl1Size)) # adjusting first set (input --> hidden) weights
    self.W2 += self.hl1.T.dot(self.z3_delta) # adjusting second set (hidden --> output) weights
    self.W3 += self.hl2.reshape(self.hl2Size,1).dot(self.o_delta.reshape(1,self.outputSize)) # adjusting second set (hidden --> output) weights

  def train(self, X, y, n=1,batch=50):
    min,hw1,hw2,hw3,hb1,hb2,hb3=1,0,0,0,0,0,0
    print("Before Training loss: "+str(np.mean(np.square(y - self.forward(X)))))
    for j in range(n):
        for k in range(0,len(X),batch):
            avg_loss=[]
            for i in range(k,k+batch):
                o = self.forward(X[i])
                self.backward(X[i], y[i], o)
                l=np.mean(np.square(y - self.forward(X)))
                avg_loss.append(l)
                if min>l:
                    min=l
                    hw1,hb1=self.W1,self.b1
                    hw2,hb2=self.W2,self.b2
                    hw3,hb3=self.W3,self.b3
            print("avg loss of this batch is " + str(np.mean(avg_loss)))
              
    print("min loss: "+str(min))
    with open("output.txt",'w') as f:
        f.write("w1: \n"+str(hw1)+"\n w2: \n"+str(hw2)+"\n w3: \n"+str(hw3)+"\n b1: \n"+str(hb1)+"\n b2: \n"+str(hb2)+"\n b3: \n"+str(hb3)+"\n")
    self.W1,self.b1=hw1,hb1
    self.W2,self.b2=hw2,hb2
    self.W3,self.b3=hw3,hb3
    print("After Training loss: "+str(np.mean(np.square(y - self.forward(X)))))
    

  def predict(self,X):
    return self.forward(X).round()

NN=Neural_Network()
NN.train(train_X,train_Y,batch=1000)
