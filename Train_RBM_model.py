import numpy as np
import argparse
import csv
import os.path
import argparse
#import tsne

rng=np.random.RandomState(123)
train_file="train.csv"
val_file="val.csv"
test_file="test.csv"
#reading and normalising training,validation and test data
from numpy import genfromtxt
data1 = genfromtxt(train_file, delimiter=',')
trainData=data1[1:55001,1:785]
actualLabel_train=data1[1:55001,785]
trainData = trainData.astype(np.float32)
actualLabel_train=actualLabel_train.astype(int)

data2 = genfromtxt(val_file, delimiter=',')
valData=data2[1:5001,1:785]
actualLabel_val=data2[1:5001,785]
valData = valData.astype(np.float32)
actualLabel_val=actualLabel_val.astype(int)

train_data=np.append(trainData,valData,axis=0)   #saved in downloads as train_data



print("Data loaded")
for i in range(len(train_data)):
    for j in range(784):
        if(train_data[i,j]>=127):
            train_data[i,j]=1
        else:
            train_data[i,j]=0               	

data1 = genfromtxt(test_file, delimiter=',')
testData=data1[1:10001,1:785]
testData = testData.astype(np.float32)

for i in range(len(testData)):
    for j in range(784):
        if(testData[i,j]>=127):
            testData[i,j]=1
        else:
            testData[i,j]=0                 	
       	 

print("Visible layers binary")

def sig_act(x):						#sigmoid activation function
	return 1.0/(1.0+np.exp(-x))

 #sigmoid activation function
def sig_new(x):
    y=np.zeros(len(x))
    for i in range(len(x)):
        y[i]=1.0/(1.0+np.exp(-x[i]))
    return y 


def prob_h_given_v(W,b,c,i,v):
    x=np.dot(np.transpose(W[:,i]),v)
    p_h_v=sig_act(x+c[i])
    return p_h_v
    

def prob_v_given_h(W,b,c,i,h):
    x=np.dot(np.transpose(W[i,:]),h)
    p_h_v=sig_act(x+b[i])
    return p_h_v    

def sampling(probs):    
    samples = np.random.uniform(size=probs.shape)
    probs[samples < probs] = 1
    np.floor(probs, probs)
    return probs    

    
def sample_h(W,b,c,v_d):
    prob1=np.zeros(n)
    for i in range(n):
        prob1[i]=prob_h_given_v(W,b,c,i,v_d)
    newh=sampling(prob1)
    return newh 
    #s = np.random.binomial(n, prob1, 10)
    

    
def sample_v(W,b,c,h):
    prob2=np.zeros(784)
    for i in range(784):
        prob2[i]=prob_v_given_h(W,b,c,i,h)
    newv=sampling(prob2)
    return newv



def freeenergy(W,b,c,h,v):
    vb=np.dot(v,b)
    hb=0
    for i in range(n):
        E=x=np.dot(np.transpose(W[:,i]),v)+c[i]        
        hb=hb+np.log(1 + np.exp(E))
    return -vb-hb

def restructed_image1(W,b,c,n,v_sample):
    prob_v=np.zeros(n)
    prob_v1=np.zeros(784)
    h_n=np.zeros(n)
    v_n=np.zeros(784)
    match=np.zeros(5)
    flag=0
    for j in range(5):     
        for i in range(n):
            prob_v[i]=prob_h_given_v(W,b,c,i,v_sample[j][:])
        h_n=np.floor(prob_v)
        for i in range(784):
            prob_v1[i]=prob_v_given_h(W,b,c,i,h_n)
        v_n=np.floor(prob_v)
        match[j]=np.sum(v_n==v_sample[j][:])
        if(match[j]>=700):
            flag=1
            print("j",j)
            j=5000
        else:
            j=j+1               
    return h_n,v_n,flag
      

#For CD run at different k and n
def CD_algo(k,eta,n,ep):
    W=np.asarray(rng.uniform(low=-4*np.sqrt(6./(n+784)),high=4*np.sqrt(6./(n+784)),size=(784,n)))
    b=np.zeros(784)
    c=np.zeros(n)
    h=np.zeros(n)
    for j in range(ep): 
        cost=0   
        for d in range(len(train_data)):
            v_d=train_data[d,:]
            v_t=v_d
            for t in range(k):        
                h=sample_h(W,b,c,v_t)
                v_t=sample_v(W,b,c,h)   
            a1=np.dot(np.transpose(W),v_d)+c
            b1=np.dot(np.transpose(W),v_t)+c
            W=W+eta*(np.multiply(v_d.reshape(784,1),sig_new(a1).reshape(1,n))-np.multiply(v_t.reshape(784,1),sig_new(b1).reshape(1,n)))  
    #v_t size check
            b=b+eta*(v_d-v_t)
            c=c+eta*(sig_new(a1)-sig_new(b1))
            cost=cost+freeenergy(W,b,c,h,v_t)
        cost1=cost/len(train_data)
        print("epoch", j,"cost",cost1)
    return W,b,c,h

    
#For ques3    
rng = np.random.RandomState(123)   
def CD_algo(k,eta,n,ep):
    W=np.asarray(rng.uniform(low=-4*np.sqrt(6./(n+784)),high=4*np.sqrt(6./(n+784)),size=(784,n)))
    b=np.zeros(784)
    c=np.zeros(n)
    h=np.zeros(n)
    new_vis=[]
    v_n_big=[]
    up=0 
    for j in range(ep):
        cost=0   
        for d in range(len(train_data)):
            v_d=train_data[d,:]
            v_t=v_d
            for t in range(k):        
                h=sample_h(W,b,c,v_t,n)
                v_t=sample_v(W,b,c,h,n)   
            a1=np.dot(np.transpose(W),v_d)+c
            b1=np.dot(np.transpose(W),v_t)+c
            W=W+eta*(np.multiply(v_d.reshape(784,1),sig_new(a1).reshape(1,n))-np.multiply(v_t.reshape(784,1),sig_new(b1).reshape(1,n)))  
    #v_t size check
            b=b+eta*(v_d-v_t)
            c=c+eta*(sig_new(a1)-sig_new(b1))
            cost=cost+freeenergy(W,b,c,h,v_t,n)
            up=up+1
            if(np.mod(up,15000)==0):
                [new_v,v_n_o]=restructed_image(W,b,c,n)
                new_vis.append(new_v)
                v_n_big.append(v_n_o) 
        cost1=cost/len(train_data)
        print("epoch", j,"cost",cost1)
    return W,b,c,h,new_vis,v_n_big    
    
    
    
    
    
    

 
#For Normal run            
def Gibbs_algo(k,eta,n,r,ep):
    W=np.asarray(rng.uniform(low=-4*np.sqrt(6./(n+784)),high=4*np.sqrt(6./(n+784)),size=(784,n)))
    b=np.random.uniform(-1,1,(784))
    c=np.random.uniform(-1,1,(n))
    h_t=np.zeros(n)
    for j in range(1): 
        cost=0   
        for d in range(len(train_data)):
            v_d=train_data[d,:]
            v_big=[]
            h_big=[]
            v_t=np.random.uniform(-1,1,(784))
            for t in range(k+r):        
                h_t=sample_h(W,b,c,v_t)
                v_t=sample_v(W,b,c,h_t)
                v_big.append(v_t)
                h_big.append(h_t)
            a2=np.dot(np.transpose(W),v_d)+c
            b2=np.zeros([784,n])
            p=np.zeros(784)
            q=np.zeros(n)
            for t in range(k,k+r):
                temp=v_big[t]
                b1=np.dot(np.transpose(W),temp)+c
                b2=b2+np.multiply(v_t.reshape(784,1),sig_new(b1).reshape(1,n))
                p=p+v_big[t]
                q=q+sig_new(b1)           
            W=W+eta*(np.multiply(v_d.reshape(784,1),sig_new(a2).reshape(1,n))-b2/r)  
            b=b+eta*(v_d-p/r)
            c=c+eta*(sig_new(a2)-q/r)
            #cost=cost+freeenergy(W,b,c,h_t,v_t)
        #cost1=cost/len(train_data)
        print("epoch", j)
    return W,b,c,h_t   

    
 ###To find out r at which sample match is 700   
def Gibbs_algo1(W1,b1,c1,k,eta,n,r,ep):
    v_sample=[train_data[0,:],train_data[5000,:],train_data[10000,:],train_data[15000,:],train_data[20000,:]]
    W=W1
    b=b1
    c=c1
    h_t=np.zeros(n)    
    update=0
    for j in range(ep):
        for d in range(len(train_data)):
            print("d",d)
            v_d=train_data[d,:]
            v_big=[]
            h_big=[]
            v_t=np.random.uniform(-1,1,(784))
            for r in range(1,100):
                for t in range(k+r):        
                    h_t=sample_h(W,b,c,v_t)
                    v_t=sample_v(W,b,c,h_t)
                    v_big.append(v_t)
                    h_big.append(h_t)
                    a2=np.dot(np.transpose(W),v_d)+c
                b2=np.zeros([784,n])
                p=np.zeros(784)
                q=np.zeros(n)
                for t in range(k,k+r):
                    temp=v_big[t]
                    b1=np.dot(np.transpose(W),temp)+c
                    b2=b2+np.multiply(v_t.reshape(784,1),sig_new(b1).reshape(1,n))
                    p=p+v_big[t]
                    q=q+sig_new(b1)           
                W1=W+eta*(np.multiply(v_d.reshape(784,1),sig_new(a2).reshape(1,n))-b2/r)  
                b1=b+eta*(v_d-p/r)
                c1=c+eta*(sig_new(a2)-q/r)
                [h_n,v_n,flag]=restructed_image1(W1,b1,c1,n,v_sample)
                
                if(flag==1 or r==999):
                    print("r",r)
                    W=W1
                    b=b1
                    c=c1
                    break;
    print("epochs", j)
    return W,b,c,h_t,r       
    
parser=argparse.ArgumentParser()
parser.add_argument('--lr', type=float)		
parser.add_argument('--epochs', type=int)	
parser.add_argument('--hidden_nodes', type=int)
parser.add_argument('--K', type=int)
parser.add_argument('--r', type=int)
parser.add_argument('--method', type=int)
args = parser.parse_args()

eta=args.lr
ep=args.epochs
n=args.hidden_nodes
k=args.K                                       
r=args.r
m=args.method

if(m==1):
    [Wx,bx,cx,hx]=CD_algo(k,eta,n,ep)
    #np.save("Hcd.npy",hx)
else:
    [Wx,bx,cx,hx]=Gibbs_algo(k,eta,n,r,ep)
    [W,b,c,h_t,r]=Gibbs_algo1(Wx,bx,cx,k,eta,n,r,ep)

    #np.save("Hig.npy",hy)
	
	
h_test=np.zeros([10000,len(hx)])
for d in range(len(testData)):
    v_test=testData[d,:]
    prob_test=np.zeros(n)
    for i in range(n):
        prob_test[i]=prob_h_given_v(Wx,bx,cx,i,v_test)
        if(prob_test[i]>0.5):
            h_test[d,i]=1
        else:
            h_test[d,i]=0

np.save("Hig.npy",h_test)

    
    
    

    
    















