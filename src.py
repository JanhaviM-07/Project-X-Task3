import torch

torch.manual_seed(42)
X=torch.zeros((256,8))
Y=torch.zeros((256,1))

for i in range((256)):
    binary=f"{i:08b}"
    X[i]=torch.tensor([float(char) for char in binary])


    if torch.equal(X[i],torch.flip(X[i],dims=[0])):
     Y[i]=1.0
    else:
     Y[i]=0.0

shuffle=torch.randperm(256)
sizeTrain=int(0.8*256)
train=shuffle[:sizeTrain]
test=shuffle[sizeTrain:]

Xtrain,Ytrain=X[train],Y[train]
Xtest,Ytest=X[test],Y[test]
print(f'Total Train datasets:{sizeTrain}')


w1=torch.randn(8,8)*0.5
b1=torch.randn(1,8)*0.5

w2=torch.randn(8,1)*0.5
b2=torch.randn(1,1)*0.5


v_w1=torch.zeros_like(w1)
v_b1=torch.zeros_like(b1)
v_w2=torch.zeros_like(w2)
v_b2=torch.zeros_like(b2)

lr=0.25
alpha=0.9
epochs=8000

def sigmoid(X):
    return 1.0/(1.0+torch.exp(-X))

print("Generating Random weights")
print("\n")

for epoch in range(1,epochs+1):
    trainError=0.0
    indices=torch.randperm(sizeTrain)

    for i in indices:
        X_i=X[i].unsqueeze(0)
        target=Y[i].unsqueeze(0)

#for front pass
        hiddenInput=sigmoid(torch.matmul(X_i,w1)+b1)
        hiddenOutput=sigmoid(torch.matmul(hiddenInput,w2)+b2)

        trainError+=0.5*torch.sum((hiddenOutput-target)**2).item()

#now for backward pass
        delOutput=(hiddenOutput-target)*(hiddenOutput*(1.0-hiddenOutput))
        delHidden=torch.matmul(delOutput,w2.t())*(hiddenInput*(1.0-hiddenInput))

        v_w2=(-lr*torch.matmul(hiddenInput.t(),delOutput))+(alpha*v_w2)
        v_b2=(-lr*delOutput)+(alpha*v_b2)
        v_w1=(-lr*torch.matmul(X_i.t(),delHidden))+(alpha*v_w1)
        v_b1=(-lr*delHidden)+(alpha*v_b1)


        w2+=v_w2
        b2+=v_b2
        w1+=v_w1
        b1+=v_b1

    if epoch==1 or epoch%2000==0:
        print(f"Epoch {epoch:5d}|Total system error: {trainError:.5f}")

print("\n")
print("Evaluating Model against Test")

testError=0.0
correctAns=0.0

with torch.no_grad():
   for i in range(len(Xtest)):
      Xi=Xtest[i].unsqueeze(0)
      target=Ytest[i].item()

      hiddenInput=sigmoid(torch.matmul(Xi,w1)+b1)
      hiddenOutput=sigmoid(torch.matmul(hiddenInput,w2)+b2)

      binaryAns=1.0 if hiddenOutput>=0.5 else 0.0
      if binaryAns==target:
         correctAns+=1
      testError+=0.5*(hiddenOutput-target)**2  

accuracy=(correctAns/len(Xtest))*100

print(f"Total Test Error:{testError}")
