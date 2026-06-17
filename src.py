import torch


X=torch.zeros((16,4))
Y=torch.zeros((16,1))
for i in range((16)):
    binary=f"{i:04b}"
    X[i]=torch.tensor([float(char) for char in binary])


    if torch.equal(X[i],torch.flip(X[i],dims=[0])):
     Y[i]=1.0
    else:
     Y[i]=0.0

w1=torch.randn(4,2)*0.5
b1=torch.randn(1,2)*0.5

w2=torch.randn(2,1)*0.5
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
    epochError=0.0
    indices=torch.randperm(16)

    for i in indices:
        X_i=X[i].unsqueeze(0)
        target=Y[i].unsqueeze(0)

#for front pass
        hiddenInput=sigmoid(torch.matmul(X_i,w1)+b1)
        hiddenOutput=sigmoid(torch.matmul(hiddenInput,w2)+b2)

        epochError+=0.5*torch.sum((hiddenOutput-target)**2).item()

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
        print(f"Epoch {epoch:4d}|Total system error: {epochError:.5f}")

print("\n")
print("verify randomly generated weights:")
print(f"Final weights for Hidden unit1:{w1[:,0].numpy()}")
print(f"Final weights for Hidden unit2:{w1[:,1].numpy()}")
print("They will still be inverse, mirror-image of each another.")