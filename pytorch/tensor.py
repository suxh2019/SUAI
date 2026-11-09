import torch
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Tensor oprations
x = torch.empty(2,2)
y = torch.rand(2,2)
z = torch.zeros(2,2)
z1 = torch.ones(2,2, dtype = torch.int)
x1 = torch.tensor([2.5, 0.1])
z2 = x * y
y.add_(x) # _: in placement
z3 = torch.sub(x,y)
x = torch.rand(4,4)
#print(x[:,0]) # row, col
#print(x[1,1].item())
y = x.view(16)
#print(y)
y = x.view(-1, 8) # torch.size([2,8])

a = torch.ones(5) # if a and b are on cpu, change one value and affect another one
b = a.numpy()
# print(b)

a = np.ones(5)
b = torch.from_numpy(a)

if torch.cuda.is_available():
    device = torch.device("cuda")
    x = torch.ones(5, device = device)
    y = torch.ones(5)
    y = y.to(device)
    z = x + y
    z =z.to("cpu")
x = torch.ones(5, requires_grad=True) # calculate the gradient for this tensor
#--------------------------------------------------------
# Gradient calculation with Autograd
x = torch.rand(3, requires_grad = True)
y = x + 2
z = y*y*2
# z = z.mean() or the below
v = torch.tensor([0.1,1.0,0.01],dtype = torch.float)
z.backward(v) # dz/dx
print(x.grad)

# x.requires_grad_(False)
x.requires_grad_(False)
print(x)
#x.detach
#with torch.no_grad():

# toy training example
weights = torch.ones(4, requires_grad= True)
for epoch in range(3):
    model_output = (weights * 4).sum()
    model_output.backward()
    #print(weights.grad)
    weights.grad.zero_() # clear the gradient, otherwise, gradients will be cumulated

weights = torch.ones(4, requires_grad= True)
# Create optimizer
optimizer = torch.optim.SGD([weights], lr=0.01)
# Clear previous gradients
optimizer.zero_grad()
# Forward pass
model_output = (weights * 4).sum()
# Backward pass: compute gradients
model_output.backward()
# Update weights
optimizer.step()

print("Updated weights:", weights)
print("Gradients:", weights.grad)

#Backpropogation
x = torch.tensor(1.0)
y = torch.tensor(2.0)
w = torch.tensor(1.0, requires_grad= True)
y_hat = w * x
loss = (y_hat - y) ** 2
loss.backward()
print(w.grad)

#update weights
#forward pass
#---------------------------------------------------------------
#   Manually use numpy to calculate the forward, gradient, loss
# f = w * x
# f = x * 2
x = np.array([1,2,3,4],dtype=np.float32)
y = np.array([2,4,6,8],dtype=np.float32)
w = 0.0
# model prediciton
def forward(x):
    return w * x
# loss = MSE
def loss(y,y_predicted):
    return ((y_predicted -y) **2).mean()
#gradient MSE = 1/N* (w*x-y)**2
# dJ/dw = 1/N 2x(w*x-y)
def gradient(x,y,y_predicted):
    return np.dot(2*x, y_predicted-y).mean()
print(f'Prediction before training: f(5)={forward(5):.3f}')  
#Training
learning_rate = 0.01
n_iters = 10
for epoch in range(n_iters):
    #prediction = forward pass
    y_pred = forward(x)
    l = loss(y, y_pred)
    dw = gradient(x,y,y_pred)
    # update weights
    w = w-learning_rate * dw
    #if epoch % 1 == 0:
    #    print(f'epoch {epoch+1}:w={w:.3f},loss={l:.3f}')
print(f'Prediction after training: f(5)={forward(5):.3f}')       

#---------------------------------------------------------------
#   use torch to calculate the forward, gradient, loss
# f = w * x
# f = x * 2
x = torch.tensor([1,2,3,4],dtype=torch.float32)
y = torch.tensor([2,4,6,8],dtype=torch.float32)
w = torch.tensor(0.0,dtype=torch.float32, requires_grad=True)
# model prediciton
def forward(x):
    return w * x
# loss = MSE
def loss(y,y_predicted):
    return ((y_predicted -y) **2).mean()
#gradient MSE = 1/N* (w*x-y)**2
# dJ/dw = 1/N 2x(w*x-y)
print(f'Prediction before training: f(5)={forward(5):.3f}')  
#Training
learning_rate = 0.01
n_iters = 100
for epoch in range(n_iters):
    #prediction = forward pass
    y_pred = forward(x)
    l = loss(y, y_pred)
    l.backward() # dl/dw
    # update weights
    with torch.no_grad():
        w -= learning_rate * w.grad
    # zero gradients
    w.grad.zero_()
    #if epoch % 1 == 0:
    #    print(f'epoch {epoch+1}:w={w:.3f},loss={l:.3f}')
print(f'Prediction after training: f(5)={forward(5):.3f}')  


#----------- loss computation and parameter updates
# prepare data
bc = datasets.load_breast_cancer()
x, y = bc.data, bc.target

n_samples,n_features = x.shape
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size = 0.2, random_state=1234)

#scale
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

x_train = torch.from_numpy(x_train.astype(np.float32))
x_test = torch.from_numpy(x_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

# Reshape the training and testing labels 
# from a 1D tensor into a 2D column tensor.
#This is commonly used in PyTorch regression, binary classification, 
# and neural network training to make the target tensor's shape match the model's output.
y_train = y_train.view(y_train.shape[0],1)
y_test = y_test.view(y_test.shape[0],1)