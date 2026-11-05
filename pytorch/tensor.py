import torch
import numpy as np

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

# Gradient calculation with Autograd
x = torch.rand(3, requires_grad = True)
y = x + 2
z = y*y*2
z = z.mean()
z.backward() # dz/dx
print(x.grad)