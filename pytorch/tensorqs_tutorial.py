import torch
import numpy as np
import torch.nn as nn

data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)

print(x_data.shape)

np_array = np.array(data)
x_np = torch.from_numpy(np_array)
print(x_np.shape)

x_ones = torch.ones_like(x_data)
print(x_ones)

x_rand = torch.rand_like(x_data, dtype=torch.float)
x_rand

shape = (
    2,
    3,
)
print(shape)

rand_tensor = torch.rand(shape)
print(rand_tensor)

ones_tensor = torch.ones(shape)
print(ones_tensor)

zeros_tensor = torch.zeros(shape)
print(zeros_tensor)

tensor = torch.rand(3, 4)
print(tensor.shape)
print(tensor.dtype)
print(tensor.device)

tensor2 = torch.ones(4, 4)
print(tensor2.shape)
print(tensor2)
# first row
print(tensor2[0])
# first column
print(tensor2[:, 0])
# last column
# ... is called ellipsis, which is the same as :
print(tensor2[..., -1])

# Join
t1 = torch.cat((tensor2, tensor2), dim=0)
print(t1)
print(t1.shape)
t2 = torch.cat((tensor2, tensor2), dim=1)
print(t2)


# Arithmetic operations

# 8 * 4
print(t1.shape)

# 4 * 8
print(t2.shape)

# 8 * 8
print(t1 @ t2)
print(t1.matmul(t1.T))

print(t1 * t1)
print(t1.mul(t1))

z3 = torch.rand_like(t1)
print(z3)
torch.mul(t1, t1, out=z3)

print(torch.tensor([1, 2]) * torch.tensor([3, 4]))

print(t1.sum())
print(t1.sum().item())

print(t1.add(5))
print(t1)
print(t1.add_(5))
print(t1)


s1 = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(s1.shape)
print(torch.sum(s1, dim=0))
print(torch.sum(s1, dim=1))

s2 = torch.ones(5, 1)
s2.shape

squeezed = torch.squeeze(s2)
squeezed.shape
squeezed

unsqueeze1 = torch.unsqueeze(squeezed, 0)
unsqueeze1.shape
unsqueeze1

unsqueeze2 = torch.unsqueeze(squeezed, dim=1)
unsqueeze2.shape
unsqueeze2
