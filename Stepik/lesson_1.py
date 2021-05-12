import torch

w = torch.tensor(
    [[5.,10.],
      [1.,2.]],requires_grad=True)
alpha = 0.001
optimizer =  torch.optim.SGD([w],lr=alpha)
for _ in range(500):

    function = torch.log(torch.log(w + 7)).prod()
    function.backward()
    optimizer.step()
    w.grad.zero_()

# print(w) # Код для самопроверки
