import torch
import cupy as cp
from littlegrad3.engine import Tensor
import matplotlib.pyplot as plt
import csv
import time

def my_test():
    a = [2, 3]
    b = [[5, 10]]
    c = 4
    threshold = 1e-10    # for some reason in cupy 1/x and x**-1 give slightly different answers, so cupy array and Tensor division gives slightly different answers

    # a = [2., 3.]
    # b = [[5., 10.]]
    # c = 4.

    # a = -5
    # b = 3
    # c = 1

    aVal = Tensor(a)
    bVal = Tensor(b)

    a = cp.array(a).reshape((1,-1))
    b = cp.array(b).reshape((1,-1))

    print("a:      | Passed == ", type(aVal) == Tensor)
    print("b:      | Passed == ", type(bVal) == Tensor)
    print("a.data: | Passed == ", aVal.data == a)
    print("b.data: | Passed == ", bVal.data == b)
    print("a.ndim: | Passed == ", aVal.data.ndim == 2)
    print("b.ndim: | Passed == ", bVal.data.ndim == 2)

    print("a+c:    | Passed == ", (aVal+c).data == a+c)
    print("c+a:    | Passed == ", (c+aVal).data == c+a)
    print("a+b:    | Passed == ", (aVal+bVal).data == a+b)
    print("b+a:    | Passed == ", (bVal+aVal).data == b+a)

    print("a-c:    | Passed == ", (aVal-c).data == a-c)
    print("c-a:    | Passed == ", (c-aVal).data == c-a)
    print("a-b:    | Passed == ", (aVal-bVal).data == a-b)
    print("b-a:    | Passed == ", (bVal-aVal).data == b-a)

    print("a*c:    | Passed == ", (aVal*c).data == a*c)
    print("c*a:    | Passed == ", (c*aVal).data == c*a)
    print("a*b:    | Passed == ", (aVal*bVal).data == a*b)
    print("b*a:    | Passed == ", (bVal*aVal).data == b*a)

    print("a/c:    | Passed == ", (aVal/c).data -  a/c < threshold)
    print("c/a:    | Passed == ", (c/aVal).data - c/a < threshold)
    print("a/b:    | Passed == ", (aVal/bVal).data - a/b < threshold)
    print("b/a:    | Passed == ", (bVal/aVal).data - b/a < threshold)

    print("-a:     | Passed == ", (-aVal).data == -a)
    print("a**c:   | Passed == ", (aVal**c).data == a**c)
    print("c**a:   | Passed == ", (c**aVal).data == c**a)
    print()

# from karpathy/micrograd test_engine.py:
def test_sanity_check():

    x = Tensor(-4.0)
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backprop()
    xmg, ymg = x, y

    x = torch.Tensor([-4.0]).double()
    x.requires_grad = True
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    xpt, ypt = x, y

    # forward pass went well
    assert ymg.data == ypt.data.item()
    # backward pass went well
    #print('xmg.grad: ', xmg.grad)
    #print('xpt.grad.item(): ', xpt.grad.item())
    assert xmg.grad == xpt.grad.item()
    print('Karpathy #1: Passed == True')

def test_more_ops():

    a = Tensor(-4.0)
    b = Tensor(2.0)
    c = a + b
    d = a * b + b**3
    c += c + 1
    c += 1 + c + (-a)
    d += d * 2 + (b + a).relu()
    d += 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g += 10.0 / f
    g.backprop()
    amg, bmg, gmg = a, b, g

    a = torch.Tensor([-4.0]).double()
    b = torch.Tensor([2.0]).double()
    a.requires_grad = True
    b.requires_grad = True
    c = a + b
    d = a * b + b**3
    c = c + c + 1
    c = c + 1 + c + (-a)
    d = d + d * 2 + (b + a).relu()
    d = d + 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g = g + 10.0 / f
    g.backward()
    apt, bpt, gpt = a, b, g

    tol = 1e-6
    # forward pass went well
    assert abs(gmg.data - gpt.data.item()) < tol
    # backward pass went well
    assert abs(amg.grad - apt.grad.item()) < tol
    assert abs(bmg.grad - bpt.grad.item()) < tol
    print('Karpathy #2: Passed == True')
    print()

def mlp_test():
    class LinearNet:
        def __init__(self):
            self.params = {
            'w1' : Tensor(2 * cp.random.random_sample((3, 4)) - 1),
            'b1' : Tensor(cp.zeros((1, 4))),
            'w2' : Tensor(2 * cp.random.random_sample((4, 4)) - 1),
            'b2' : Tensor(cp.zeros((1, 4))),
            'w3' : Tensor(2 * cp.random.random_sample((4, 1)) - 1),
            'b3' : Tensor(cp.zeros((1, 1)))
            }

        def parameters(self):
            return self.params.values()
        
        def zero_grad(self):
            for param in self.params.values():
                param.grad *= 0

        def __call__(self, x:Tensor) -> Tensor:
            return ((x@self.params['w1']+self.params['b1']).relu()@self.params['w2']+self.params['b2']).relu()@self.params['w3']+self.params['b3']
    
    n = LinearNet()
    xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
    ]
    ys = [1.0, -1.0, -1.0, 1.0] # desired targets

    for k in range(100):

        # forward pass
        ypred = [n(Tensor(x)) for x in xs]
        loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))
        
        # backward pass
        n.zero_grad()
        loss.backprop()
        
        # update
        for p in n.parameters():
            p.data += -0.01 * p.grad
        
        print(k, loss.data)

    print(ypred)
    print()

#############################################################################################

my_test()
test_sanity_check()
test_more_ops()
mlp_test()