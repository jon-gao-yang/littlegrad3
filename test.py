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
    threshold = 1e-10    # for some reason in cupy 1/b and b**-1 give slightly different answers, so Tensor and cupy array division gives slightly different answers

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

#based on a practice assignment from Andrew Ng's "Advanced Learning Algorithms" Coursera course
def plot_kaggle_data(X, y, model, predict=False):
    m, n = X.shape

    fig, axes = plt.subplots(8,8, figsize=(5,5))
    fig.tight_layout(pad=0.13,rect=[0, 0.03, 1, 0.91]) #[left, bottom, right, top]

    for i,ax in enumerate(axes.flat):
        # Select random indices
        random_index = cp.random.randint(m)
        
        # Select rows corresponding to the random indices and reshape the image
        X_random_reshaped = X[random_index].reshape((28,28))
        
        # Display the image
        ax.imshow(X_random_reshaped, cmap='gray')

        yhat = None
        # Predict using the Neural Network
        if predict:
            probs, log_softmax = softmax(model(Tensor(X[random_index])))
            yhat = cp.argmax(probs.data)
        
        # Display the label above the image
        ax.set_title(f"{int(y[random_index])},{yhat}",fontsize=10)
        ax.set_axis_off()
    fig.suptitle("Label, yhat", fontsize=14)
    plt.show()

def write_kaggle_submission(model):
    X = cp.empty((28000, 28*28), dtype = int)
    with open('digit-recognizer/test.csv', newline='\n') as csvfile:
        digitreader = csv.reader(csvfile, delimiter=',')
        for row in digitreader:
            if digitreader.line_num != 1: #line_num starts at 1, not 0
                X[digitreader.line_num-2] = [int(char) for char in row] #no labels so entire row is pixel data
    
    X = (X-cp.average(X)) / cp.std(X)  #data normalization
    with open('digit-recognizer/submission.csv', newline='\n', mode = 'w') as csvfile:
        digitwriter = csv.writer(csvfile, delimiter=',')
        digitwriter.writerow(['ImageId','Label'])
        for i in range(X.shape[0]):
            probs, log_softmax = softmax(model(Tensor(X[i])))
            digitwriter.writerow([i+1, cp.argmax(probs.data)])  #take most likely digit as guess

def softmax(logits):
  counts = logits.exp()
  denominator = counts @ cp.ones(shape = (counts.data.size, counts.data.size)) #2D ones matrix avoids denom broadcasting which fucks up gradient shape
  return counts / denominator, logits - denominator.log() #probs, log_softmax

#modified from karpathy's demo.ipynb
def loss(X, y, model, batch_size=None):

    if batch_size is None:  #dataloader
        Xb, yb = X, y
    else:
        ri = cp.random.permutation(X.shape[0])[:batch_size] #shuffles the X indexes and returns the first 10
        Xb, yb = X[ri], y[ri]

    losses, accuracy = [], []
    for (xrow, yrow) in zip(Xb, yb):
        probs, log_softmax = softmax(model(Tensor(xrow)))        
        losses.append(-log_softmax @ Tensor([index == yrow for index in range(log_softmax.data.size)]).transpose())
        # ^ cross entropy loss (can't just take log_softmax[yrow] or else you lose track of gradients and backward() doesn't work)
        accuracy.append(yrow == cp.argmax(probs.data))

    # L2 regularization
    alpha = 0.0033
    reg_loss = alpha * sum([p.flatten()@p.flatten().transpose() for p in model.parameters()])
    return cp.average(losses) + reg_loss, cp.average(accuracy) # (total_loss = data_loss + reg_loss)

def kaggle_training(epochs = 10, batch_size = None):
    X = cp.empty((42000, 28*28), dtype = int)
    y = cp.empty(42000, dtype = int)
    with open('digit-recognizer/train.csv', newline='\n') as csvfile:
        digitreader = csv.reader(csvfile, delimiter=',')
        for row in digitreader:
            if digitreader.line_num != 1: #line_num starts at 1, not 0
                y[digitreader.line_num-2] = int(row[0])
                X[digitreader.line_num-2] = [int(char) for char in row[1:]]
    
    X = (X-cp.average(X)) / cp.std(X)  #data normalization
        
    class LinearNet:
        def __init__(self):
            
            self.params = {                
                'w1' : Tensor(cp.random.randn(28*28, 320) * cp.sqrt(2 / (28*28))), # 2 / (# of inputs from last layer)
                'b1' : Tensor(cp.zeros((1, 320))),
                'w2' : Tensor(cp.random.randn(320, 160) * cp.sqrt(2 / 320)), # 2 / (# of inputs from last layer)
                'b2' : Tensor(cp.zeros((1, 160))),
                'w3' : Tensor(cp.random.randn(160, 80) * cp.sqrt(2 / 160)), # 2 / (# of inputs from last layer)
                'b3' : Tensor(cp.zeros((1, 80))),
                'w4' : Tensor(cp.random.randn(80, 40) * cp.sqrt(2 / 80)), # 2 / (# of inputs from last layer)
                'b4' : Tensor(cp.zeros((1, 40))),
                'w5' : Tensor(cp.random.randn(40, 20) * cp.sqrt(2 / 40)), # 2 / (# of inputs from last layer)
                'b5' : Tensor(cp.zeros((1, 20))),
                'w6' : Tensor(cp.random.randn(20, 10) * cp.sqrt(2 / 20)), # 2 / (# of inputs from last layer)
                'b6' : Tensor(cp.zeros((1, 10)))
            }

        def parameters(self):
            return self.params.values()
        
        def zero_grad(self):
            for param in self.params.values():
                param.grad.fill(0)

        def __call__(self, x:Tensor) -> Tensor:
            l1 = ((x @ self.params['w1']) + self.params['b1']).relu()
            l2 = ((l1 @ self.params['w2']) + self.params['b2']).relu()
            l3 = ((l2 @ self.params['w3']) + self.params['b3']).relu()
            l4 = ((l3 @ self.params['w4']) + self.params['b4']).relu()
            l5 = ((l4 @ self.params['w5']) + self.params['b5']).relu()
            return (l5 @ self.params['w6']) + self.params['b6']
    
    model = LinearNet()
    learning_rate, beta1, beta2, epsilon, weight_decay = 0.00058, 0.9, 0.999, 1e-10, 0.05 #NOTE: cost will not converge if learning rate is too high
    print('TRAINING BEGINS')
    startTime = time.time()

    # optimization
    for k in range(epochs):
        
        # forward
        total_loss, acc = loss(X, y, model, batch_size = batch_size)

        # backward
        model.zero_grad()
        total_loss.backprop()
        
        #update parameters w/ AdamW Algorithm
        for p in model.parameters(): 
            p.data -= p.data * learning_rate * weight_decay
            p.v = (beta1 * p.v) + ((1-beta1) * p.grad)
            p.s = (beta2 * p.s) + ((1-beta2) * p.grad * p.grad)
            v_dp_corrected = p.v / (1 - (beta1**(k+1)))
            s_dp_corrected = p.s / (1 - (beta2**(k+1)))
            p.data -= learning_rate * v_dp_corrected / (cp.sqrt(s_dp_corrected) + epsilon)
        
        if k % 1 == 0:
            print(f"step {k} loss {total_loss.data}, accuracy {acc*100}%")

    endTime = time.time()
    print('TRAINING COMPLETE (in', round((endTime - startTime) / 60, 3), 'min)')
    plot_kaggle_data(X, y, model, predict = True)
    print('BEGINNING TEST SET INFERENCE')
    write_kaggle_submission(model)
    print('TEST SET INFERENCE COMPLETE')

#############################################################################################

my_test()
#test_sanity_check()
#test_more_ops()
#mlp_test()

#kaggle_training(epochs = 510, batch_size = 400)