#import torch
import cupy as cp
from littlegrad3.engine import Tensor
import matplotlib.pyplot as plt
#import csv
import time

###### [ 1/4 : MODEL INITIALIZATION ] ######

class LinearNet:
    def __init__(self):
        
        self.params = {                
            'w1' : Tensor(cp.random.randn(28*28, 160) * cp.sqrt(2 / (28*28))), # 2 / (# of inputs from last layer)
            'b1' : Tensor(cp.zeros((1, 160))),
            'w2' : Tensor(cp.random.randn(160, 80) * cp.sqrt(2 / 160)), # 2 / (# of inputs from last layer)
            'b2' : Tensor(cp.zeros((1, 80))),
            'w3' : Tensor(cp.random.randn(80, 40) * cp.sqrt(2 / 80)), # 2 / (# of inputs from last layer)
            'b3' : Tensor(cp.zeros((1, 40))),
            'w4' : Tensor(cp.random.randn(40, 20) * cp.sqrt(2 / 40)), # 2 / (# of inputs from last layer)
            'b4' : Tensor(cp.zeros((1, 20))),
            'w5' : Tensor(cp.random.randn(20, 10) * cp.sqrt(2 / 20)), # 2 / (# of inputs from last layer)
            'b5' : Tensor(cp.zeros((1, 10)))
        }

    def parameters(self):
        return self.params.values()
    
    def zero_grad(self):
        for param in self.params.values():
            param.grad.fill(0)

    def param_num(self):
        return sum([t.data.size for t in self.params.values()])

    def __call__(self, x:Tensor) -> Tensor:
        l1 = ((x @ self.params['w1']) + self.params['b1']).relu()
        l2 = ((l1 @ self.params['w2']) + self.params['b2']).relu()
        l3 = ((l2 @ self.params['w3']) + self.params['b3']).relu()
        l4 = ((l3 @ self.params['w4']) + self.params['b4']).relu()
        return (l4 @ self.params['w5']) + self.params['b5']
    
###### [ 2/4 : HELPER FUNCTIONS ] ######

# based on code from Andrew Ng's "Advanced Learning Algorithms" Coursera course
def plot_kaggle_data(X, y, model, predict=False):
    fig, axes = plt.subplots(8,8, figsize=(5,5))
    fig.tight_layout(pad=0.13,rect=[0, 0.03, 1, 0.91]) #[left, bottom, right, top]

    for i,ax in enumerate(axes.flat):
        # Select random indices
        random_index = cp.random.randint(X.shape[0])
        
        # Display the image
        ax.imshow(X[random_index].reshape(28, 28).get(), cmap='gray')

        yhat = None
        # Predict using the Neural Network
        if predict:
            probs, log_softmax = softmax(model(Tensor(X[random_index])))
            yhat = cp.argmax(probs.data)
        
        # Display the label above the image
        ax.set_title(f"{y[random_index]},{yhat}",fontsize=10)
        ax.set_axis_off()
    fig.suptitle("Label, yhat", fontsize=14)
    plt.show()

def write_kaggle_submission(model):
    print('BEGINNING TEST SET INFERENCE')

    X = cp.loadtxt('digit-recognizer/test.csv', dtype = int, delimiter = ',', skiprows = 1) # data loading
    # X = X/255  # data normalization
    X = (X - cp.mean(X))/cp.std(X) # data normalization

    probs, log_softmax = softmax(model(Tensor(X))) # inference
    out = cp.concatenate((cp.arange(1, X.shape[0]+1).reshape((-1, 1)), cp.argmax(probs.data, axis = 1).reshape((-1, 1))), axis = 1)
    cp.savetxt('digit-recognizer/submission.csv', out, delimiter = ',', fmt = '%s', header = 'ImageId,Label', comments = '')

    print('TEST SET INFERENCE COMPLETE')

def softmax(logits):
  counts = logits.exp()
  denominator = counts @ cp.ones(shape = (counts.data.shape[-1], counts.data.shape[-1])) #2D ones matrix avoids denom broadcasting which fucks up gradient shape
  return counts / denominator, logits - denominator.log() #probs, log_softmax

# based on code from Andrej Karpathy's "Micrograd" github repo
def loss(X, y, model, batch_size=None, regularization=True, alpha=1e-8):

    if batch_size is None:  #dataloader
        Xb, yb = X, y
    else:
        ri = cp.random.permutation(X.shape[0])[:batch_size] # shuffles the X indexes and selects the first batch_size indices
        Xb, yb = X[ri], y[ri]

    # x --(model)--> logits --(softmax)--> probs --(-log)--> nll loss --(avg over batch)--> cost --(backprop)--> grads
    probs, log_softmax = softmax(model(Tensor(Xb))) 
    losses = Tensor(cp.zeros_like(probs.data))
    losses.data[cp.arange(probs.data.shape[0]), yb] = -1
    losses = (losses * log_softmax).flatten() @ Tensor(cp.ones((probs.data.size, 1))) / probs.data.shape[0]
    accuracy = cp.average(cp.argmax(probs.data, axis = -1) == yb)

    if regularization: # L2 regularization (total_loss = data_loss + reg_loss)
        losses += alpha * sum([(p.reshape((1, -1))@p.reshape((-1, 1))).data for p in model.parameters()])
    return losses, accuracy

###### [ 3/4 : MAIN FUNCTION ] ######

def kaggle_training(model, epochs = 10, batch_size = None, regularization = True, learning_rate = 0.0001, alpha = 1e-8):
    [y, X] = cp.split(cp.loadtxt('digit-recognizer/train.csv', dtype = int, delimiter = ',', skiprows = 1), [1], axis = 1)
    # ^ NOTE: loading data from file, then splitting into labels (first col) and pixel vals
    y = cp.squeeze(y) # 2D -> 1D

    # image translation for data augmentation (NOTE: default cp.concatenate() axis is 0)
    px = 1
    Xsplit = cp.pad(X.reshape((42000, 28, 28)), px)[px:-px]                  # adding px pixels to height and width for translation
    X = cp.concatenate((Xsplit[:, 0:-px*2, 0:-px*2].reshape((42000, 28*28)), # slice up & left       --> shifts image down & right
                    Xsplit[:, 0:-px*2, px:-px].reshape((42000, 28*28)),      # slice up & center     --> shifts image down & center
                    Xsplit[:, 0:-px*2, px*2:].reshape((42000, 28*28)),       # slice up & right      --> shifts image down & left
                    Xsplit[:, px:-px, 0:-px*2].reshape((42000, 28*28)),      # slice center & left   --> shifts image center & right
                    Xsplit[:, px:-px, px:-px].reshape((42000, 28*28)),       # slice center & center --> shifts image center & center
                    Xsplit[:, px:-px, px*2:].reshape((42000, 28*28)),        # slice center & right  --> shifts image center & left
                    Xsplit[:, px*2:, 0:-px*2].reshape((42000, 28*28)),       # slice down & left     --> shifts image up & right
                    Xsplit[:, px*2:, px:-px].reshape((42000, 28*28)),        # slice down & center   --> shifts image up & center
                    Xsplit[:, px*2:, px*2:].reshape((42000, 28*28))))        # slice down & right    --> shifts image up & left
    y = cp.tile(y, 9) # duplicating labels for augmented images

    X = (X - cp.mean(X))/cp.std(X) # data normalization
    #X = X/255  # data normalization
    beta1, beta2, epsilon, weight_decay = 0.9, 0.999, 1e-10, 0.01 # AdamW hyperparameter creation
    print('TRAINING BEGINS (with', model.param_num(), 'parameters)')
    startTime = time.time()

    # optimization
    for k in range(epochs):
        
        # forward
        total_loss, acc = loss(X, y, model, batch_size = batch_size, regularization = regularization, alpha = alpha)

        # backward
        model.zero_grad()
        total_loss.backprop()
        
        # update parameters w/ AdamW Algorithm
        for p in model.parameters(): 
            #p.data -= learning_rate * p.grad # TODO: ALLOW OPTIMIZER CHOICE
            p.data -= p.data * learning_rate * weight_decay
            p.v = (beta1 * p.v) + ((1-beta1) * p.grad)
            p.s = (beta2 * p.s) + ((1-beta2) * p.grad * p.grad)
            v_dp_corrected = p.v / (1 - (beta1**(k+1)))
            s_dp_corrected = p.s / (1 - (beta2**(k+1)))
            p.data -= learning_rate * v_dp_corrected / (cp.sqrt(s_dp_corrected) + epsilon) # doesn't work for broadasted bias v/s/grad tensors

        #print(f"step {k} loss {total_loss.data.real[0, 0]}, accuracy {acc*100}%") # NOTE: COMMENT OUT THIS LINE FOR FASTER TRAINING

    print('TRAINING COMPLETE (in', time.time() - startTime, 'sec)')
    plot_kaggle_data(X, y, model, predict = True)
    write_kaggle_submission(model)

###### [ 4/4 : MAIN FUNCTION EXECUTION ] ###### 
# NOTE: cost will not converge if learning rate is too high

# 03/15/25 - achieved ~98% (97.76%) accuracy on kaggle MNIST test set with <5 (4.995) seconds of training with the following command:
kaggle_training(model = LinearNet(), epochs = 600, batch_size = 2000, regularization = False, learning_rate = 0.0025799, alpha = 0.0001)
