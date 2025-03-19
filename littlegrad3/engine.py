from math import ulp
import cupy as cp

class Tensor:
    """ stores a multidimensional matrix of values and their corresponding gradients """

    def __init__(self, data, children = (), op = ''):    # user should be forced to specify data but not children or op
        self.data = cp.array(data, dtype = cp.float64)    # cupy handles data type conversion, cp.array() on a cp.array does nothing
        self.data = self.data if self.data.ndim >= 2 else cp.atleast_2d(self.data)
        self.grad, self.v, self.s = cp.zeros_like(self.data), cp.zeros_like(self.data), cp.zeros_like(self.data) # zeros_like() uses the same data type as its input by default
        self.prev = set(children)    # tuples aka "()" are used for convenience as inputs, but need to be converted to set so elements can be added/removed later
        self.backward = lambda: None    # lambda function (aka single-line function with no name) that does nothing, that will be filled in with a custom backward function based on the operation that created that Tensor
        self.op = op
    
    def __add__(self, other):
        """ elementwise addition of two Tensors """

        (self, other) = Tensor.makeCompatible(self, other, same_size = True)
        out = Tensor(self.data + other.data, (self, other), '+')
    
        def backward():
            self.grad += out.grad
            other.grad += out.grad
        
        out.backward = backward
        return out
    
    def __radd__(self, other):    
        """ Tensor.__add__(non-Tensor self, Tensor other) failed, so interpreter tries Tensor.__radd__(Tensor self, non-Tensor other) """
        
        return self + other
    
    def __mul__(self, other):
        """ elementwise multiplication of two Tensors """

        (self, other) = Tensor.makeCompatible(self, other, same_size = True)
        out = Tensor(self.data * other.data, (self, other), '*')
    
        def backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        
        out.backward = backward
        return out
    
    def __rmul__(self, other):
        """ Tensor.__mul__(non-Tensor self, Tensor other) failed, so interpreter tries Tensor.__rmul__(Tensor self, non-Tensor other) """
        
        return self * other
        
    def __sub__(self, other):
        """ elementwise subtraction of two Tensors """

        other = other if isinstance(other, Tensor) else Tensor(other)
        return self + -other
    
    def __rsub__(self, other):
        """ Tensor.__sub__(non-Tensor self, Tensor other) failed, so interpreter tries Tensor.__rsub__(Tensor self, non-Tensor other) """

        return -self + other
      
    def __truediv__(self, other):
        """ elementwise true (aka floating-point) division of two Tensors """

        other = other if isinstance(other, Tensor) else Tensor(other)
        return self * other**-1
    
    def __rtruediv__(self, other):
        """ Tensor.__truediv__(non-Tensor self, Tensor other) failed, so interpreter tries Tensor.__rtruediv__(Tensor self, non-Tensor other) """
        
        return self**-1 * other
    
    def __pow__(self, other):
        """ elementwise exponentiation of two Tensors """

        (self, other) = Tensor.makeCompatible(self, other, same_size = True)
        out = Tensor(self.data ** other.data, (self, other), '**')
    
        def backward():
            self.grad += other.data * (self.data ** (other.data - 1)) * out.grad
            other.grad += (self.data ** other.data) * cp.log(self.data + ((self.data == 0) * ulp(0.0))) * out.grad
        
        out.backward = backward
        return out
    
    def __rpow__(self, other):
        """ Tensor.__pow__(non-Tensor self, Tensor other) failed, so interpreter tries Tensor.__rpow__(Tensor self, non-Tensor other) """
        
        other = other if isinstance(other, Tensor) else Tensor(other)
        return other ** self
    
    def __neg__(self):
        """ elementwise negative of one Tensor """

        return self * -1
    
    def exp(self):
        """ elementwise natural exponential of one Tensor """

        out = Tensor(cp.exp(self.data), (self,), 'exp')
    
        def backward():
            self.grad += cp.exp(self.data) * out.grad
        
        out.backward = backward
        return out
    
    def log(self):
        """ elementwise natural logarithm of one Tensor """

        out = Tensor(cp.log(self.data), (self,), 'log')
    
        def backward():
            self.grad += out.grad / self.data
        
        out.backward = backward
        return out
    
    def relu(self):
        """ elementwise rectified linear unit for one Tensor """

        out = Tensor((self.data > 0) * self.data, (self,), 'ReLU')
    
        def backward():
            self.grad += (self.data > 0) * out.grad
        
        out.backward = backward
        return out
    
    def transpose(self):
        """ swaps the last two axes of a Tensor """

        out = Tensor(cp.swapaxes(self.data, -2, -1), (self,), 'T')
    
        def backward():
            self.grad += cp.swapaxes(out.grad, -2, -1)
        
        out.backward = backward
        return out
    
    def reshape(self, shape):
        """ changes the shape of a Tensor """

        out = Tensor(cp.reshape(self.data, shape), (self,), 'R')
    
        def backward():
            self.grad += cp.reshape(out.grad, self.data.shape)
        
        out.backward = backward
        return out
    
    def flatten(self):
        """ swaps the last two axes of a Tensor"""

        return self.reshape((1, -1))
    
    def tile(self, reps):
        """ duplicates a Tensor a certain number of times (specified by reps) """

        reps = reps if isinstance(reps, tuple) else tuple(reps)
        out = Tensor(data = cp.tile(self.data, reps), children = (self,), op = 'tile')

        def backward():
            reps_arr = []
            for i in range(len(reps)):
                if reps[i] > 1:
                    reps_arr.append(i)
            self.grad += cp.average(out.grad, axis = tuple(reps_arr))
        out.backward = backward
        return out
    
    def __matmul__(self, other):
        """ matrix product of two Tensors """

        (self, other) = Tensor.makeCompatible(self, other, same_size = False)
        out = Tensor(self.data @ other.data, (self, other), '@')
    
        def backward():
            self.grad += out.grad @ cp.swapaxes(other.data, -2, -1)
            other.grad += cp.swapaxes(self.data, -2, -1) @ out.grad
        
        out.backward = backward
        return out
    
    def makeCompatible(self, other, same_size:bool): 
        """ manual broadcasting of two Tensors to keep track of gradients """

        other = other if isinstance(other, Tensor) else Tensor(other) # make sure 'other' is Tensor

        # make sure self and other have same length
        dimDiff = self.data.ndim - other.data.ndim
        if dimDiff > 0: # if self.data.ndim > other.data.ndim
            other = other.reshape([other.data.shape[i-dimDiff] if i-dimDiff > 0 else 1 for i in range(self.data.ndim)])
        elif dimDiff < 0: # if other.data.ndim > self.data.ndim
            self = self.reshape([self.data.shape[i+dimDiff] if i+dimDiff > 0 else 1 for i in range(other.data.shape)])
        
        # get broadcast repetition counts for each axis that needs broadcasting and make min broadcast count 1
        otherBroadcastDims = [self.data.shape[idx] if other.data.shape[idx] == 1 else 1 for idx in range(other.data.ndim)]
        selfBroadcastDims = [other.data.shape[idx] if self.data.shape[idx] == 1 else 1 for idx in range(self.data.ndim)]

        # if matmul instead of element-wise op, make inner dims compatible
        if not same_size:
            selfBroadcastDims[-2:] = [1, 1]
            otherBroadcastDims[-2:] = [1, 1]
            if self.data.shape[-1] == 1:
                selfBroadcastDims[-1] = other.data.shape[-2] # make inner dims compatible
            elif other.data.shape[-2] == 1:
                otherBroadcastDims[-2] = self.data.shape[-1] # make inner dims compatible

        # if any axes need broadcasting, broadcast them w/ cp.tile before returning Tensors
        self = self.tile(selfBroadcastDims) if (sorted(selfBroadcastDims)[-1] > 1) else self
        other = other.tile(otherBroadcastDims) if sorted(otherBroadcastDims)[-1] > 1 else other
        return (self, other)
    
    def backprop(self):
        """ automatic partial differentiation of one Tensor """

        nodeList = []
        visited = set()
        def toposort(node):
            for child in node.prev:
                if child not in visited:
                    toposort(child)
            visited.add(node)
            nodeList.append(node)
        toposort(self)
        
        self.grad = cp.ones_like(self.data) # ones_like() uses the same data type as its input by default
        for node in reversed(nodeList): # nodeList.reverse() changes nodeList without returning anything, so doesn't work for inline reversal
            node.backward()