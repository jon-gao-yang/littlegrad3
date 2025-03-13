import math
import cupy as cp

class Tensor:
    """ stores a multidimensional matrix of values and their corresponding gradients """

    def __init__(self, data, children = (), op = ''):    # user should be forced to specify data but not children or op

        # TODO: THIS STILL ALLOWS BROADCASTING, CHECK IF THAT'S OKAY
        # TODO: CONFIRM LITTLEGRAD SELF.PREV COMMENT THAT PREV W/O SET() BREAKS TOPOSORT BECAUSE TUPLES CAN'T ADD/REMOVE ITEMS
        # TODO: IS RPOW RIGHT?
        # TODO: TRY DIFFERENT ACTIVATION FUNCTIONS THAN RELU?
        # NOTE: if other is a scalar then grad doesn't need to be tracked, it it's not a number that's user error (removed form sub())
        # NOTE: for some reason in cupy 1/b and b**-1 give slightly different answers, so Tensor and cupy array division gives slightly different answers

        self.data = cp.array(data, dtype = cp.float64)    # cupy handles data type conversion, cp.array() on a cp.array does nothing
        self.data = self.data if self.data.ndim >= 2 else cp.atleast_2d(self.data)
        self.grad = cp.zeros_like(self.data)    # zeros_like() uses the same data type as its input by default
        self.prev = set(children)    # tuples aka "()" are used for convenience as inputs, but need to be converted to set so elements can be added/removed later
        self.backward = lambda: None    # lambda function (aka single-line function with no name) that does nothing, that will be filled in with a custom backward function based on the operation that created that Tensor
        self.op = op
    
    def __add__(self, other):
        """ elementwise addition of two Tensors """

        other = other if isinstance(other, Tensor) else Tensor(other)
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

        other = other if isinstance(other, Tensor) else Tensor(other)
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

        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data ** other.data, (self, other), '**')
    
        def backward():
            self.grad += other.data * (self.data ** (other.data - 1)) * out.grad
            other.grad += (self.data ** other.data) * cp.log(self.data + ((self.data == 0) * math.ulp(0.0))) * out.grad
        
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
        """ swaps the last two axes of a Tensor"""

        out = Tensor(cp.swapaxes(self.data, -2, -1), (self,), 'T')
    
        def backward():
            self.grad += cp.swapaxes(out.grad, -2, -1)
        
        out.backward = backward
        return out
    
    def __matmul__(self, other):
        """ matrix product of two Tensors """

        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data, (self, other), '@')
    
        def backward():
            self.grad += out.grad @ cp.swapaxes(other.data, -2, -1)
            other.grad += cp.swapaxes(self.data, -2, -1) * out.grad
        
        out.backward = backward
        return out
    
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