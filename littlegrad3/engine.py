import math
import cupy as cp

class Tensor:
    def __init__(self, data, children = (), op = ''):    # user should be forced to specify data but not children or op

        # TODO: THIS STILL ALLOWS BROADCASTING, CHECK IF THAT'S OKAY
        # TODO: CONFIRM LITTLEGRAD SELF.PREV COMMENT THAT PREV W/O SET() BREAKS TOPOSORT BECAUSE TUPLES CAN'T ADD/REMOVE ITEMS
        # TODO: IS RPOW RIGHT?
        # NOTE: if other is a scalar then grad doesn't need to be tracked, it it's not a number that's user error (removed form sub())
        # NOTE: for some reason in cupy 1/b and b**-1 give slightly different answers, so Tensor and cupy array division gives slightly different answers

        self.data = cp.array(data, dtype = cp.float64)    # cupy handles data type conversion, cp.array() on a cp.array does nothing
        self.data = self.data if self.data.ndim >= 2 else cp.atleast_2d(self.data)
        self.grad = cp.zeros_like(self.data)    # zeros_like() uses the same data type as its input by default
        self.prev = set(children)    # tuples aka "()" are used for convenience as inputs, but need to be converted to set so elements can be added/removed later
        self.backwards = lambda: None    # lambda function (aka single-line function with no name) that does nothing, that will be filled in with a custom backward function based on the operation that created that Tensor
        self.op = op
    
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), '+')
    
        def backwards():
            self.grad += out.grad
            other.grad += out.grad
        
        out.backwards = backwards
        return out
    
    def __radd__(self, other):    # non-Tensor self + Tensor other failed, so interpreter tries Tensor self + non-Tensor other
        return self + other
    
    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), '*')
    
        def backwards():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        
        out.backwards = backwards
        return out
    
    def __rmul__(self, other):    # non-Tensor self * Tensor other failed, so interpreter tries Tensor self * non-Tensor other
        return self * other
        
    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self + -other
    
    def __rsub__(self, other):    # non-Tensor self - Tensor other failed, so interpreter tries Tensor self - non-Tensor other
        return -self + other
      
    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self * other**-1
    
    def __rtruediv__(self, other):    # non-Tensor self / Tensor other failed, so interpreter tries Tensor self / non-Tensor other
        return self**-1 * other
    
    def __pow__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data ** other.data, (self, other), '**')
    
        def backwards():
            self.grad += other.data * (self.data ** (other.data - 1)) * out.grad
            other.grad += (self.data ** other.data) * cp.log(self.data + ((self.data == 0) * math.ulp(0.0))) * out.grad
        
        out.backwards = backwards
        return out
    
    def __rpow__(self, other):    # non-Tensor self ** Tensor other failed, so interpreter tries Tensor self ** non-Tensor other
        other = other if isinstance(other, Tensor) else Tensor(other)
        return other ** self
    
    def __neg__(self):
        return self * -1