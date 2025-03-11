import cupy as cp

class Tensor:
    def __init__(self, data = [], grad = 0.0, op = ''):
        self.data = data
        self.grad = grad
        self.op = op
    
    def __add__(self, other):
        other = other if type(other) == Tensor else Tensor(other)
        out = Tensor(data = (self.data + other.data))
    
        def backwards():
            self.grad += out.grad
            other.grad += out.grad
        
        return out
    
a = [2, 3]
b = [5, 10]
c = 4

#a = -5
#b = 3
#c = 1

aVal = Tensor(a)
bVal = Tensor(b)

#a = cp.array(a).reshape((1,-1))
#b = cp.array(b).reshape((1,-1))

print("a:      | Passed == ", type(aVal) == Tensor)
print("b:      | Passed == ", type(bVal) == Tensor)
print("a.data: | Passed == ", aVal.data == a)
print("b.data: | Passed == ", bVal.data == b)
print("a+c:    | Passed == ", (aVal+c).data == a+c)
print("c+a:    | Passed == ", (c+aVal).data == c+a)
print("a-c:    | Passed == ", (aVal-c).data == a-c)
print("c-a:    | Passed == ", (c-aVal).data == c-a)
print("a*c:    | Passed == ", (aVal*c).data == a*c)
print("c*a:    | Passed == ", (c*aVal).data == c*a)
print("-a:     | Passed == ", (-aVal).data == -a)
print("a/c:    | Passed == ", (aVal/c).data == a/c)
print("c/a:    | Passed == ", (c/aVal).data == c/a)
print("a**c:   | Passed == ", (aVal**c).data == a**c)
print("c**a:   | Passed == ", (c**aVal).data == c**a)