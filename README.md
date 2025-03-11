<div align="center">

littlegrad3: For something between [tinygrad/tinygrad](https://github.com/tinygrad/tinygrad) and [karpathy/micrograd](https://github.com/karpathy/micrograd).

</div>

---

### REPOSITORY SETUP NOTES

INSTALLING CUPY LIBRARY [[source](https://docs.cupy.dev/en/stable/install.html)]
```
sudo apt install nvidia-cuda-toolkit
nvcc --version
pip install cupy-cuda12x (or 11x if nvcc version is 11.x)
```

GIT COMMIT PREREQUISITES [[source](https://docs.github.com/en/get-started/git-basics/setting-your-username-in-git)]
```
git config --global user.email X
git config --global user.name X
```

README FORMATTING [[source](https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax)]

not sure why, but readme HTML tags require 2 linebreaks in order to render readme formatting symbols, 1 linebreak prints everything inside like pure HTML :confused:

GITIGNORE FORMATTING[[source](https://git-scm.com/docs/gitignore#_examples)]

```
inside .gitignore:

*.pyc to ignore all pyc files
dir/ to ignore all files in dir directory
```


### PYTHON NOTES

PYTHON FILE STRUCTURE [[source](https://docs.python.org/3/tutorial/modules.html)]

there are two ways to interact with python files: executing them as a script and importing them as a module. both allow you to execute the code in the file, but importing sets the `__name__` variable to the file name without the `.py` extension, whereas executing as a script sets `__name__` to `"__main__"` (checking for this allows you to read in command line arguments).

packages are collections of modules, similar to directories of files (except formated like package.submodule instead of directory/file). like how different directories can have files with the same name, packages allow submodules to have the same name (like numpy.tensor and torch.tensor). to make sure that all packages/subpackages/submodules are imported correctly even if they have the same name, there needs to be an `__init__.py` file inside the package (even if it is blank, it still tells python that the parent directory is supposed to be a python package as opposed to something else).

PYTHON CLASSES [[source](https://docs.python.org/3/tutorial/classes.html)]

python classes allow you to bundle variables and functions into one unit to interact with. there are two methods of interaction: referencing an attribute (`MyClass.myVariable or MyClass.myFunction`) or instantiation (`myObject = MyClass(optionalArgument1, optionalArgument2)`). 

class instantiation creates an instance object that can only be interacted with through attribute references (`myObject.myVariable or myObject.myFunction`). variables referenced through objects are called "data attributes" and functions referenced through objects are called "methods". calling a method (`myObject.myFunction(optionalArgument1, optionalArgument2)`) actually calls the class function with the instance object prepended as the first argument (`myObject.myFunction(optionalArgument1, optionalArgument2) == MyClass.myFunction(myObject, optionalArgument1, optionalArgument2)`). in class function definitions this first argument is called `self` (not a requirement, but a convention). during class instantiation an `__init__()` method is called automatically if it is defined.

```python
>>> class Dog:    # based on source
...     kind = 'canine'         # class variable shared by all instances
...     def __init__(self, name):
...             self.name = name    # instance variable unique to each instance
...             self.tricks = []    # instance variable unique to each instance
...     def add_trick(self, trick):
...             self.tricks.append(trick)
... 
>>> Dog
<class '__main__.Dog'>
>>> Dog.kind
'canine'
>>> Dog.add_trick
<function Dog.add_trick at 0x793f5dd74f40>
>>> Dog.__init__
<function Dog.__init__ at 0x793f5f28c540>
>>> d = Dog('Fido')
>>> e = Dog('Buddy')
>>> d.add_trick('roll over')
>>> Dog.add_trick(e, 'play dead')
>>> d.kind, d.name, d.tricks
('canine', 'Fido', ['roll over'])
>>> e.kind, e.name, e.tricks
('canine', 'Buddy', ['play dead'])
>>> d.add_trick
<bound method Dog.add_trick of <__main__.Dog object at 0x793f5f28b650>>
>>> d.__init__
<bound method Dog.__init__ of <__main__.Dog object at 0x793f5f28b650>>
>>> d
<__main__.Dog object at 0x793f5f28b650>

```

PYCACHE & PYC FILES [[source](https://docs.python.org/3/tutorial/modules.html#compiled-python-files)]

To speed up loading modules, Python caches the compiled version of each module in the `__pycache__` directory under the name `module.version.pyc`, where the version is usually the python version number.

### MACHINE LEARNING NOTES

MACHINE LEARNING THEORY [[source](https://www.youtube.com/watch?v=VMj-3S1tku0)]

this is the code repository for a machine learning library, which allows a machine (computer) to learn to complete a task without explicit programming. the library does this by implementing an artificial neural network (multilayer perceptron).

a neural network is a mathematical expression that takes in data and parameters in order to return an output (this process is called "inference" or "forward pass"). 

during the learning (or training) process, another mathematical expression called a cost function takes in the network's output and compares it to a target output, returning a number that represents the distance between the network output and the target (called the cost).

A mathematical operation called partial differentiation is applied to the cost in order to calculate the partial contributions each parameter makes to the cost. This process is usually called "backpropagation" or "backward pass" and the partial contributions are known as the gradient. After differentiation, each parameter is changed according to its own gradient value such that the cost decreases as a result (this is called "gradient descent"). 

After the parameters are updated they are fed into the neural network along with more data for another forward pass, continuing the learning (or training) process. 

The cost function is chosen specifically so that, the lower it is, the closer the network is to returning the target values. A cost of 0 would mean the network returns the exact target values; this most likely means that the network has simply memorized the target values (called overfitting) and will not do well in a real world use case with less predictable user input data.

MACHINE LEARNING IMPLEMENTATION [[source](https://github.com/karpathy/micrograd)] [[source](https://docs.cupy.dev/en/stable/overview.html)] [[source](https://en.wikipedia.org/wiki/Chain_rule)]

the steps involved in neural network training are: forward pass, cost calculation, backward pass, and gradient descent. 

cost calculation and gradient descent implementations are given in the test and example files of this library, but they are not too different from normal python programming. the main difficuly in implementing neural networks is designing efficient forward and backward passes. because of this, a new python class named Tensor was created. the Tensor class uses the CuPy array library to implement forward passes as a series of simple array operations on the GPU. For the backward pass, an algorithm called "automatic differentiation" is used to calculate the gradient for every value stored in a given Tensor.

automatic differentiation is based on a property of differentiation known as the Chain Rule. expressed mathematically in Leibniz's notation, the chain rule is `dz/dx = dz/dy * dy/dx`. because this library implements forward pass as a series of array operations, the overall gradient of the cost would simply be the product of the local gradients of each operation. during the backward pass, the derivative of the cost with respect to itself (`dz/dz`) is set to 1. this derivative is passed backwards through the neural network, getting multiplied by a local partial derivative (`dz/dy`, `dy/dx`) for every operation encountered in the forward pass. running the algorithm on a sequence of scalar arithmetic operations looks like this:

![visualization of automatic differentiation on scalar arithmetic operations](https://raw.githubusercontent.com/karpathy/micrograd/c911406e5ace8742e5841a7e0df113ecb5d54685/gout.svg)

if the same variable (parameter) is used in multiple intermediate functions, the partial derivative for that variable is the sum of the partial derivatives of the intermediate functions ([further reading](https://en.wikipedia.org/wiki/Chain_rule#Multivariable_case)).

### LIBRARY PROGRAMMING NOTES

TENSOR IMPLEMENTATION NOTES

a tensor is essentially a collection of two CuPy arrays--one to store values and one to store gradients. the tensor methods essentially supplement the CuPy array methods so that the gradients know how to backpropagate through them.

each tensor method needs to do 3 things: define a forward pass operation, define a backward pass operation, and keep track gradients (no adding a tensor to an integer, careless broadcasting, or anything else that would mess up the gradient path during backward pass)

there also needs to be an overall backward pass function that sorts all the local array operations into the appropriate sequence, sets the cost gradient to 1, and iteratively calls the backward method for each array operation until all parameter gradients are correctly calculated.