<div align="center">

littlegrad3: for something between [tinygrad/tinygrad](https://github.com/tinygrad/tinygrad) and [karpathy/micrograd](https://github.com/karpathy/micrograd).

this is far from the best deep learning framework, but it is a deep learning framework. it uses the [CuPy](https://cupy.dev/) array library as a backend to perform GPU accelerated computing, and achieved ~98% accuracy on [kaggle](https://www.kaggle.com/competitions/digit-recognizer)'s MNIST test set with <5s of training time with [mnist.py](/mnist.py):

![/misc-resources/98%20kaggle.png](/misc-resources/98%20kaggle.png)

![/misc-resources/98%20vscode.png](/misc-resources/98%20vscode.png)

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

GITIGNORE FORMATTING [[source](https://git-scm.com/docs/gitignore#_examples)]

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

LAMBDA FUNCTIONS IN TENSOR INITIALIZATION [[source](https://www.w3schools.com/python/python_lambda.asp)]

a lambda function is a function with no name that can only execute a single expression. the entire function can be written in one line, for example `lambda a, b: a+b` which returns the sum of its two arguments. the Tensor `__init__()` method contains a `lamda: None` function as a placeholder, which is later replaced with an actual function that backpropagates a gradient through a specific array operation.

TENSOR OBJECTS [[source](https://github.com/karpathy/micrograd)]

a Tensor object contains two arrays--one for values and one for gradients (each value has a corresponding gradient). a Tensor operation accounts for both arrays--first the Tensor operation calls a CuPy array operation to act on the value arrays of the input Tensors (forward pass). then a backpropagation function is stored in the output Tensor that will calculate the correct gradients for the input Tensors during the backward pass.

a higher-level backpropagation function can be called on the final output of a group of Tensor operations (usually the cost). this sets the final output Tensor's gradients to 1.0, sorts all the Tensors involved (using a topological sort algorithm) and then iteratively calls the local backpropogation function of each Tensor until all the involved Tensors' gradients have been calculated.

to maintain the pairing between value and gradient the Tensor.make_compatible() function was defined. this function replaces normal array broadcasting (which does not broadcast the gradients), and manually broadcasts both the value and gradient arrays so array operations do not affect gradient calculation.

TOPOLOGICAL SORT

if higher-level backpropagation simply started from the final output Tensor and iteratively called local backpropagation on previous Tensors, then a Tensor's local backpropagation function would get called during after its first gradient update, and not its last. This means that if a Tensor receives multiple gradient updates from multiple Tensor operations, only the first Tensor operation would be sucessfully backpropagated. In order to calculate gradients correctly, a Tensor needs to call its local backpropogation function during its last gradient update rather than its first, which means Tensors need to be added to a list during their first appearence in the forward pass. the topological sort algorithm recursively searches previous Tensors (children) until it arrives at the starting Tensors. only then are Tensors added to a list (and also a set to avoid duplicates later in the tree). after nodes are added from the beginning, the list is reversed, and then the local backwards functions are called starting with the final output Tensor. this way, Tensors pass their own gradients backwards when they have the correct gradient, and not before.

RATIONALE FOR USING SETS [[source](https://www.youtube.com/watch?v=VMj-3S1tku0)]

Tensors are defined with their "children" argument being a tuple. this is because of the convenience of using paretheses to create them. however, tuples cannot be used for a backward pass because they are completely unchangable--a Tensor stored as a tuple element will not be able to update its gradient array for backpropagation. because of this, the "children" tuple gets converted to a set during Tensor initization. list could have also worked, but they allow duplicates, are less convenient to create than tuples ([] vs ()), and are less efficient than sets for backpropagation.

MISC NOTES
- NOTE: for some reason in CuPy 1/b and b**-1 give slightly different answers, so Tensor and cupy array division gives slightly different answers
- NOTE: fixed matplotlib error using `pip install PyQt5` from https://stackoverflow.com/questions/77507580/userwarning-figurecanvasagg-is-non-interactive-and-thus-cannot-be-shown-plt-sh
- NOTE: mnist.py data augmentation and normalization based on https://medium.com/data-science/going-beyond-99-mnist-handwritten-digits-recognition-cfff96337392