<div align="center">

littlegrad3: For something between [tinygrad/tinygrad](https://github.com/tinygrad/tinygrad) and [karpathy/micrograd](https://github.com/karpathy/micrograd).

</div>

---

NOTES:

File Structure:

the python [docs](https://docs.python.org/3/tutorial/modules.html) describe two ways to interact with python files: executing them as a script and importing them as a module. Both allow you to execute the code in the file, but importing sets the `__name__` variable to the file name without the `.py` extension, whereas executing as a script sets `__name__` to `"__main__"` (checking for this allows you to read in command line arguments).

packages are collections of modules, similar to directories of files (except formated like package.submodule instead of directory/file). Like how different directories can have files with the same name, packages allow submodules to have the same name (like numpy.tensor and torch.tensor). to make sure that all packages/subpackages/submodules are imported correctly even if they have the same name, there needs to be an `__init__.py` file inside the package (even if it is blank, it still tells python that the parent directory is supposed to be a python package as opposed to something else).

[Installing](https://docs.cupy.dev/en/stable/install.html) CuPy:
```
sudo apt install nvidia-cuda-toolkit
nvcc --version
pip install cupy-cuda12x (or 11x if nvcc version is 11.x)
```

Git commit [prerequisites](https://docs.github.com/en/get-started/git-basics/setting-your-username-in-git):
```
git config --global user.email X
git config --global user.name X
```