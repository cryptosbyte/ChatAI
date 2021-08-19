# AI

This is an artificial intelligence that uses [NLTK](http://www.nltk.org/), [PyTorch](https://pytorch.org/) & [Numpy](https://numpy.org/) tools in it's code.

## Usage

Read the `setup.sh` file and make sure that the part of the script that installs & setups PyTorch and Python [VEnv](https://docs.python.org/3/library/venv.html) is either commented or uncommented to your preferred choice (i.e. comment it if you don't want it to run and vice versa).

## `setup.sh` Run Down
- Sets up PyTorch's folder
- Installs Python Virtual Env (VEnv)
- Downloads & Activates PyTorch & Torchvision & Deactivates
- Goes to `~/AI` folder (default folder of the code)
- Activates PyTorch Environment
- Installs Dependencies
    - NLTK, PyTorch & Numpy
- Downloads `punkt` from `nltk` package
    - using Python3 CLI `-c` option
- Trains AI 
    - (converts `intents.json` into `data.pth`)
- Runs AI

*Inspired by [python-engineer](https://github.com/python-engineer/python-fun) & Refactored code by [cryptosbyte](https://github.com/cryptosbyte)*