Installation
============

You have to take following steps to use ReNom in your environment.

1. Install the python(we cofirm the operation in python 2.7 and python 3.4)
2. Install the ReNom environment

First, you have to install the python.
There are many web pages that explain how to intall the python.
And, you can download ReNom from following link.

URL: https://github.com/ReNom-dev-team/ReNom

If you already installed the GPU environments.

.. code-block:: sh

   git clone https://github.com/ReNom-dev-team/ReNom.git
   cd ReNom
   python setup.py build_ext -f -i
   pip install -e .

If you did not set the GPU environments.

.. code-block:: sh

   git clone https://github.com/ReNom-dev-team/ReNom.git
   cd ReNom
   pip install -e .

**Requirements**

ReNom requires following libraries.

- Linux / Ubuntu
- Python 2.7, 3.4
- Numpy 1.13.0, 1.12.1 http://www.numpy.org/
- cuDNN 5.1 https://developer.nvidia.com/cudnn
- CUDA ToolKit 8.0 https://developer.nvidia.com/cuda-toolkit
- bottle 0.12.13 https://bottlepy.org/docs/dev/
- matplotlib 2.0.2 https://matplotlib.org
- networkx 1.11 https://networkx.github.io
- pandas 0.20.3 http://pandas.pydata.org
- scikit-learn 0.18.2 http://scikit-learn.org/stable/
- scipy https://www.scipy.org
