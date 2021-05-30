# Working with Colab

You don't need to install enything, as you can work with the Google's Colab.
Just navigate to the [colab website](https://colab.research.google.com), and create/open a notebook.

# Installing Prerequisites for Local Work

## `Conda`

We will working in a [`conda`](https://docs.conda.io/projects/conda/en/latest/commands/install.html) environment.
You are welcome to use any installation, but in this document we will describe the basic installation of the `miniconda`.


1. Navigate to the [Miniconda website](https://docs.conda.io/en/latest/miniconda.html)
2. Download the binary that is the most appropriate for you
  - We will be using **Python 3.8**
  - Most modern OSs are **64-bit**
3. Once downloaded, install it -- you can keep all the settings as default.


Once installed, you can create a new conda environment.
In this example, we will call it `py38-ml`:

```shell
$ conda create -n py38-ml python=3.8
$ conda activate py38-ml
```

## Python packages

Once `conda` is installed, you should install some python packages. Below is the list of some packages that we will definitely use (there will be more though):

```shell
$ conda activate py38-ml
$ conda install numpy matplotlib seaborn
```

Also, we will use pytorch.
If you nVidia GPU:

```shell
$ conda install pytorch cudatoolkit=10.2 -c pytorch
```

Or if there is only a CPU:

```shell
$ conda install pytorch cpuinly -c pytorch
```
