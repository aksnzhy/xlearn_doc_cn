详细安装指南
----------------------------------

目前 xLearn 可以支持 Linux 和 Mac OS X 平台，我们将在后续支持 Windows 平台。这个页面介绍了如何通过 pip 安装 xLearn，并且详细介绍了如何
手动编译并安装 xLearn 源代码。无论你使用哪种方法安装 xLearn，请确保你的机器上已经安装了支持 C++11 的编译器，例如 ``GCC`` 或者 ``Clang`` 。
除此之外，您还需要提前安装好 ``CMake``.

Install GCC or Clang
^^^^^^^^^^^^^^^^^^^^^^^^

If you have already installed your C++ compiler before, you can skip this step.

* On Cygwin, run ``setup.exe`` and install ``gcc`` and ``binutils``.
* On Debian/Ubuntu Linux, type the command: ::

      sudo apt-get install gcc binutils 

  to install GCC (or Clang) by using :: 

      sudo apt-get install clang 

* On FreeBSD, type the following command to install Clang :: 

      sudo pkg_add -r clang 

* On Mac OS X, install ``XCode`` gets you Clang.


Install CMake
^^^^^^^^^^^^^^^^^^^^^^^^

If you have already installed CMake before, you can skip this step.

To install CMake from binary packages:

* On Cygwin, run ``setup.exe`` and install cmake.
* On Debian/Ubuntu Linux, type the command to install cmake: ::

      sudo apt-get install cmake

* On FreeBSD, type the command: ::
   
      sudo pkg_add -r cmake

On Mac OS X, if you have ``homebrew``, you can use the command :: 

     brew install cmake

or if you have ``MacPorts``, run :: 

     sudo port install cmake

You won't want to have both Homebrew and MacPorts installed.


Install xLearn from pip
^^^^^^^^^^^^^^^^^^^^^^^^

The easiest way to install xLearn Python package is to use ``pip``. The following command will 
download the xLearn source code from pip and install Python package locally.  ::

    sudo pip install xlearn

The installation process will take a while to complete. After that, you can type the following script 
in your python shell to check whether the xLearn has been installed successfully:

>>> import xlearn as xl
>>> xl.hello()

You will see: ::

  -------------------------------------------------------------------------
           _
          | |
     __  _| |     ___  __ _ _ __ _ __
     \ \/ / |    / _ \/ _` | '__| '_ \
      >  <| |___|  __/ (_| | |  | | | |
     /_/\_\_____/\___|\__,_|_|  |_| |_|

        xLearn   -- 0.31 Version --
  -------------------------------------------------------------------------

If you want to build the latest code from `Github`__, or you want to use the xLearn command line 
instead of the Python API, you can see how to build xLearn from source code as follow. *We highly
recommend that you can build xLearn from source code.*

.. __: https://github.com/aksnzhy/xlearn

Install xLearn from Source Code
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Building xLearn from source code consists two steps.

First, you need to build the executable files (``xlearn_train`` and ``xlearn_predict``), as well as the 
shared library (``libxlearn_api.so`` for Linux or ``libxlearn_api.dylib`` for Mac OSX) from the C++ code.

Then, you can install the Python package through ``install-python.sh``.

Fortunately, we write a script ``build.sh`` to do all the cumbersome work for users. 

You just need to clone the code from github ::

  git clone https://github.com/aksnzhy/xlearn.git

and then build xLearn using the folloing commands: ::

  cd xlearn
  sudo ./build.sh

You may be asked to input your password during installation.

Test Your Building
^^^^^^^^^^^^^^^^^^^^^^^^

Now you can test your installation by using the following command: ::

  cd build
  ./run_example.sh

You can also test the Python package by using the following command: ::

  cd python-package/test
  python test_python.py

Install R Package
^^^^^^^^^^^^^^^^^^^^^^^^

The R package installation guide is coming soon.
