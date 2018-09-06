详细安装指南
----------------------------------

目前 xLearn 可以支持 Linux 和 Mac OS X 平台，我们将在后续支持 Windows 平台。这个页面介绍了如何通过 ``pip`` 安装 xLearn，并且详细介绍了如何
手动编译并安装 xLearn 源代码。无论你使用哪种方法安装 xLearn，请确保你的机器上已经安装了支持 C++11 的编译器，例如 ``GCC`` 或者 ``Clang`` 。
除此之外，您还需要提前安装好 ``CMake``.

安装 GCC 或 Clang
^^^^^^^^^^^^^^^^^^^^^^^^

如果你已经安装了支持 C++ 11 的编译器，请忽略此节内容。

* 在 Cygwin 上, 运行 ``setup.exe`` 并安装 ``gcc`` 和 ``binutils``.
* 在 Debian/Ubuntu Linux 上, 输入如下命令: ::

      sudo apt-get install gcc binutils 

  安装 GCC (或者 Clang) :: 

      sudo apt-get install clang 

* 在 FreeBSD 上, 输入以下命令安装 Clang :: 

      sudo pkg_add -r clang 

* 在 Mac OS X, 安装 ``XCode`` 来获得 Clang.


安装 CMake
^^^^^^^^^^^^^^^^^^^^^^^^

如果你已经安装了 CMake，请忽略此节内容。

* 在 Cygwin 上, 运行 ``setup.exe`` 并安装 cmake.
* 在 Debian/Ubuntu Linux 上, 输入以下命令安装 cmake: ::

      sudo apt-get install cmake

* 在 FreeBSD 上, 输入以下命令: ::
   
      sudo pkg_add -r cmake

在 Mac OS X, 如果你安装了 ``homebrew``, 输入以下命令 :: 

     brew install cmake

或者你安装了 ``MacPorts``, 输入以下命令 :: 

     sudo port install cmake


通过 pip 安装 xLearn
^^^^^^^^^^^^^^^^^^^^^^^^

安装 xLearn 最简单的方法是使用 ``pip``. 如下命令会下载 xLearn 源代码，并在你的本地计算机进行编译和安转 ::

    sudo pip install xlearn

上述安装过程可能会持续一段时间，请耐心等候。安装完成后，用户可以使用下面的代码来检测 xLearn 是否安装成功。 ::

  >>> import xlearn as xl
  >>> xl.hello()

如果安装成功，你会看到如下显示: ::

  -------------------------------------------------------------------------
           _
          | |
     __  _| |     ___  __ _ _ __ _ __
     \ \/ / |    / _ \/ _` | '__| '_ \
      >  <| |___|  __/ (_| | |  | | | |
     /_/\_\_____/\___|\__,_|_|  |_| |_|

        xLearn   -- 0.31 Version --
  -------------------------------------------------------------------------

如果你在安装的过程中遇到了任何问题，或者你希望自己通过在 `Github`__ 上最新的源代码进行手动编译，或者你想使用 xLearn 的命令行界面，下面的
章节将会介绍如何从源码手动编译并安装 xLearn。*我们强烈建议你尝试从源码编译安装 xLearn*。

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
