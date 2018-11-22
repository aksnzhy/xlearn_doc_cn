详细安装指南
----------------------------------

目前 xLearn 可以支持 Linux 和 Mac OS X 平台，我们将在后续支持 Windows 平台。这一节主要介绍了如何通过 ``pip`` 工具安装 xLearn，并且详细介绍了如何
通过源码手动编译并安装 xLearn. 无论你使用哪种方法安装 xLearn，请确保你的机器上已经安装了支持 C++11 的编译器，例如 ``GCC`` 或者 ``Clang``.
除此之外，用户还需要提前安装好 ``CMake`` 编译工具.

安装 GCC 或 Clang
^^^^^^^^^^^^^^^^^^^^^^^^

*如果你已经安装了支持 C++ 11 的编译器，请忽略此节内容。*

* 在 Cygwin 上, 运行 ``setup.exe`` 并安装 ``gcc`` 和 ``binutils``.
* 在 Debian/Ubuntu Linux 上, 输入如下命令: ::

      sudo apt-get install gcc binutils 

  安装 GCC (或者 Clang) :: 

      sudo apt-get install clang 

* 在 FreeBSD 上, 输入以下命令安装 Clang: :: 

      sudo pkg_add -r clang 

* 在 Mac OS X, 安装 ``XCode`` 来获得 Clang.


安装 CMake
^^^^^^^^^^^^^^^^^^^^^^^^

*如果你已经安装了 CMake，请忽略此节内容。*

* 在 Cygwin 上, 运行 ``setup.exe`` 并安装 cmake.
* 在 Debian/Ubuntu Linux 上, 输入以下命令安装 cmake: ::

      sudo apt-get install cmake

* 在 FreeBSD 上, 输入以下命令: ::
   
      sudo pkg_add -r cmake

在 Mac OS X, 如果你安装了 ``homebrew``, 输入以下命令: :: 

     brew install cmake

或者你安装了 ``MacPorts``, 输入以下命令: :: 

     sudo port install cmake

.. __: https://github.com/aksnzhy/xlearn

从源码安装 xLearn
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

从源码安装 xLearn 分为两个步骤：

首先，我们需要编译 xLearn 得到 ``xlearn_train`` 和 ``xlearn_predict`` 这两个可执行文件。除此之外，我们还需要得到 ``libxlearn_api.so`` (Linux 平台) 和 ``libxlearn_api.dylib`` (Mac OS X 平台) 这两个动态链接库 (用来进行 Python 调用)。随后，用户可以安装 xLearn Python Package.

编译 xLearn
===========

用户从 Github 上 clone 下 xLearn 源代码: ::

  git clone https://github.com/aksnzhy/xlearn.git

  cd xlearn
  mkdir build
  cd build
  cmake ../
  make

如果编译成功，用户将在 build 文件夹下看到 ``xlearn_train`` 和 ``xlearn_predict`` 这两个可执行文件。用户可以通过如下命令检查 xLearn 是否安装成功: ::

  ./run_example.sh

安装 Python Package
====================

之后，你就可以通过 ``install-python.sh`` 脚本来安装 xLearn Python 包: ::

  cd python-package
  sudo ./install-python.sh

用户可以通过如下命令检测 xLearn Python 库是否安装成功: ::

  cd ../
  python run_demo_ctr.py

一键安装脚本
============

我们已经写好了一个脚本 ``build.sh`` 来帮助用户做上述所有的安装工作。

用户只需要从 Github 上 clone 下 xLearn 源代码: ::

  git clone https://github.com/aksnzhy/xlearn.git

然后通过以下命令进行编译和安装: ::

  cd xlearn
  sudo ./build.sh

在安装过程中用户可能会被要求输入管理员账户密码。

通过 pip 安装 xLearn
^^^^^^^^^^^^^^^^^^^^^^^^

安装 xLearn 最简单的方法是使用 ``pip`` 安装工具. 如下命令会下载 xLearn 源代码，并在你的本地计算机进行编译和安装工作，该方法使用的前提是你已经安装了 xLearn 所需的开发环境，例如 C++11 和 CMake: ::

    sudo pip install xlearn

上述安装过程可能会持续一段时间，请耐心等候。安装完成后，用户可以使用下面的代码来检测 xLearn 是否安装成功: ::

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

        xLearn   -- 0.38 Version --
  -------------------------------------------------------------------------

安装 R 库
^^^^^^^^^^^^^^^^^^^^^^^^

The R package installation guide is coming soon.
