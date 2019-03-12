Windows 安装指南
----------------------------------

xLearn 支持 Windows 平台的安装和使用。本小节主要介绍如何在 Windows 平台安装并使用 xLearn 库。

安装 Visual Studio 2017
^^^^^^^^^^^^^^^^^^^^^^^^

*如果你的 Windows 系统已经安装过 Visual studio，你可以跳过这一步。*
 
从 https://visualstudio.microsoft.com/downloads/ 下载你所需要的 Visual studio （``vs_xxxx_xxxx.exe``）。之后，你可以通过 VS 的安装说明 （https://docs.microsoft.com/en-us/visualstudio/install/install-visual-studio?view=vs-2017.）进行安装。

安装 CMake
^^^^^^^^^^^^^^^^^^^^^^^^

*如果你的系统已经安装了 CMake，你可以跳过这一步*

从这里 https://cmake.org/download/ 下载最新版本 (至少 v3.10) CMake。请确保安装 CMake 后将其路径正确添加到你的系统路径。

从源码安装 xLearn
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

从源码安装 xLearn 包括了两个步骤：

First, you need to build the executable files (``xlearn_train.exe`` and ``xlearn_predict.exe``), as well as the 
shared library (``xlearn_api.dll`` for Windows) from the C++ code. After that, users need to install the xLearn Python Package.

Build from Source Code
=======================
First, users should enter DOS as Administrator. 
Then, users need to clone the code from github: ::

  git clone https://github.com/aksnzhy/xlearn.git

  cd xlearn
  mkdir build
  cd build
  cmake -G "Visual Studio 15 Win64" ../
  "C:\Program Files (x86)\Microsoft Visual Studio\2017\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64
  MSBuild xLearn.sln /p:Configuration=Release
  
**Note:** You should replace this path ``"C:\Program Files (x86)\Microsoft Visual Studio\2017\Enterprise\VC\Auxiliary\Build\vcvarsall.bat"``
to yourself installation path of VS2017.

Suppose you install the VS Community version, the path should be ``"C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvarsall.bat"``
if you install it in default path.

If the building is successful, users can find two executable files (``xlearn_train.exe`` and ``xlearn_predict.exe``) in the ``build\Release`` path. 
Users can test the installation by using the following command: ::

  run_example.bat

Install Python Package
=======================

Then, you can install the Python package through ``install-python.sh``: ::

  cd python-package
  python setup.py install 

You can also test the Python package by using the following command: ::

  cd ../
  python test_python.py

One-Button Building
=======================

We have already write a script ``build.bat`` to do all the cumbersome work for users, and users can just use the folloing commands: ::

  git clone https://github.com/aksnzhy/xlearn.git

  cd xlearn
  build.bat

You should make sure that you enter DOS as Administrator.

Install xLearn from pip
^^^^^^^^^^^^^^^^^^^^^^^^

We will update Python package for Windows soon later.

The installation process will take a while to complete. 
After that, you can type the following script in your python shell to check whether the xLearn has been installed successfully: ::

  >>> import xlearn as xl
  >>> xl.hello()

You will see the following message if the installation is successful: ::

  -------------------------------------------------------------------------
           _
          | |
     __  _| |     ___  __ _ _ __ _ __
     \ \/ / |    / _ \/ _` | '__| '_ \
      >  <| |___|  __/ (_| | |  | | | |
     /_/\_\_____/\___|\__,_|_|  |_| |_|

        xLearn   -- 0.42 Version --
  -------------------------------------------------------------------------
