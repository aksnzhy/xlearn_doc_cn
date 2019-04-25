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

首先你需要编译源码得到两个可执行文件：``xlearn_train.exe`` 和 ``xlearn_predict.exe``，并且得到动态链接库 ``xlearn_api.dll``。 之后，需要安装 xLearn Python 包。

编译源代码
=======================

用户进入 DOS 控制台，输入命令: ::

  git clone https://github.com/aksnzhy/xlearn.git

  cd xlearn
  mkdir build
  cd build
  cmake -G "Visual Studio 15 Win64" ../
  "C:\Program Files (x86)\Microsoft Visual Studio\2017\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64
  MSBuild xLearn.sln /p:Configuration=Release
  
**注意:** 你需要将路径 ``"C:\Program Files (x86)\Microsoft Visual Studio\2017\Enterprise\VC\Auxiliary\Build\vcvarsall.bat"``
替换成你自己的 VS 安装路径.

例如，默认情况下 VS 的路径为 ``"C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvarsall.bat"``.

如果安装成果, 用户可以在 `build\Release`` 路径下看到 ``xlearn_train.exe`` 和 ``xlearn_predict.exe`` 两个可执行文件。

用户可以通过如下命令进行测试: ::

  run_example.bat

从 Visual Studio解决方案编译源码
=======================
这个编译方法是上面“编译源码”方法的一个备用选择，如果你已经使用上面方法进行编译，你可以跳过这个部分。

我们为用户提供了Visual Studio解决方案，这些文件在xLearn项目根目录的windows目录下面，用户可以直接使用``xLearn.sln``进行源代码。

There are three vs project in this solution: xlearn_train, xlearn_test, xlearn_api, respectively relation to build executable train,predict entry program and DLL(dynamic link library) API for windows.
这个解决方案包括三个项目：``xlearn_train``, ``xlearn_test``, ``xlearn_api``，分别对应产生xLearn的训练、预测的可执行文件和动态链接库。

用户需要保证所使用的VS的工具平台版本在v141及其之上。

**注意：** 从这个解决方案编译得到的可执行文件和动态链接库会和使用cmake构建、编译得到的有所不同，这是因为它们构建结构不相同。

安装 Python 包
=======================

用户可以通过如下命令安装 Python 包: ::

  cd python-package
  python setup.py install 

然后通过如下命令对安装进行测试: ::

  cd ../
  python test_python.py

一键安装
=======================

用户可以通过 ``build.bat`` 脚本来对 xLearn 进行一键安装: ::

  git clone https://github.com/aksnzhy/xlearn.git

  cd xlearn
  build.bat

从pip安装
^^^^^^^^^^^^^^^^^^^^^^^^

我们现在提供了windows平台下的二进制Python包，它支持64位Python的一下版本：``2.7, 3.4, 3.5, 3.6, 3.7``。

用户可以从 release_ 栏（xLearn项目主页）下载，然后用 ``pip`` 命令安装下载下来的后缀为 ``.whl`` 的二进制安装包文件。

.. _release: https://github.com/aksnzhy/xlearn/releases


用户可以通过如下命令检查是 xLearn 是否安装成功: ::

  >>> import xlearn as xl
  >>> xl.hello()

如果安装成功，你可以看到: ::

  -------------------------------------------------------------------------
           _
          | |
     __  _| |     ___  __ _ _ __ _ __
     \ \/ / |    / _ \/ _` | '__| '_ \
      >  <| |___|  __/ (_| | |  | | | |
     /_/\_\_____/\___|\__,_|_|  |_| |_|

        xLearn   -- 0.44 Version --
  -------------------------------------------------------------------------
