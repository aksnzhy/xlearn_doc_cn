.. xlearn_doc documentation master file, created by
   sphinx-quickstart on Sun Dec  3 18:43:51 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

欢迎使用 xLearn !
===============================

xLearn 是一款高性能的，易用的，并且可扩展的机器学习算法库，你可以用它来解决大规模机器学习问题，尤其是大规模稀疏数据机器学习问题。在近年来，大规模稀疏数据机器学习算法被广泛应用在各种领域，例如广告点击率预测、推荐系统等。如果你是 liblinear、libfm、libffm 的用户，那么现在 xLearn 将会是你更好的选择，因为 xLearn 几乎囊括了这些系统的全部功能，并且具有更好的性能，易用性，以及可扩展性。

.. image:: ./images/speed.png
    :width: 650  

快速开始
----------------------------------

我们接下来展示如何在一个小型数据样例 (Criteo 广告点击预测数据) 上使用 xLearn 来解决二分类问题。在这个问题里，机器学习算法需要判断当前用户是否会点击给定的广告。

安装 xLearn
^^^^^^^^^^^^^

xLearn 最简单的安装方法是使用 ``pip`` 安装工具. 下面的命令会下载 xLearn 的源代码，并且在用户的本地机器上进行编译和安装。 ::

    sudo pip install xlearn

上述安装过程可能会持续一段时间，请耐心等候。安装完成后，用户可以使用下面的代码来检测 xLearn 是否安装成功。 ::

  >>> import xlearn as xl
  >>> xl.hello()

如果安装成功，用户会看到如下显示: ::

  -------------------------------------------------------------------------
           _
          | |
     __  _| |     ___  __ _ _ __ _ __
     \ \/ / |    / _ \/ _` | '__| '_ \
      >  <| |___|  __/ (_| | |  | | | |
     /_/\_\_____/\___|\__,_|_|  |_| |_|

        xLearn   -- 0.40 Version --
  -------------------------------------------------------------------------

如果你在安装的过程中遇到了任何问题，或者你希望自己通过在 `Github`__ 上最新的源代码进行手动编译，或者你想使用 xLearn 的命令行接口，你可以从这里 (`Installation Guide`__) 查看如何对 xLearn 进行从源码的手动编译和安装。

.. __: https://github.com/aksnzhy/xlearn
.. __: ./install/index.html

Python 样例
^^^^^^^^^^^^^^

下面的 Python 代码展示了如何使用 xLearn 的 *FFM* 算法来处理机器学习二分类任务： 

.. code-block:: python

    import xlearn as xl

    # Training task
    ffm_model = xl.create_ffm()                # Use field-aware factorization machine (ffm)
    ffm_model.setTrain("./small_train.txt")    # Set the path of training dataset
    ffm_model.setValidate("./small_test.txt")  # Set the path of validation dataset

    # Parameters:
    #  0. task: binary classification
    #  1. learning rate: 0.2
    #  2. regular lambda: 0.002
    #  3. evaluation metric: accuracy
    param = {'task':'binary', 'lr':0.2, 'lambda':0.002, 'metric':'acc'}

    # Start to train
    # The trained model will be stored in model.out
    ffm_model.fit(param, './model.out')

    # Prediction task
    ffm_model.setTest("./small_test.txt")  # Set the path of test dataset
    ffm_model.setSigmoid()                 # Convert output to 0-1

    # Start to predict
    # The output result will be stored in output.txt
    ffm_model.predict("./model.out", "./output.txt")

上述样例通过使用 *field-aware factorization machines (FFM)* 来解决一个简单的二分类任务。用户可以在 ``demo/classification/criteo_ctr`` 
路径下找到我们所使用的样例数据 (``small_train.txt`` 和 ``small_test.txt``).

其他资源链接
----------------------------------------

.. toctree::
   :glob:
   :maxdepth: 1

   self
   install/index
   cli_api/index
   python_api/index
   R_api/index
   all_api/index
   large/index
   demo/index
   tutorial/index