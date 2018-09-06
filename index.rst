.. xlearn_doc documentation master file, created by
   sphinx-quickstart on Sun Dec  3 18:43:51 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

欢迎使用 xLearn !
===============================

xLearn 是一款高性能的，易用的，并且可扩展的机器学习库，你可以用它来解决大规模机器学习问题，尤其是大规模稀疏数据机器学习问题。在近年来，
大规模稀疏数据机器学习被广泛应用在广告点击率预测、推荐系统等领域。如果你是 liblinear、libfm、libffm 的用户，那么现在 xLearn 将是你
更好的选择，因为 xLearn 几乎囊括了这些系统的全部功能，并且具有更好的性能和可扩展性。

.. image:: ./images/speed.png
    :width: 650  


快速开始
----------------------------------

我们接下来快速展示如何在一个小型样例 （Criteo 广告点击预测数据）上使用 xLearn 来解决二分类问题。  

安装 xLearn
^^^^^^^^^^^^^^

xLearn 最简单的安装方法是使用 ``pip`` . 下面的命令将会下载 xLearn 的源代码，并且在你的本地机器上编译和安装。 ::

    sudo pip install xlearn

上述安装过程可能会持续一段时间，请耐心等候。安装完成后，用户可以使用下面的代码来检测 xLearn 是否安装成功。 ::

  >>> import xlearn as xl
  >>> xl.hello()

如果安装成功，用户会看到如下显示 ::

  -------------------------------------------------------------------------
           _
          | |
     __  _| |     ___  __ _ _ __ _ __
     \ \/ / |    / _ \/ _` | '__| '_ \
      >  <| |___|  __/ (_| | |  | | | |
     /_/\_\_____/\___|\__,_|_|  |_| |_|

        xLearn   -- 0.31 Version --
  -------------------------------------------------------------------------


如果你在安装的过程中遇到了任何问题，或者你希望自己通过在 Github 上最新的源代码进行手动编译，或者你想使用 xLearn 的命令行界面，你可以
从这里（`Installation Guide`__）查看如何对 xLearn 进行手动编译和安装。

.. __: ./install/index.html

Python Demo
^^^^^^^^^^^^^^

Here is a simple Python demo no how to use xLearn for a binary classification problem:

.. code-block:: python

    import xlearn as xl

    # Training task
    ffm_model = xl.create_ffm()                # Use field-aware factorization machine (ffm)
    ffm_model.setTrain("./small_train.txt")    # Path of training data
    ffm_model.setValidate("./small_test.txt")  # Path of validation data

    # param:
    #  0. task: binary classification
    #  1. learning rate: 0.2
    #  2. regular lambda: 0.002
    #  3. evaluation metric: accuracy
    param = {'task':'binary', 'lr':0.2, 'lambda':0.002, 'metric':'acc'}

    # Start to train
    # The trained model will be stored in model.out
    ffm_model.fit(param, './model.out')

    # Prediction task
    ffm_model.setTest("./small_test.txt")  # Path of test data
    ffm_model.setSigmoid()                 # Convert output to 0-1

    # Start to predict
    # The output result will be stored in output.txt
    ffm_model.predict("./model.out", "./output.txt")

This example shows how to use *field-aware factorizations machine (ffm)* to solve a 
simple binary classification task. You can check out the demo data 
(``small_train.txt`` and ``small_test.txt``) from the path ``demo/classification/criteo_ctr``.


Link to the Other Helpful Resources
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