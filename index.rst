.. xlearn_doc documentation master file, created by
   sphinx-quickstart on Sun Dec  3 18:43:51 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

欢迎使用 xLearn !
===============================

xLearn 是一款高性能，易用的，并且可扩展的机器学习库，你可以用它来解决大规模机器学习问题，尤其是大规模稀疏数据机器学习问题。在近年来，
大规模稀疏数据机器学习被广泛应用在广告点击率预测、推荐系统等领域。如果你是 liblinear、libfm、libffm 的用户，那么现在 xLearn 将是你
更好的选择，因为 xLearn 几乎囊括了这些系统的全部功能，并且具有更好的性能和可扩展性。

.. image:: ./images/speed.png
    :width: 650  


A Quick Example
----------------------------------

This is a quick start tutorial showing snippets for you to quickly try out xLearn on a small demo dataset 
(Criteo CTR prediction) for a binary classification task.

Installation
^^^^^^^^^^^^^^

The easiest way to install xLearn Python package is to use ``pip``. The following command will 
download the xLearn source code, build and install python package on your locally machine. ::

    sudo pip install xlearn

The installation process will take a while to complete. After that you can use the following 
script in your python shell to check whether the xLearn has been installed successfully:

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


If you meet any installation problem, or you want to build the latest code from Github, or you want to use 
the xLearn command line instead of the python API, you can see how to build xLearn from source code 
in `Installation Guide`__.

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