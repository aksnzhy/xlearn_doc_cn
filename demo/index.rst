xLearn 样例程序
^^^^^^^^^^^^^^^^^^^^^^^^^^^

注意：这是用的所有数据集所有权属于原作者。

Criteo 在线广告预估
---------------------------

Kaggle 预测广告是否会被用户点击 (`链接`__)

Display advertising is a billion dollar effort and one of the central uses of machine learning on the Internet. 
However, its data and methods are usually kept under lock and key. In this research competition, CriteoLabs is 
sharing a week’s worth of data for you to develop models predicting ad click-through rate (CTR). Given a user 
and the page he is visiting, what is the probability that he will click on a given ad?

样例数据在： ``/demo/classification/criteo_ctr/``.

The follow code is the Python demo:

.. code-block:: python

    import xlearn as xl

    # Training task
    ffm_model = xl.create_ffm() # Use field-aware factorization machine
    ffm_model.setTrain("./small_train.txt")  # Training data
    ffm_model.setValidate("./small_test.txt")  # Validation data

    # param:
    #  0. binary classification
    #  1. learning rate: 0.2
    #  2. regular lambda: 0.002
    #  3. evaluation metric: accuracy
    param = {'task':'binary', 'lr':0.2, 'lambda':0.002, 'metric':'acc'}

    # Start to train
    # The trained model will be stored in model.out
    ffm_model.fit(param, './model.out')

    # Prediction task
    ffm_model.setTest("./small_test.txt")  # Test data
    ffm_model.setSigmoid()  # Convert output to 0-1

    # Start to predict
    # The output result will be stored in output.txt
    ffm_model.predict("./model.out", "./output.txt")

蘑菇分类
---------------------------

数据集来自 UCI Machine Learning Repositpry (`链接`__)

This data set includes descriptions of hypothetical samples corresponding to 23 species of gilled mushrooms in 
the Agaricus and Lepiota Family (pp. 500-525). Each species is identified as definitely edible, definitely poisonous, 
or of unknown edibility and not recommended. This latter class was combined with the poisonous one. The Guide clearly 
states that there is no simple rule for determining the edibility of a mushroom; no rule like *leaflets three, let it be*
for Poisonous Oak and Ivy.

样例数据在： ``/demo/classification/mushroom/``.

The follow code is the Python demo:

.. code-block:: python

    import xlearn as xl

    # Training task
    linear_model = xl.create_linear()  # Use linear model
    linear_model.setTrain("./agaricus_train.txt")  # Training data
    linear_model.setValidate("./agaricus_test.txt")  # Validation data

    # param:
    #  0. binary classification
    #  1. learning rate: 0.2
    #  2. lambda: 0.002
    #  3. evaluation metric: accuarcy
    #  4. use sgd optimization method
    param = {'task':'binary', 'lr':0.2, 
             'lambda':0.002, 'metric':'acc', 
             'opt':'sgd'}

    # Start to train
    # The trained model will be stored in model.out
    linear_model.fit(param, './model.out')

    # Prediction task
    linear_model.setTest("./agaricus_test.txt")  # Test data
    linear_model.setSigmoid()  # Convert output to 0-1

    # Start to predict
    # The output result will be stored in output.txt
    linear_model.predict("./model.out", "./output.txt")

泰塔尼克生还预测
-----------------------------

This challenge comes from the Kaggle. In this challenge, we ask you to complete the analysis of what sorts of people 
were likely to survive. In particular, we ask you to apply the tools of machine learning to predict which passengers 
survived the tragedy. (`链接`__)

You can find the data used in this demo in the path ``/demo/classification/titanic/``.

The follow code is the Python demo:

.. code-block:: python

    import xlearn as xl

    # Training task
    fm_model = xl.create_fm()  # Use factorization machine
    fm_model.setTrain("./titanic_train.txt")  # Training data

    # param:
    #  0. Binary classification task
    #  1. learning rate: 0.2
    #  2. lambda: 0.002
    #  3. metric: accuracy
    param = {'task':'binary', 'lr':0.2, 'lambda':0.002, 'metric':'acc'}

    # Use cross-validation
    fm_model.cv(param)

房价预测
-----------------------------

This demo shows how to use xLearn to solve the regression problem, and it comes from the Kaggle. The Ames 
Housing dataset was compiled by Dean De Cock for use in data science education. It's an incredible alternative 
for data scientists looking for a modernized and expanded version of the often cited Boston 
Housing dataset. (`链接`__)

样例数据在： ``/demo/regression/house_price/``.

The follow code is the Python demo:

.. code-block:: python

    import xlearn as xl

    # Training task
    ffm_model = xl.create_fm()  # Use factorization machine
    ffm_model.setTrain("./house_price_train.txt")  # Training data

    # param:
    #  0. Binary task
    #  1. learning rate: 0.2
    #  2. regular lambda: 0.002
    #  4. evaluation metric: rmse
    param = {'task':'reg', 'lr':0.2, 'lambda':0.002, 'metric':'rmse'}

    # Use cross-validation
    ffm_model.cv(param)

More Demo in xLearn is coming soon.

.. __: https://www.kaggle.com/c/criteo-display-ad-challenge
.. __: https://archive.ics.uci.edu/ml/datasets/Mushroom
.. __: https://www.kaggle.com/c/titanic
.. __: https://www.kaggle.com/c/house-prices-advanced-regression-techniques
