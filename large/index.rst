xLearn 大规模机器学习
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

我们在这一节里主要展示如何使用 xLearn 来处理大规模机器学习问题。近年来，快速增长的海量数据为机器学习任务带来了挑战。例如，我们的数据集可能会用数千亿条训练样本，这些数据是不可能被存放在单台计算机的内存中的。正因如此，我们在设计 xLearn 时专门考虑了如何解决大规模数据的机器学习训练功能。首先，xLearn 可以支持外村计算，通过利用单台计算机的磁盘来处理 TB 量级的数据训练任务。此外，xLearn 可以通过基于参数服务器的分布式架构来进行多机分布式训练。

外存计算
--------------------------------

外存计算适用于那些数据量过大不能被内存装下，但是可以被磁盘等外部存储设备装下的情况。通常情况下，单台机器的内存容量从几个 GB 到几百个 GB 不等。然而，当前的服务器外存容量通常可以很容易达到几个 TB. 外存计算的核心是通过 mini-batch 的方法，在每一次的计算时只读取一小部分数据进入内存，增量式地学习所有的训练数据。外存计算需要用户设定合适的 mini-batch-size.

.. image:: ../images/out-of-core.png
    :width: 500   

命令行接口
===================================================

在 xLearn 中，用户可以通过设置 ``--disk`` 选项来进行外存计算。例如: ::

    ./xlearn_train ./big_data.txt -s 2 --disk

   Epoch      Train log_loss     Time cost (sec)
       1            0.483997                4.41
       2            0.466553                4.56
       3            0.458234                4.88
       4            0.451463                4.77
       5            0.445169                4.79
       6            0.438834                4.71
       7            0.432173                4.84
       8            0.424904                4.91
       9            0.416855                5.03
      10            0.407846                4.53

在上述示例中，xLearn 需要花费将近 4.5 进行每一个 epoch 的训练任务。如果我们取消 ``--disk`` 选项，xLearn 的训练速度会变快: ::

    ./xlearn_train ./big_data.txt -s 2

    Epoch      Train log_loss     Time cost (sec)
        1            0.484022                1.65
        2            0.466452                1.64
        3            0.458112                1.64
        4            0.451371                1.76
        5            0.445040                1.83
        6            0.438680                1.92
        7            0.432007                1.99
        8            0.424695                1.95
        9            0.416579                1.96
       10            0.407518                2.11

这一次，每一个 epoch 的训练时间变成了 ``1.8`` 秒。我们还可以通过 ``-block`` 选项来设置外存计算的内存 block 大小 （MB）。

Python 接口
===================================================

In Python, users can use ``setOnDisk`` API to perform *out-of-core* learning. For example: ::

    import xlearn as xl

    # Training task
    ffm_model = xl.create_ffm() # Use field-aware factorization machine

    # On-disk training
    ffm_model.setOnDisk()

    ffm_model.setTrain("./small_train.txt")  # Training data
    ffm_model.setValidate("./small_test.txt")  # Validation data

    # param:
    #  0. binary classification
    #  1. learning rate: 0.2
    #  2. regular lambda: 0.002
    #  3. evaluation metric: accuracy
    param = {'task':'binary', 'lr':0.2, 
             'lambda':0.002, 'metric':'acc'}

    # Start to train
    # The trained model will be stored in model.out
    ffm_model.fit(param, './model.out')

    # Prediction task
    ffm_model.setTest("./small_test.txt")  # Test data
    ffm_model.setSigmoid()  # Convert output to 0-1

    # Start to predict
    # The output result will be stored in output.txt
    ffm_model.predict("./model.out", "./output.txt")

We can set the block size for on-disk training by using ``block_size`` parameter.

R 接口
===================================================

The R guide is coming soon.

分布式计算 （参数服务器架构）
--------------------------------

As we mentioned before, for some large-scale machine challenges like computational advertising, we
focus on the problem with potentially trillions of training examples and billions of model parameters,
both of which cannot fit into the memory of a single machine, which brings the *scalability challenge*
for users and system designer. For this challenge, parallelizing the training process across machines has 
become a prerequisite.

The *Parameter Server* (PS) framework has emerged as an efficient approach to solve the “big model” machine learning 
challenge recently. Under this framework, both the training data and workloads are spread across worker nodes, while 
the server nodes maintain the globally shared model pa- rameters. The following figure demonstrates the architecture 
of the PS framework. 

.. image:: ../images/ps.png
    :width: 500   

As we can see, the *Parameter Server* provides two concise APIs for users. 

*Push* sends a vector of (key, value) paris
to the server nodes. To be more specific – in the distributed gradient descent, the worker nodes might send the locally 
computed gradients to servers. Due to the data sparsity, only a part the gradients is non-zero. Often it is desirable to 
present the gradient as a list of (key, value) pairs, where the feature index is the key and the according gradient item 
is value.

*Pull* requests the values associated with a list of keys, which will get the newest parameters from the server nodes. This 
is particularly useful whenever the main memory of a single worker cannot hold a full model. Instead, workers prefetch the 
model entries relevant for solving the model only when needed.

The distributed training guide for xLearn is coming soon.
