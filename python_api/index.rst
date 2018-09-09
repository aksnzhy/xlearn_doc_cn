xLearn Python API 使用指南
^^^^^^^^^^^^^^^^^^^^^^^^^^^

xLearn 支持简单易用的 Python 接口。在使用之前，请确保你已经成功安装了 xLearn Python Package. 用户可以进入 Python shell，然后输入如下代码来检查是否成功安装 xLearn Python Package: ::

    >>> import xlearn as xl
    >>> xl.hello()

如果你已经成功安装了 xLearn Python Package，你将会看到: ::

  -------------------------------------------------------------------------
           _
          | |
     __  _| |     ___  __ _ _ __ _ __
     \ \/ / |    / _ \/ _` | '__| '_ \
      >  <| |___|  __/ (_| | |  | | | |
     /_/\_\_____/\___|\__,_|_|  |_| |_|

        xLearn   -- 0.31 Version --
  -------------------------------------------------------------------------

快速开始
----------------------------------------

如下代码展示如何使用 xLearn Python API，你可以在 ``demo/classification/criteo_ctr`` 路径下找到样例数据 (``small_train.txt`` and ``small_test.txt``):

.. code-block:: python

   import xlearn as xl

   # Training task
   ffm_model = xl.create_ffm()                # Use field-aware factorization machine (ffm)
   ffm_model.setTrain("./small_train.txt")    # Set the path of training data

   # parameter:
   #  0. task: binary classification
   #  1. learning rate : 0.2
   #  2. regular lambda : 0.002
   param = {'task':'binary', 'lr':0.2, 'lambda':0.002}
            
   # Train model
   ffm_model.fit(param, "./model.out")  

以下是 xLearn 的部分输出: ::

   ...
 [ ACTION     ] Start to train ...
 [------------] Epoch      Train log_loss     Time cost (sec)
 [   10%      ]     1            0.595881                0.00
 [   20%      ]     2            0.538845                0.00
 [   30%      ]     3            0.520051                0.00
 [   40%      ]     4            0.504366                0.00
 [   50%      ]     5            0.492811                0.00
 [   60%      ]     6            0.483286                0.00
 [   70%      ]     7            0.472567                0.00
 [   80%      ]     8            0.465035                0.00
 [   90%      ]     9            0.457047                0.00
 [  100%      ]    10            0.448725                0.00
 [ ACTION     ] Start to save model ...

在上述例子中，xLearn 使用 *feild-ware factorization machines (ffm)* 来解决一个机器学习二分类问题。如果想解决回归 (regression) 问题，用户可以通过将 ``task`` 参数设置为 ``reg`` 来实现: ::

    param = {'task':'reg', 'lr':0.2, 'lambda':0.002} 

我们发现，xLearn 训练过后在当前文件夹下产生了一个叫 ``model.out`` 的新文件。这个文件用来存储训练后的模型，我们可以用这个模型在未来进行预测: ::

    ffm_model.setTest("./small_test.txt")
    ffm_model.predict("./model.out", "./output.txt")      

运行上述命令之后，我们在当前文件夹下得到了一个新的文件 ``output.txt``，这是我们进行预测任务的输出。我们可以通过如下命令显示这个输出文件的前几行数据: ::

    head -n 5 ./output.txt

    -1.58631
    -0.393496
    -0.638334
    -0.38465
    -1.15343

这里每一行的分数都对应了测试数据中的一行预测样本。负数代表我们预测该样本为负样本，正数代表正样本 (在这个例子中没有)。在 xLearn 中，用户可以将分数通过 ``setSigmoid()`` API 转换到（0-1）之间: ::

   ffm_model.setSigmoid()
   ffm_model.setTest("./small_test.txt")  
   ffm_model.predict("./model.out", "./output.txt")      

结果如下: ::

   head -n 5 ./output.txt

  0.174698
  0.413642
  0.353551
  0.414588
  0.250373

用户还可以使用 ``setSign()`` API 将预测结果转换成 0 或 1: ::

   ffm_model.setSign()
   ffm_model.setTest("./small_test.txt")  
   ffm_model.predict("./model.out", "./output.txt")

结果如下: ::

   head -n 5 ./output.txt

   0
   0
   0
   0
   0

模型输出
----------------------------------------

用户还可以通过 ``setTXTModel()`` API 将模型输出成人类可读的 ``TXT`` 格式，例如: ::

    ffm_model.setSign()
    ffm_model.setTXTModel("./model.txt")
    ffm_model.setTest("./small_test.txt")  
    ffm_model.predict("./model.out", "./output.txt")

运行上述命令后，我们发现在当前文件夹下生成了一个新的文件 ``model.txt``，这个文件存储着 ``TXT`` 格式的输出模型: ::

  head -n 5 ./model.txt

  -1.041
  0.31609
  0
  0
  0

对于线性模型来说，TXT 格式的模型输出将每一个模型参数存储在一行。对于 FM 和 FFM，模型将每一个 latent vector 存储在一行。

选择机器学习算法
----------------------------------------

目前，xLearn 可以支持三种不同的机器学习算法，包括了线性模型 (LR)、factorization machine (FM)，以及 field-aware factorization machine (FFM): ::
   
    import xlearn as xl

    ffm_model = xl.create_ffm()
    fm_model = xl.create_fm()
    lr_model = xl.create_linear()

对于 LR 和 FM 算法而言，我们的输入数据格式必须是 ``CSV`` 或者 ``libsvm``. 对于 FFM 算法而言，我们的输入数据必须是 ``libffm`` 格式: ::

  libsvm format:

     y index_1:value_1 index_2:value_2 ... index_n:value_n

     0   0:0.1   1:0.5   3:0.2   ...
     0   0:0.2   2:0.3   5:0.1   ...
     1   0:0.2   2:0.3   5:0.1   ...

  CSV format:

     y value_1 value_2 .. value_n

     0      0.1     0.2     0.2   ...
     1      0.2     0.3     0.1   ...
     0      0.1     0.2     0.4   ...

  libffm format:

     y field_1:index_1:value_1 field_2:index_2:value_2   ...

     0   0:0:0.1   1:1:0.5   2:3:0.2   ...
     0   0:0:0.2   1:2:0.3   2:5:0.1   ...
     1   0:0:0.2   1:2:0.3   2:5:0.1   ...

注意，如果输入的 csv 文件里不含 ``y`` 值，用户必须手动向其每一行数据添加一个占位符 (同样针对测试数据)。否则，xLearn 会将第一个元素视为 ``y``.

LR 和 FM 算法的输入可以是 ``libffm`` 格式，xLearn 会忽略其中的 ``field`` 项并将其视为 ``libsvm`` 格式。

设置 Validation Dataset (验证集)
----------------------------------------

在机器学习中，我们可以通过 Validation Dataset (验证集) 来进行超参数调优。在 xLearn 中，用户可以使用 ``setValidate()`` API 来指定验证集文件，例如: ::

   import xlearn as xl

   ffm_model = xl.create_ffm()
   ffm_model.setTrain("./small_train.txt")
   ffm_model.setValidate("./small_test.txt")  
   param = {'task':'binary', 'lr':0.2, 'lambda':0.002} 
            
   ffm_model.fit(param, "./model.out") 

下面是程序的一部分输出: ::

  [ ACTION     ] Start to train ...
  [------------] Epoch      Train log_loss       Test log_loss     Time cost (sec)
  [   10%      ]     1            0.589475            0.535867                0.00
  [   20%      ]     2            0.540977            0.546504                0.00
  [   30%      ]     3            0.521881            0.531474                0.00
  [   40%      ]     4            0.507194            0.530958                0.00
  [   50%      ]     5            0.495460            0.530627                0.00
  [   60%      ]     6            0.483910            0.533307                0.00
  [   70%      ]     7            0.470661            0.527650                0.00
  [   80%      ]     8            0.465455            0.532556                0.00
  [   90%      ]     9            0.455787            0.538841                0.00
  [ ACTION     ] Early-stopping at epoch 7

我们可以看到，在这个任务中 ``Train log_loss`` 在不断的下降，而 ``Test log_loss`` (validation loss) 则是先下降，后上升。这代表当前我们训练的模型已经 overfit （过拟合）我们的训练数据。

在默认的情况下，xLearn 会在每一轮 epoch 结束后计算 validation loss 的数值，而用户可以使用 ``metric``  参数来制定不同的评价指标。对于分类任务而言，评价指标有：``acc`` (accuracy), ``prec`` (precision), ``f1``, 以及 ``auc``，例如: ::

   param = {'task':'binary', 'lr':0.2, 'lambda':0.002, 'metric': 'acc'}
   param = {'task':'binary', 'lr':0.2, 'lambda':0.002, 'metric': 'prec'}
   param = {'task':'binary', 'lr':0.2, 'lambda':0.002, 'metric': 'f1'}
   param = {'task':'binary', 'lr':0.2, 'lambda':0.002, 'metric': 'auc'}           

对于回归任务而言，评价指标包括：mae, mape, 以及 rmsd (或者叫作 rmse)，例如: ::

   param = {'task':'binary', 'lr':0.2, 'lambda':0.002, 'metric': 'rmse'}
   param = {'task':'binary', 'lr':0.2, 'lambda':0.002, 'metric': 'mae'}    
   param = {'task':'binary', 'lr':0.2, 'lambda':0.002, 'metric': 'mape'}  

Cross-Validation (交叉验证)
----------------------------------------

在机器学习中，Cross-Validation (交叉验证) 是一种被广泛使用的模型超参数调优技术。在 xLearn 中，用户可以使用 ``cv()`` API 来使用交叉验证功能，例如: ::

    import xlearn as xl

    ffm_model = xl.create_ffm()
    ffm_model.setTrain("./small_train.txt")  
    param = {'task':'binary', 'lr':0.2, 'lambda':0.002} 
            
    ffm_model.cv(param)

在默认的情况下，xLearn 使用 5-folds 交叉验证 (即将数据集平均分成 5 份)，用户也可以通过 ``fold`` 参数来指定数据划分的份数，例如: ::

    import xlearn as xl

    ffm_model = xl.create_ffm()
    ffm_model.setTrain("./small_train.txt")  
    param = {'task':'binary', 'lr':0.2, 'lambda':0.002, 'fold':3} 
            
    ffm_model.cv(param)     

上述命令将数据集划分成为 3 份，并且 xLearn 会在最后计算出平均的 validation loss: ::

  [------------] Average log_loss: 0.549758
  [ ACTION     ] Finish Cross-Validation
  [ ACTION     ] Clear the xLearn environment ...
  [------------] Total time cost: 0.05 (sec)

选择优化算法
----------------------------------------

在 xLearn 中，用户可以通过 ``opt`` 参数来选择使用不同的优化算法。目前，xLearn 支持 ``SGD``, ``AdaGrad``, 以及 ``FTRL`` 这三种优化算法。 在默认的情况下，xLearn 使用 ``AdaGrad`` 优化算法: ::

   param = {'task':'binary', 'lr':0.2, 'lambda':0.002, 'opt':'sgd'} 
   param = {'task':'binary', 'lr':0.2, 'lambda':0.002, 'opt':'adagrad'} 
   param = {'task':'binary', 'lr':0.2, 'lambda':0.002, 'opt':'ftrl'} 

相比于传统的 SGD (随机梯度下降) 算法，AdaGrad 可以自适应的调整学习速率 learning rate，对于不常用的参数进行较大的更新，对于常用的参数进行较小的更新。 正因如此，AdaGrad 算法常用于稀疏数据的优化问题上。除此之外，相比于 AdaGrad，SGD 对学习速率的大小更敏感，这增加了用户调参的难度。

FTRL (Follow-the-Regularized-Leader) 同样被广泛应用于大规模稀疏数据的优化问题上。相比于 SGD 和 AdaGrad, FTRL 需要用户调试更多的超参数，我们将在下一节详细介绍 xLearn 的超参数调优。

超参数调优
----------------------------------------

在机器学习中，hyper-parameter (超参数) 是指在训练之前设置的参数，而模型参数是指在训练过程中更新的参数。超参数调优通常是机器学习训练过程中不可避免的一个环节。

首先，``learning rate`` (学习速率) 是机器学习中的一个非常重要的超参数，用来控制每次模型迭代时更新的步长。在默认的情况下，这个值在 xLearn 中被设置为 0.2，用户可以通过 ``lr`` 参数来改变这个值: ::

    param = {'task':'binary', 'lr':0.2} 
    param = {'task':'binary', 'lr':0.5}
    param = {'task':'binary', 'lr':0.01}

用户还可以通过 ``-b`` 选项来控制 regularization (正则项)。xLearn 使用 ``L2`` 正则项，这个值被默认设置为 ``0.00002``: ::

    param = {'task':'binary', 'lr':0.2, 'lambda':0.01}
    param = {'task':'binary', 'lr':0.2, 'lambda':0.02} 
    param = {'task':'binary', 'lr':0.2, 'lambda':0.002} 

对于 FTRL 算法来说，除了学习速率和正则项，我们还需要调节其他的超参数，包括：``-alpha``, ``-beta``, ``-lambda_1`` 和 ``-lambda_2``，例如: ::

    param = {'alpha':0.002, 'beta':0.8, 'lambda_1':0.001, 'lambda_2': 1.0}

对于 FM 和 FFM 模型，用户需要通过 ``-k`` 选项来设置 latent vector (隐向量) 的长度。在默认的情况下，xLearn 将其设置为 ``4``: ::

    param = {'task':'binary', 'lr':0.2, 'lambda':0.01, 'k':2}
    param = {'task':'binary', 'lr':0.2, 'lambda':0.01, 'k':4}
    param = {'task':'binary', 'lr':0.2, 'lambda':0.01, 'k':5}
    param = {'task':'binary', 'lr':0.2, 'lambda':0.01, 'k':8}

注意，xLearn 使用了 *SSE* 硬件指令来加速向量运算，该指令会同时进行向量长度为 4 的运算，因此将 k=2 和 k=4 所需的运算时间是相同的。

除此之外，对于 FM 和 FFM，用户可以通过设置超参数 ``-u`` 来调节模型的初始化参数。在默认的情况下，这个值被设置为 ``0.66``: ::

    param = {'task':'binary', 'lr':0.2, 'lambda':0.01, 'init':0.80}
    param = {'task':'binary', 'lr':0.2, 'lambda':0.01, 'init':0.40}
    param = {'task':'binary', 'lr':0.2, 'lambda':0.01, 'init':0.10}
  
迭代次数 & Early-Stop (提前终止)
----------------------------------------

在模型的训练过程中，每一个 epoch 都会遍历整个训练数据。在 xLearn 中，用户可以通过 ``epoch`` 选项来设置需要的 epoch 数量: ::

    param = {'task':'binary', 'lr':0.2, 'lambda':0.01, 'epoch':3}
    param = {'task':'binary', 'lr':0.2, 'lambda':0.01, 'epoch':5}
    param = {'task':'binary', 'lr':0.2, 'lambda':0.01, 'epoch':10}

如果用户设置了 validation dataset (验证集)，xLearn 在默认情况下会在得到最好的 validation 结果时进行 early-stopping (提前终止训练)，例如: ::

   import xlearn as xl

   ffm_model = xl.create_ffm()
   ffm_model.setTrain("./small_train.txt")
   ffm_model.setValidate("./small_test.txt")
   param = {'task':'binary', 'lr':0.2, 'lambda':0.002, 'epoch':10} 
            
   ffm_model.fit(param, "./model.out") 

在上述命令中，我们设置 epoch 的大小为 10，但是 xLearn 会在第 7 轮提前停止训练 (你可能在你的本地计算机上会得到不同的轮次): ::

    Early-stopping at epoch 7
    Start to save model ...

用户可以通过 ``stop_window`` 参数来设置提前停止机制的窗口大小。即，``stop_window=2`` 意味着如果在后两轮的时间窗口之内都没有比当前更好的验证结果，则停止训练，并保存之前最好的模型: ::

    param = {'task':'binary',  'lr':0.2, 
             'lambda':0.002, 'epoch':10,
             'stop_window':3} 
            
    ffm_model.fit(param, "./model.out") 

用户可以通过 ``disableEarlyStop()`` 选项来禁止 early-stop: ::

   import xlearn as xl

   ffm_model = xl.create_ffm()
   ffm_model.setTrain("./small_train.txt")
   ffm_model.setValidate("./small_test.txt")
   ffm_model.disableEarlyStop();
   param = {'task':'binary', 'lr':0.2, 'lambda':0.002, 'epoch':10} 
            
   ffm_model.fit(param, "./model.out") 

在上述命令中，xLearn 将进行完整的 10 轮 epoch 训练。

无锁（Lock-free）学习
----------------------------------------

在默认情况下，xLearn 会进行 *Hogwild!* 无锁学习，该方法通过 CPU 多核进行并行训练，提高 CPU 利用率，加快算法收敛速度。但是，该无锁算法是非确定性的算法 (non-deterministic). 即，如果我们多次运行如下的命令，我们会在每一次运行得到略微不同的 loss 结果: ::

   import xlearn as xl

   ffm_model = xl.create_ffm()
   ffm_model.setTrain("./small_train.txt")  
   param = {'task':'binary', 'lr':0.2, 'lambda':0.002} 
            
   ffm_model.fit(param, "./model.out") 

   The 1st time: 0.449056
   The 2nd time: 0.449302
   The 3nd time: 0.449185

用户可以通过 ``nthread`` 参数来设置使用 CPU 核心的数量，例如: ::

   import xlearn as xl

   ffm_model = xl.create_ffm()
   ffm_model.setTrain("./small_train.txt")  
   param = {'task':'binary', 'lr':0.2, 'lambda':0.002, 'nthread':4} 
            
   ffm_model.fit(param, "./model.out") 

上述代码指定使用 4 个 CPU Core 来进行模型训练。如果用户不设置该选项，xLearn 在默认情况下会使用全部的 CPU 核心进行计算。

用户可以通过设置 ``disableLockFree()`` API 禁止多核无锁学习: ::

   import xlearn as xl

   ffm_model = xl.create_ffm()
   ffm_model.setTrain("./small_train.txt")  
   ffm_model.disableLockFree()
   param = {'task':'binary', 'lr':0.2, 'lambda':0.002} 
            
   ffm_model.fit(param, "./model.out") 

这时，xLearn 计算的结果是确定性的 (determinnistic): ::

   The 1st time: 0.449172
   The 2nd time: 0.449172
   The 3nd time: 0.449172

使用 ``disableLockFree()`` 的缺点是训练速度会比无锁训练慢很多，我们的建议是在大规模数据训练下开启此功能。

Instance-Wise 归一化
----------------------------------------

对于 FM 和 FFM 来说，xLearn 会默认对特征进行 Instance-Wise Normalizarion (归一化). 在一些大规模稀疏数据的场景 (例如 CTR 预估), 这一技术非常的有效，但是有些时候它也会影响模型的准确率。用户可以通过设置 ``disableNorm()`` API 来关掉该功能: ::

   import xlearn as xl

   ffm_model = xl.create_ffm()
   ffm_model.setTrain("./small_train.txt")  
   ffm_model.disableNorm()
   param = {'task':'binary', 'lr':0.2, 'lambda':0.002} 
            
   ffm_model.fit(param, "./model.out") 

Quiet Model 安静模式
----------------------------------------

xLearn 的训练支持安静模式，在安静模式下，用户通过调用 ``setQuiet()`` API 来使得 xLearn 的训练过程不会计算任何评价指标，这样可以很大程度上提高训练速度: ::

   import xlearn as xl

   ffm_model = xl.create_ffm()
   ffm_model.setTrain("./small_train.txt")  
   ffm_model.setQuiet()
   param = {'task':'binary', 'lr':0.2, 'lambda':0.002} 
            
   ffm_model.fit(param, "./model.out") 

Scikit-learn API
----------------------------------------

xLearn 还可以支持 Scikit-learn API: ::

  import numpy as np
  import xlearn as xl
  from sklearn.datasets import load_iris
  from sklearn.model_selection import train_test_split

  # Load dataset
  iris_data = load_iris()
  X = iris_data['data']
  y = (iris_data['target'] == 2)

  X_train,   \
  X_val,     \
  y_train,   \
  y_val = train_test_split(X, y, test_size=0.3, random_state=0)

  # param:
  #  0. binary classification
  #  1. model scale: 0.1
  #  2. epoch number: 10 (auto early-stop)
  #  3. learning rate: 0.1
  #  4. regular lambda: 1.0
  #  5. use sgd optimization method
  linear_model = xl.LRModel(task='binary', init=0.1, 
                            epoch=10, lr=0.1, 
                            reg_lambda=1.0, opt='sgd')

  # Start to train
  linear_model.fit(X_train, y_train, 
                   eval_set=[X_val, y_val], 
                   is_lock_free=False)

  # Generate predictions
  y_pred = linear_model.predict(X_val)

.. __: https://github.com/aksnzhy/xlearn/tree/master/demo/classification/scikit_learn_demo
