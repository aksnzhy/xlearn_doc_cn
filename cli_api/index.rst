xLearn 命令行接口使用指南
===============================

如果你已经编译并安装好 xLearn，你会在当前的 ``build`` 文件夹下看见 ``xlearn_train`` 和 ``xlearn_predict`` 两个可执行文件，它们可以
被用来进行模型训练和模型预测。

快速开始
----------------------------------------

确保你现在正在 xLearn 的 ``build`` 文件夹下，你可以在该文件夹下看见 ``small_test.txt`` 和 ``small_train.txt`` 这两个数据集。我们使用以下命令
进行模型训练：::

    ./xlearn_train ./small_train.txt

下面是一部分程序的输出。注意，这里的 loss 值可能和你本地计算出的 loss 值不完全一样。 ::

    Epoch      Train log_loss     Time cost (sec)
        1            0.567514                0.00
        2            0.516861                0.00
        3            0.489884                0.00
        4            0.469971                0.00
        5            0.452699                0.00
        6            0.437590                0.00
        7            0.425759                0.00
        8            0.415190                0.00
        9            0.405954                0.00
       10            0.396313                0.00

在默认的情况下，xLearn 会使用 logistic regression (LR) 来训练我们的模型（in 10 epoch）。

我们可以看见，训练过后在当前文件夹下产生了一个叫 ``small_train.txt.model`` 的新文件。这个文件用来存储训练后的模型，我们可以用这个模型在未来进行预测。::

    ./xlearn_predict ./small_test.txt ./small_train.txt.model

运行上述命令之后，我们在当前文件夹下得到了一个新的文件 ``small_test.txt.out``。这是我们进行预测任务的输出。我们可以通过如下命令显示这个文件的前几行数据: ::
    
    head -n 5 ./small_test.txt.out

    -1.9872
    -0.0707959
    -0.456214
    -0.170811
    -1.28986

这里每一行的分数都对应了测试数据中的一行样本。负数代表负样本，正数代表正样本（在这个例子中没有）。在 xLearn 中，用户可以将分数通过 ``--sigmoid`` 选项转换到（0-1）之间，还可以使用 ``--sign`` 选项将其转换成 0 和 1: ::

    ./xlearn_predict ./small_test.txt ./small_train.txt.model --sigmoid
    head -n 5 ./small_test.txt.out

    0.120553
    0.482308
    0.387884
    0.457401
    0.215877

    ./xlearn_predict ./small_test.txt ./small_train.txt.model --sign
    head -n 5 ./small_test.txt.out

    0
    0
    0
    0
    0

用户可以通过设置不同的超参数来生成不同的模型，通过 ``-m`` 选项来制定这些输出模型的名字。在默认的情况下，模型文件的名字是 ``training_data_name`` + ``.model`` ::

  ./xlearn_train ./small_train.txt -m new_model

用户还可以通过 ``-t`` 选项将模型输出成可读的 ``TXT`` 格式，例如：::

  ./xlearn_train ./small_train.txt -t model.txt

运行上述命令我们可以看到在当前文件夹下生成了一个新的文件 ``model.txt``，这个文件存储着 ``TXT`` 格式的模型: ::

  head -n 5 ./model.txt

  -0.688182
  0.458082
  0
  0
  0

对于线性模型来说，``TXT`` 格式的模型将每一个模型参数存储在一行。对于 FM 和 FFM，模型将每一个 latent vector 存储在一行。

用户可以通过 ``-o`` 选项来指定预测输出文件的路径和名字。例如 ::

  ./xlearn_predict ./small_test.txt ./small_train.txt.model -o output.txt  
  head -n 5 ./output.txt

  -2.01192
  -0.0657416
  -0.456185
  -0.170979
  -1.28849

在默认的情况下，输出文件的路径格式是 ``test_data_name`` + ``.out`` .


选择机器学习算法
----------------------------------------

目前，xLearn 可以支持三种不同的机器学习算法，包括了线性模型（LR）、factorization machine (FM)，以及 field-aware factorization machine (FFM)。

用户可以通过 ``-s`` 选项来选择不同的算法: ::

  -s <type> : Type of machine learning model (default 0)
     for classification task:
         0 -- linear model (GLM)
         1 -- factorization machines (FM)
         2 -- field-aware factorization machines (FFM)
     for regression task:
         3 -- linear model (GLM)
         4 -- factorization machines (FM)
         5 -- field-aware factorization machines (FFM)

对于 LR 和 FM 算法，我们的输入数据格式必须是 ``CSV`` 或者 ``libsvm``. 对于 FFM 算法，我们的输入数据必须是 ``libffm`` 格式. ::

  libsvm format:

     label index_1:value_1 index_2:value_2 ... index_n:value_n

  CSV format:

     label value_1 value_2 .. value_n

  libffm format:

     label field_1:index_1:value_1 field_2:index_2:value_2 ...

注意，如果输入的 csv 文件里不含 ``y`` 值，用户必须手动向其添加一个占位符（同样针对测试数据）。否则，xLearn 会将第一个元素视为 ``y``.

Users can also give a ``libffm`` file to LR and FM task. At that time, xLearn will 
treat this data as ``libsvm`` format. The following command shows how to use different
machine learning algorithms to solve the binary classification problem:  ::

./xlearn_train ./small_train.txt -s 0  # Linear model (GLM)
./xlearn_train ./small_train.txt -s 1  # Factorization machine (FM)
./xlearn_train ./small_train.txt -s 2  # Field-awre factorization machine (FFM)

设置 Validation Dataset（验证集）
----------------------------------------

A validation dataset is used to tune the hyper-parameters of a machine learning model. 
In xLearn, users can use ``-v`` option to set the validation dataset. For example: ::

    ./xlearn_train ./small_train.txt -v ./small_test.txt    

A portion of xLearn's output: ::

    Epoch      Train log_loss       Test log_loss     Time cost (sec)
        1            0.575049            0.530560                0.00
        2            0.517496            0.537741                0.00
        3            0.488428            0.527205                0.00
        4            0.469010            0.538175                0.00
        5            0.452817            0.537245                0.00
        6            0.438929            0.536588                0.00
        7            0.423491            0.532349                0.00
        8            0.416492            0.541107                0.00
        9            0.404554            0.546218                0.00

Here we can see that the training loss continuously goes down. But the validation loss (test loss) goes down 
first, and then goes up. This is because the model has already overfitted current training dataset. By default, 
xLearn will calculate the validation loss in each epoch, while users can also set different evaluation metrics by 
using ``-x`` option. For classification problems, the metric can be : ``acc`` (accuracy), ``prec`` (precision), 
``f1`` (f1 score), ``auc`` (AUC score). For example: ::

    ./xlearn_train ./small_train.txt -v ./small_test.txt -x acc
    ./xlearn_train ./small_train.txt -v ./small_test.txt -x prec
    ./xlearn_train ./small_train.txt -v ./small_test.txt -x f1
    ./xlearn_train ./small_train.txt -v ./small_test.txt -x auc

For regression problems, the metric can be ``mae``, ``mape``, and ``rmsd`` (rmse). For example: ::

    cd demo/house_price/
    ../../xlearn_train ./house_price_train.txt -s 3 -x rmse --cv
    ../../xlearn_train ./house_price_train.txt -s 3 -x rmsd --cv

Note that, in the above example we use cross-validation by using ``--cv`` option, which will be 
introduced in the next section.

Cross-Validation
----------------------------------------

Cross-validation, sometimes called rotation estimation, is a model validation technique for assessing 
how the results of a statistical analysis will generalize to an independent dataset. In xLearn, users 
can use the ``--cv`` option to use this technique. For example: ::

    ./xlearn_train ./small_train.txt --cv

On default, xLearn uses 5-folds cross validation, and users can set the number of fold by using 
``-f`` option: ::
    
    ./xlearn_train ./small_train.txt -f 3 --cv

Here we set the number of folds to ``3``. The xLearn will calculate the average validation loss at 
the end of its output message. ::

     ...
    [------------] Average log_loss: 0.549417
    [ ACTION     ] Finish Cross-Validation
    [ ACTION     ] Clear the xLearn environment ...
    [------------] Total time cost: 0.03 (sec)

Choose Optimization Method
----------------------------------------
 
In xLearn, users can choose different optimization methods by using ``-p`` option. For now, xLearn 
can support ``sgd``, ``adagrad``, and ``ftrl`` method. By default, xLearn uses the ``adagrad`` method. 
For example: ::

    ./xlearn_train ./small_train.txt -p sgd
    ./xlearn_train ./small_train.txt -p adagrad
    ./xlearn_train ./small_train.txt -p ftrl

Compared to traditional ``sgd`` method, ``adagrad`` adapts the learning rate to the parameters, performing 
larger updates for infrequent and smaller updates for frequent parameters. For this reason, it is well-suited for 
dealing with sparse data. In addition, ``sgd`` is more sensitive to the learning rate compared with ``adagrad``.

``FTRL`` (Follow-the-Regularized-Leader) is also a famous method that has been widely used in the large-scale 
sparse problem. To use FTRL, users need to tune more hyper-parameters compared with ``sgd`` and ``adagrad``. 

Hyper-parameter Tuning
----------------------------------------

In machine learning, a *hyper-parameter* is a parameter whose value is set before the learning process begins. 
By contrast, the value of other parameters is derived via training. Hyper-parameter tuning is the problem of 
choosing a set of optimal hyper-parameters for a learning algorithm.

First, the ``learning rate`` is one of the most important hyper-parameters used in machine learning. 
By default, this value is set to ``0.2`` in xLearn, and we can tune this value by using ``-r`` option: ::

    ./xlearn_train ./small_train.txt -v ./small_test.txt -r 0.1
    ./xlearn_train ./small_train.txt -v ./small_test.txt -r 0.5
    ./xlearn_train ./small_train.txt -v ./small_test.txt -r 0.01

We can also use the ``-b`` option to perform regularization. By default, xLearn uses ``L2`` regularization, and 
the *regular_lambda* has been set to ``0.00002``. ::

    ./xlearn_train ./small_train.txt -v ./small_test.txt -r 0.1 -b 0.001
    ./xlearn_train ./small_train.txt -v ./small_test.txt -r 0.1 -b 0.002
    ./xlearn_train ./small_train.txt -v ./small_test.txt -r 0.1 -b 0.01


For the ``FTRL`` method, we also need to tune another four hyper-parameters, including ``-alpha``, ``-beta``, 
``-lambda_1``, and ``-lambda_2``. For example: ::

    ./xlearn_train ./small_train.txt -p ftrl -alpha 0.002 -beta 0.8 -lambda_1 0.001 -lambda_2 1.0

For FM and FFM, users also need to set the size of *latent factor* by using ``-k`` option. By default, xLearn 
uses ``4`` for this value. ::

    ./xlearn_train ./small_train.txt -s 1 -v ./small_test.txt -k 2
    ./xlearn_train ./small_train.txt -s 1 -v ./small_test.txt -k 4
    ./xlearn_train ./small_train.txt -s 1 -v ./small_test.txt -k 5
    ./xlearn_train ./small_train.txt -s 1 -v ./small_test.txt -k 8

xLearn uses *SSE* instruction to accelerate vector operation, and hence the time cost for ``k=2`` and ``k=4`` are the same.

For FM and FFM, users can also set the hyper-parameter ``-u`` for model initialization. By default, this value is set to 0.66. ::

    ./xlearn_train ./small_train.txt -s 1 -v ./small_test.txt -u 0.80
    ./xlearn_train ./small_train.txt -s 1 -v ./small_test.txt -u 0.40
    ./xlearn_train ./small_train.txt -s 1 -v ./small_test.txt -u 0.10

Set Epoch Number and Early-Stopping
----------------------------------------

For machine learning tasks, one epoch consists of one full training cycle on the training set. 
In xLearn, users can set the number of epoch for training by using ``-e`` option. ::

    ./xlearn_train ./small_train.txt -e 3
    ./xlearn_train ./small_train.txt -e 5
    ./xlearn_train ./small_train.txt -e 10   

If you set the validation data, xLearn will perform early-stopping by default. For example: ::
  
    ./xlearn_train ./small_train.txt -s 2 -v ./small_test.txt -e 10

Here, we set epoch number to ``10``, but xLearn stopped at epoch ``7`` because we get the best model 
at that epoch (you may get different a stopping number on your local machine) ::

   ...
  [ ACTION     ] Early-stopping at epoch 7
  [ ACTION     ] Start to save model ...

Users can set the ``window size`` for early stopping by using ``-sw`` option. ::

    ./xlearn_train ./small_train.txt -e 10 -v ./small_test.txt -sw 3

Users can disable early-stopping by using ``--dis-es`` option ::

    ./xlearn_train ./small_train.txt -s 2 -v ./small_test.txt -e 10 --dis-es

At this time, xLearn performed completed 10 epoch for training.

Lock-Free Learning
----------------------------------------

By default, xLearn performs *Hogwild! lock-free* learning, which takes advantages of multiple cores of modern CPU to 
accelerate training task. But lock-free training is *non-deterministic*. For example, if we run the following command 
multiple times, we may get different loss value at each epoch. ::

   ./xlearn_train ./small_train.txt 

   The 1st time: 0.396352
   The 2nd time: 0.396119
   The 3nd time: 0.396187
   ...

Users can set the number of thread for xLearn by using ``-nthread`` option: ::

   ./xlearn_train ./small_train.txt -nthread 2

If you don't set this option, xLearn uses all of the CPU cores by default.

Users can disable lock-free training by using ``--dis-lock-free`` ::

  ./xlearn_train ./small_train.txt --dis-lock-free

In thie time, our result are *determinnistic*. ::

   The 1st time: 0.396372
   The 2nd time: 0.396372
   The 3nd time: 0.396372

The disadvantage of ``--dis-lock-free`` is that it is *much slower* than lock-free training. 

Instance-wise Normalization
----------------------------------------

For FM and FFM, xLearn uses *instance-wise normalizarion* by default. In some scenes like CTR prediction, this technique is very
useful. But sometimes it hurts model performance. Users can disable instance-wise normalization by using ``--no-norm`` option ::

  ./xlearn_train ./small_train.txt -s 1 -v ./small_test.txt --no-norm

Note that we usually use ``--no-norm`` in regression tasks.

Quiet Training
----------------------------------------

When using ``--quiet`` option, xLearn will not calculate any evaluation information during the training, and 
it will just train the model quietly ::

  ./xlearn_train ./small_train.txt --quiet

In this way, xLearn can accelerate its training speed significantly.

xLearn can also support Python API, and we will introduce it in the next section.
