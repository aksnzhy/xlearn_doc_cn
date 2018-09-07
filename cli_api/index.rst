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

模型输出
----------------------------------------

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

LR 和 FM 算法的输入可以是 ``libffm`` 格式，xLearn 会忽略其中的 ``libsvm`` 项并将其视为 ``libsvm`` 格式。如下命令展示了如何选择不同的机器学习算法: ::

  ./xlearn_train ./small_train.txt -s 0  # Linear model (GLM)
  ./xlearn_train ./small_train.txt -s 1  # Factorization machine (FM)
  ./xlearn_train ./small_train.txt -s 2  # Field-awre factorization machine (FFM)

设置 Validation Dataset（验证集）
----------------------------------------

在机器学习中，我们可以通过 Validation Dataset （验证集）来进行超参数调优。在 xLearn 中，用户可以使用 ``-v`` 来指定验证数据集，例如: ::

    ./xlearn_train ./small_train.txt -v ./small_test.txt    

下面是程序的一部分输出: ::

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

我们可以看到，在这个任务中 training loss 在不断的下降，而 validation loss （test loss）则是先下降，后上升。这代表当前我们训练的模型已经 overfit （过拟合）我们的训练数据。在默认的情况下，xLearn 会在每一轮 epoch 结束后计算 validation loss 的数值，用户可以选择使用不同的评价指标。对于分类任务而言，评价指标有： ``acc`` (accuracy), ``prec`` (precision), 
``f1`` (f1 score), ``auc`` (AUC score)，例如: ::

    ./xlearn_train ./small_train.txt -v ./small_test.txt -x acc
    ./xlearn_train ./small_train.txt -v ./small_test.txt -x prec
    ./xlearn_train ./small_train.txt -v ./small_test.txt -x f1
    ./xlearn_train ./small_train.txt -v ./small_test.txt -x auc

对于回归任务而言，评价指标包括：``mae``, ``mape``, and ``rmsd`` (或者 ``rmse`` ). 例如: ::

    cd demo/house_price/
    ../../xlearn_train ./house_price_train.txt -s 3 -x rmse --cv
    ../../xlearn_train ./house_price_train.txt -s 3 -x rmsd --cv

注意，这里我们通过设置 ``--cv`` 选项使用了 cross-validation （交叉验证），我们将在下一节详细介绍该功能。

交叉验证
----------------------------------------

在机器学习中，cross-validation （交叉验证）是一种被广泛使用的模型选择于调优技术。在 xLearn 中，用户可以使用 ``--cv`` 
选项来使用交叉验证功能，例如: ::

    ./xlearn_train ./small_train.txt --cv

在默认的情况下，xLearn 使用 5-folds 交叉验证（即将数据集平均分成 5 份），用户也可以通过 ``-f`` 选项来指定数据划分的份数，例如: ::
    
    ./xlearn_train ./small_train.txt -f 3 --cv

上述命令将数据集划分成为 3 份。xLearn 会在最后计算平均的 validation loss: ::

     ...
    [------------] Average log_loss: 0.549417
    [ ACTION     ] Finish Cross-Validation
    [ ACTION     ] Clear the xLearn environment ...
    [------------] Total time cost: 0.03 (sec)

选择优化算法
----------------------------------------
 
在 xLearn 中，用户可以通过 ``-p`` 选项来选择使用不同的优化算法。目前，xLearn 支持 ``sgd``, ``adagrad``, 以及 ``ftrl`` 这三种优化算法。
在默认的情况下，xLearn 使用 ``adagrad`` 优化算法: ::

    ./xlearn_train ./small_train.txt -p sgd
    ./xlearn_train ./small_train.txt -p adagrad
    ./xlearn_train ./small_train.txt -p ftrl

相比于传统的 ``sgd`` （随机梯度下降）算法，``adagrad`` 可以自适应的调整学习速率 learning rate，对于不常用的参数进行大的更新，对于常用的参数进行小的更新。
正因如此，``adagrad`` 常用语稀疏数据的优化问题上。除此之外，相比于 ``adagrad``，``sgd`` 对学习速率更敏感，这增加了用户调参的难度。

``FTRL`` (Follow-the-Regularized-Leader) 同样被广泛应用于大规模稀疏数据的优化问题上。相比于 SGD 和 Adagrad，使用 FTRL 用户需要调试更多的超参数。

超参数调优
----------------------------------------

在机器学习中，*hyper-parameter* （超参数）是指在训练之前设置的参数，而模型参数是指在训练过程中更新的参数。超参数调优通常是机器学习训练不可避免的一个环节。

首先，``learning rate`` （学习速率）是机器学习中的一个非常重要的超参数，用来控制每次模型更新的步长。在默认的情况下，这个值在 xLearn 中被设置为 ``0.2``，用户可以通过 ``-r`` 选项来改变这个值: ::

    ./xlearn_train ./small_train.txt -v ./small_test.txt -r 0.1
    ./xlearn_train ./small_train.txt -v ./small_test.txt -r 0.5
    ./xlearn_train ./small_train.txt -v ./small_test.txt -r 0.01

用户还可以通过 ``-b`` 来控制 regularization （正则项）。xLearn 使用 ``L2`` 正则项，这个值 *regular_lambda* 被默认设置为 ``0.00002``: ::

    ./xlearn_train ./small_train.txt -v ./small_test.txt -r 0.1 -b 0.001
    ./xlearn_train ./small_train.txt -v ./small_test.txt -r 0.1 -b 0.002
    ./xlearn_train ./small_train.txt -v ./small_test.txt -r 0.1 -b 0.01


对于 ``FTRL`` 算法来说，除了学习速率和正则项，我们还需要调节其他的超参数，包括：``-alpha``, ``-beta``, 
``-lambda_1``, and ``-lambda_2``. 例如: ::

    ./xlearn_train ./small_train.txt -p ftrl -alpha 0.002 -beta 0.8 -lambda_1 0.001 -lambda_2 1.0

对于 FM 和 FFM 模型，用户需要通过 ``-k`` 选项来设置 *latent factor* （隐向量）的大小。在默认的情况下，xLearn 将其设置为 ``4``: ::

    ./xlearn_train ./small_train.txt -s 1 -v ./small_test.txt -k 2
    ./xlearn_train ./small_train.txt -s 1 -v ./small_test.txt -k 4
    ./xlearn_train ./small_train.txt -s 1 -v ./small_test.txt -k 5
    ./xlearn_train ./small_train.txt -s 1 -v ./small_test.txt -k 8

xLearn 使用了 *SSE* 指令来加速向量运算，该指令会同时进行向量长度为 4 的运算，因此将 ``k=2`` 和 ``k=4`` 所需的运算时间是相同的。

除此之外，对于 FM 和 FFM，用户可以通过设置超参数 ``-u`` 来调节模型的初始化。在默认的情况下，这个值被设置为 ``0.66``: ::

    ./xlearn_train ./small_train.txt -s 1 -v ./small_test.txt -u 0.80
    ./xlearn_train ./small_train.txt -s 1 -v ./small_test.txt -u 0.40
    ./xlearn_train ./small_train.txt -s 1 -v ./small_test.txt -u 0.10

迭代次数 & 提前结束
----------------------------------------

在模型的训练过程中，每一个 epoch 会遍历整个训练数据。在 xLearn 中，用户可以通过 ``-e`` 选项来设置 epoch 的数量: ::

    ./xlearn_train ./small_train.txt -e 3
    ./xlearn_train ./small_train.txt -e 5
    ./xlearn_train ./small_train.txt -e 10   

如果用户设置了 validation dataset（验证集），xLearn 在默认情况下会在得到最好的 validation 结果时进行 early-stopping （提前停止），例如: ::
  
    ./xlearn_train ./small_train.txt -s 2 -v ./small_test.txt -e 10

在上述命令中，我们设置 epoch 的大小为 ``10``，但是 xLearn 会在第 7 轮提前停止训练（你可能在你的本地计算机上会得到不同的轮次）: ::

   ...
  [ ACTION     ] Early-stopping at epoch 7
  [ ACTION     ] Start to save model ...

用户可以通过 ``-sw`` 来设置提前停止机制的窗口大小。即，``-sw=2`` 意味着如果在后两轮之内都没有比当前更好的验证结果，则在当前轮提前停止: ::

    ./xlearn_train ./small_train.txt -e 10 -v ./small_test.txt -sw 3

用户还可以通过 ``--dis-es`` 选项来禁止 early-stopping: ::

    ./xlearn_train ./small_train.txt -s 2 -v ./small_test.txt -e 10 --dis-es

在上述命令中，xLearn 将进行完整的 10 轮 epoch 训练。

无锁 (Lock-free) 学习
----------------------------------------

在默认情况下，xLearn 会进行 *Hogwild!* 无锁学习，该方法通过 CPU 多核进行并行计算，提高 CPU 利用率，加快算法收敛速度。但是，该无锁算法是非确定性的算法（ *non-deterministic*）。例如，如果我们多次运行如下的命令，我们会在每一次运行得到不同的 loss 结果: ::

   ./xlearn_train ./small_train.txt 

   The 1st time: 0.396352
   The 2nd time: 0.396119
   The 3nd time: 0.396187
   ...

用户可以通过 ``-nthread`` 选项来设置使用 CPU 核心的数量，例如: ::

   ./xlearn_train ./small_train.txt -nthread 2

如果你不设置该选项，xLearn 在默认情况下会使用全部的 CPU 核心进行计算。

用户可以通过设置 ``--dis-lock-free`` 选项禁止多核无锁训练: ::

  ./xlearn_train ./small_train.txt --dis-lock-free

这时，xLearn 计算的结果是确定性的（*determinnistic*）: ::

   The 1st time: 0.396372
   The 2nd time: 0.396372
   The 3nd time: 0.396372

使用 ``--dis-lock-free`` 的缺点是这样训练速度会比无锁训练慢很多。

Instance-wise 归一化
----------------------------------------

对于 FM 和 FFM 来说，xLearn 会默认使用 *instance-wise normalizarion*. 在一些大规模稀疏数据的场景（例如 CTR 预估），这一技术非常的有效。但是有些时候它也会影响模型的准确率。用户可以通过设置 ``--no-norm`` 来关掉 *instance-wise normalizarion*: ::

  ./xlearn_train ./small_train.txt -s 1 -v ./small_test.txt --no-norm

安静模式
----------------------------------------

xLearn 的训练支持 *安静模式，在安静模式下，xLearn 的训练过程不会计算任何评价指标，这样可以极大的提高训练速度: ::

  ./xlearn_train ./small_train.txt --quiet

xLearn 还可以支持 Python API，我们将在下一节详细介绍。
