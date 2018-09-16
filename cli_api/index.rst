xLearn 命令行接口使用指南
===============================

如果你已经编译并安装好 xLearn，你会在当前的 ``build`` 文件夹下看见 ``xlearn_train`` 和 ``xlearn_predict`` 这两个可执行文件，它们可以被用来进行模型的训练和预测。

快速开始
----------------------------------------

确保你现在正在 xLearn 的 ``build`` 文件夹下，在该文件夹下用户可以看见 ``small_test.txt`` 和 ``small_train.txt`` 这两个样例数据集。我们使用以下命令进行模型训练: ::

    ./xlearn_train ./small_train.txt

下面是一部分程序的输出。注意，这里显示的 ``log_loss`` 值可能和你本地计算出的 ``log_loss`` 值不完全一样: ::

  [ ACTION     ] Start to train ...
  [------------] Epoch      Train log_loss     Time cost (sec)
  [   10%      ]     1            0.569292                0.00
  [   20%      ]     2            0.517142                0.00
  [   30%      ]     3            0.490124                0.00
  [   40%      ]     4            0.470445                0.00
  [   50%      ]     5            0.451919                0.00
  [   60%      ]     6            0.437888                0.00
  [   70%      ]     7            0.425603                0.00
  [   80%      ]     8            0.415573                0.00
  [   90%      ]     9            0.405933                0.00
  [  100%      ]    10            0.396388                0.00
  [ ACTION     ] Start to save model ...
  [------------] Model file: ./small_train.txt.model

在默认的情况下，xLearn 会使用 *logistic regression (LR)* 来训练我们的模型 (10 epoch).

我们发现，xLearn 训练过后在当前文件夹下产生了一个叫 ``small_train.txt.model`` 的新文件。这个文件用来存储训练后的模型，我们可以用这个模型在未来进行预测: ::

    ./xlearn_predict ./small_test.txt ./small_train.txt.model

运行上述命令之后，我们在当前文件夹下得到了一个新的文件 ``small_test.txt.out``，这是我们进行预测任务的输出。我们可以通过如下命令显示这个输出文件的前几行数据: ::
    
    head -n 5 ./small_test.txt.out

    -1.9872
    -0.0707959
    -0.456214
    -0.170811
    -1.28986

这里每一行的分数都对应了测试数据中的一行预测样本。负数代表我们预测该样本为负样本，正数代表正样本 (在这个例子中没有)。在 xLearn 中，用户可以将分数通过 ``--sigmoid`` 选项转换到 (0-1) 之间，还可以使用 ``--sign`` 选项将其转换成 0 或 1: ::

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

模型的输出
----------------------------------------

用户可以通过设置不同的超参数来生成不同的模型，xLearn 通过 ``-m`` 选项来指定这些输出模型文件的路径。在默认的情况下，模型文件的路径是当前运行文件夹下的 ``training_data_name`` + ``.model`` 文件: ::

  ./xlearn_train ./small_train.txt -m new_model

用户还可以通过 ``-t`` 选项将模型输出成人类可读的 ``TXT`` 格式，例如: ::

  ./xlearn_train ./small_train.txt -t model.txt

运行上述命令后，我们发现在当前文件夹下生成了一个新的文件 ``model.txt``，这个文件存储着 ``TXT`` 格式的输出模型: ::

  head -n 5 ./model.txt

  -0.688182
  0.458082
  0
  0
  0

对于线性模型来说，``TXT`` 格式的模型将每一个模型参数存储在一行。对于 FM 和 FFM，模型将每一个 latent vector 存储在一行。

Linear: ::

  bias: 0
  i_0: 0
  i_1: 0
  i_2: 0
  i_3: 0

FM: ::

  bias: 0
  i_0: 0
  i_1: 0
  i_2: 0
  i_3: 0
  v_0: 5.61937e-06 0.0212581 0.150338 0.222903
  v_1: 0.241989 0.0474224 0.128744 0.0995021
  v_2: 0.0657265 0.185878 0.0223869 0.140097
  v_3: 0.145557 0.202392 0.14798 0.127928

FFM: ::

  bias: 0
  i_0: 0
  i_1: 0
  i_2: 0
  i_3: 0
  v_0_0: 5.61937e-06 0.0212581 0.150338 0.222903
  v_0_1: 0.241989 0.0474224 0.128744 0.0995021
  v_0_2: 0.0657265 0.185878 0.0223869 0.140097
  v_0_3: 0.145557 0.202392 0.14798 0.127928
  v_1_0: 0.219158 0.248771 0.181553 0.241653
  v_1_1: 0.0742756 0.106513 0.224874 0.16325
  v_1_2: 0.225384 0.240383 0.0411782 0.214497
  v_1_3: 0.226711 0.0735065 0.234061 0.103661
  v_2_0: 0.0771142 0.128723 0.0988574 0.197446
  v_2_1: 0.172285 0.136068 0.148102 0.0234075
  v_2_2: 0.152371 0.108065 0.149887 0.211232
  v_2_3: 0.123096 0.193212 0.0179155 0.0479647
  v_3_0: 0.055902 0.195092 0.0209918 0.0453358
  v_3_1: 0.154174 0.144785 0.184828 0.0785329
  v_3_2: 0.109711 0.102996 0.227222 0.248076
  v_3_3: 0.144264 0.0409806 0.17463 0.083712

预测结果的输出
----------------------------------------

用户可以通过 ``-o`` 选项来指定预测结果输出文件的路径。例如: ::

  ./xlearn_predict ./small_test.txt ./small_train.txt.model -o output.txt  
  head -n 5 ./output.txt

  -2.01192
  -0.0657416
  -0.456185
  -0.170979
  -1.28849

在默认的情况下，预测结果输出文件的路径格式是当前文件夹下的 ``test_data_name`` + ``.out`` 文件。

选择机器学习算法
----------------------------------------

目前，xLearn 可以支持三种不同的机器学习算法，包括了线性模型 (LR)、factorization machine (FM)，以及 field-aware factorization machine (FFM).

用户可以通过 ``-s`` 选项来选择不同的算法: ::

  ./xlearn_train ./small_train.txt -s 0  # Classification: Linear model (GLM) 
  ./xlearn_train ./small_train.txt -s 1  # Classification: Factorization machine (FM) 
  ./xlearn_train ./small_train.txt -s 2  # Classification: Field-awre factorization machine (FFM) 

  ./xlearn_train ./small_train.txt -s 3  # Regression: Linear model (GLM) 
  ./xlearn_train ./small_train.txt -s 4  # Regression: Factorization machine (FM) 
  ./xlearn_train ./small_train.txt -s 5  # Regression: Field-awre factorization machine (FFM) 

对于 LR 和 FM 算法而言，我们的输入数据格式必须是 ``CSV`` 或者 ``libsvm``. 对于 FFM 算法，我们的输入数据必须是 ``libffm`` 格式: ::

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

设置 Validation Dataset（验证集）
----------------------------------------

在机器学习中，我们可以通过 Validation Dataset (验证集) 来进行超参数调优。在 xLearn 中，用户可以使用 ``-v`` 选项来指定验证集文件，例如: ::

    ./xlearn_train ./small_train.txt -v ./small_test.txt    

下面是程序的一部分输出: ::

  [ ACTION     ] Start to train ...
  [------------] Epoch      Train log_loss       Test log_loss     Time cost (sec)
  [   10%      ]     1            0.571922            0.531160                0.00
  [   20%      ]     2            0.520315            0.542134                0.00
  [   30%      ]     3            0.492147            0.529684                0.00
  [   40%      ]     4            0.470234            0.538684                0.00
  [   50%      ]     5            0.452695            0.537496                0.00
  [   60%      ]     6            0.439367            0.537790                0.00
  [   70%      ]     7            0.425216            0.534396                0.00
  [   80%      ]     8            0.416215            0.542883                0.00
  [   90%      ]     9            0.404673            0.547597                0.00

我们可以看到，在这个任务中 ``Train log_loss`` 在不断的下降，而 ``Test log_loss`` (validation loss) 则是先下降，后上升。这代表当前我们训练的模型已经 overfit （过拟合）我们的训练数据。

在默认的情况下，xLearn 会在每一轮 epoch 结束后计算 validation loss 的数值，而用户可以使用 ``-x`` 选项来制定不同的评价指标。对于分类任务而言，评价指标有： ``acc`` (accuracy), ``prec`` (precision), ``f1``, 以及 ``auc``，例如: ::

    ./xlearn_train ./small_train.txt -v ./small_test.txt -x acc
    ./xlearn_train ./small_train.txt -v ./small_test.txt -x prec
    ./xlearn_train ./small_train.txt -v ./small_test.txt -x f1
    ./xlearn_train ./small_train.txt -v ./small_test.txt -x auc

对于回归任务而言，评价指标包括：``mae``, ``mape``, 以及 ``rmsd`` (或者叫作 ``rmse``)，例如: ::

    cd demo/house_price/
    ../../xlearn_train ./house_price_train.txt -s 3 -x rmse --cv
    ../../xlearn_train ./house_price_train.txt -s 3 -x rmsd --cv

注意，这里我们通过设置 ``--cv`` 选项使用了 *Cross-Validation (交叉验证)* 功能, 我们将在下一节详细介绍该功能。

Cross-Validation (交叉验证)
----------------------------------------

在机器学习中，Cross-Validation (交叉验证) 是一种被广泛使用的模型超参数调优技术。在 xLearn 中，用户可以使用 ``--cv`` 
选项来使用交叉验证功能，例如: ::

    ./xlearn_train ./small_train.txt --cv

在默认的情况下，xLearn 使用 3-folds 交叉验证 (即将数据集平均分成 3 份)，用户也可以通过 ``-f`` 选项来指定数据划分的份数，例如: ::
    
    ./xlearn_train ./small_train.txt -f 5 --cv

上述命令将数据集划分成为 5 份，并且 xLearn 会在最后计算出平均的 validation loss: ::

     ...
    [------------] Average log_loss: 0.549417
    [ ACTION     ] Finish Cross-Validation
    [ ACTION     ] Clear the xLearn environment ...
    [------------] Total time cost: 0.03 (sec)

选择优化算法
----------------------------------------
 
在 xLearn 中，用户可以通过 ``-p`` 选项来选择使用不同的优化算法。目前，xLearn 支持 ``SGD``, ``AdaGrad``, 以及 ``FTRL`` 这三种优化算法。
在默认的情况下，xLearn 使用 ``AdaGrad`` 优化算法: ::

    ./xlearn_train ./small_train.txt -p sgd
    ./xlearn_train ./small_train.txt -p adagrad
    ./xlearn_train ./small_train.txt -p ftrl

相比于传统的 ``SGD`` (随机梯度下降) 算法，``AdaGrad`` 可以自适应的调整学习速率 learning rate，对于不常用的参数进行较大的更新，对于常用的参数进行较小的更新。
正因如此，``AdaGrad`` 算法常用于稀疏数据的优化问题上。除此之外，相比于 ``AdaGrad``，``SGD`` 对学习速率的大小更敏感，这增加了用户调参的难度。

``FTRL`` (Follow-the-Regularized-Leader) 同样被广泛应用于大规模稀疏数据的优化问题上。相比于 ``SGD`` 和 ``AdaGrad``, ``FTRL`` 需要用户调试更多的超参数，我们将在下一节详细介绍 xLearn 的超参数调优。

超参数调优
----------------------------------------

在机器学习中，*hyper-parameter* (超参数) 是指在训练之前设置的参数，而模型参数是指在训练过程中更新的参数。超参数调优通常是机器学习训练过程中不可避免的一个环节。

首先，``learning rate`` (学习速率) 是机器学习中的一个非常重要的超参数，用来控制每次模型迭代时更新的步长。在默认的情况下，这个值在 xLearn 中被设置为 ``0.2``，用户可以通过 ``-r`` 选项来改变这个值: ::

    ./xlearn_train ./small_train.txt -v ./small_test.txt -r 0.1
    ./xlearn_train ./small_train.txt -v ./small_test.txt -r 0.5
    ./xlearn_train ./small_train.txt -v ./small_test.txt -r 0.01

用户还可以通过 ``-b`` 选项来控制 regularization (正则项)。xLearn 使用 ``L2`` 正则项，这个值被默认设置为 ``0.00002``: ::

    ./xlearn_train ./small_train.txt -v ./small_test.txt -r 0.1 -b 0.001
    ./xlearn_train ./small_train.txt -v ./small_test.txt -r 0.1 -b 0.002
    ./xlearn_train ./small_train.txt -v ./small_test.txt -r 0.1 -b 0.01

对于 ``FTRL`` 算法来说，除了学习速率和正则项，我们还需要调节其他的超参数，包括：``-alpha``, ``-beta``, ``-lambda_1`` 和 ``-lambda_2``，例如: ::

    ./xlearn_train ./small_train.txt -p ftrl -alpha 0.002 -beta 0.8 -lambda_1 0.001 -lambda_2 1.0

对于 FM 和 FFM 模型，用户需要通过 ``-k`` 选项来设置 *latent vector* (隐向量) 的长度。在默认的情况下，xLearn 将其设置为 ``4``: ::

    ./xlearn_train ./small_train.txt -s 1 -v ./small_test.txt -k 2
    ./xlearn_train ./small_train.txt -s 1 -v ./small_test.txt -k 4
    ./xlearn_train ./small_train.txt -s 1 -v ./small_test.txt -k 5
    ./xlearn_train ./small_train.txt -s 1 -v ./small_test.txt -k 8

注意，xLearn 使用了 *SSE* 硬件指令来加速向量运算，该指令会同时进行向量长度为 ``4`` 的运算，因此将 ``k=2`` 和 ``k=4`` 所需的运算时间是相同的。

除此之外，对于 FM 和 FFM，用户可以通过设置超参数 ``-u`` 来调节模型的初始化参数。在默认的情况下，这个值被设置为 ``0.66``: ::

    ./xlearn_train ./small_train.txt -s 1 -v ./small_test.txt -u 0.80
    ./xlearn_train ./small_train.txt -s 1 -v ./small_test.txt -u 0.40
    ./xlearn_train ./small_train.txt -s 1 -v ./small_test.txt -u 0.10

迭代次数 & Early-Stop (提前终止)
----------------------------------------

在模型的训练过程中，每一个 epoch 都会遍历整个训练数据。在 xLearn 中，用户可以通过 ``-e`` 选项来设置需要的 epoch 数量: ::

    ./xlearn_train ./small_train.txt -e 3
    ./xlearn_train ./small_train.txt -e 5
    ./xlearn_train ./small_train.txt -e 10   

如果用户设置了 validation dataset (验证集)，xLearn 在默认情况下会在得到最好的 validation 结果时进行 early-stop (提前终止训练)，例如: ::
  
    ./xlearn_train ./small_train.txt -s 2 -v ./small_test.txt -e 10

在上述命令中，我们设置 epoch 的大小为 ``10``，但是 xLearn 会在第 7 轮提前停止训练 (你可能在你的本地计算机上会得到不同的轮次): ::

   ...
  [ ACTION     ] Early-stopping at epoch 7
  [ ACTION     ] Start to save model ...

用户可以通过 ``-sw`` 来设置提前停止机制的窗口大小。即，``-sw=2`` 意味着如果在后两轮的时间窗口之内都没有比当前更好的验证结果，则停止训练，并保存之前最好的模型: ::

    ./xlearn_train ./small_train.txt -e 10 -v ./small_test.txt -sw 3

用户可以通过 ``--dis-es`` 选项来禁止 early-stop: ::

    ./xlearn_train ./small_train.txt -s 2 -v ./small_test.txt -e 10 --dis-es

在上述命令中，xLearn 将进行完整的 10 轮 epoch 训练。

注意，在默认情况下，如果没有设置 metric，则 xLearn 会通过 test_loss 来选择最佳停止时机。如果设置了 metric，则 xLearn 通过 metric 的值来决定停止时机。 

无锁 (Lock-free) 学习
----------------------------------------

在默认情况下，xLearn 会进行 *Hogwild!* 无锁学习，该方法通过 CPU 多核进行并行训练，提高 CPU 利用率，加快算法收敛速度。但是，该无锁算法是非确定性的算法 (*non-deterministic*). 即，如果我们多次运行如下的命令，我们会在每一次运行得到略微不同的 loss 结果: ::

   ./xlearn_train ./small_train.txt 

   The 1st time: 0.396352

   ./xlearn_train ./small_train.txt 

   The 2nd time: 0.396119

   ./xlearn_train ./small_train.txt 

   The 3nd time: 0.396187

用户可以通过 ``-nthread`` 选项来设置使用 CPU 核心的数量，例如: ::

   ./xlearn_train ./small_train.txt -nthread 2

上述命令指定使用 2 个 CPU Core 来进行模型训练。如果用户不设置该选项，xLearn 在默认情况下会使用全部的 CPU 核心进行计算。

用户可以通过设置 ``--dis-lock-free`` 选项禁止多核无锁学习: ::

  ./xlearn_train ./small_train.txt --dis-lock-free

这时，xLearn 计算的结果是确定性的 (*determinnistic*): ::

   ./xlearn_train ./small_train.txt 

   The 1st time: 0.396372

   ./xlearn_train ./small_train.txt 

   The 2nd time: 0.396372

   ./xlearn_train ./small_train.txt 

   The 3nd time: 0.396372

使用 ``--dis-lock-free`` 的缺点是训练速度会比无锁训练慢很多，我们的建议是在大规模数据训练下开启此功能。

Instance-Wise 归一化
----------------------------------------

对于 FM 和 FFM 来说，xLearn 会默认对特征进行 *Instance-Wise Normalizarion* (归一化). 在一些大规模稀疏数据的场景 (例如 CTR 预估), 这一技术非常的有效，但是有些时候它也会影响模型的准确率。用户可以通过设置 ``--no-norm`` 来关掉该功能: ::

  ./xlearn_train ./small_train.txt -s 1 -v ./small_test.txt --no-norm

Quiet Model 安静模式
----------------------------------------

xLearn 的训练支持安静模式，在安静模式下，用户通过调用 ``--quiet()`` 选项来使得 xLearn 的训练过程不会计算任何评价指标，这样可以很大程度上提高训练速度: ::

  ./xlearn_train ./small_train.txt -e 10 --quiet

xLearn 还可以支持 Python API，我们将在下一节详细介绍。
