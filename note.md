
# 特征提取
所有鼾声的原始音频文件均为采样频率为 $8000$ Hz单声道 wav 格式的时域波形文件。采用时频分析方法提取音频特征，同时使用短时傅里叶变换，MFCC 两种方法提取出语谱图（Spectrum）、LogMel、MFCC 三种特征，其中 LogMel 特征为 MFCC 特征提取过程中的梅尔滤波器的能量，三种方法的产品都是用 colormap 的方式。把特征矩阵绘制为彩图。

## 分帧

实施三种特征提取方法之前都需要对时域波形信号进行分帧，帧长为 $512$ 采样点、 帧移为$32$ 采样点，每帧都加以汉明窗，然后每帧单独实施傅里叶变换和计算MFCC，将得到的结果再以时间顺序拼接以得到时域信息。

## Spectrum
对每帧时域波形信号进行点数为 $1024$ 的傅里叶变换。得到每帧的振幅

## LogMel/MFCC

先把傅里叶变换得到的振幅转化为能量值，从傅里叶变换的到的振幅计算为能量的公式如下：
<center>

$E = |A|^2$
</center>

其中 $A$ 为振幅， $E$ 为能量。

设置 $128$ 个梅尔滤波器，收集从傅里叶变换得到的振幅计算出的能量，每个滤波器收集的能量值取以 $2$ 为底的对数。
<center>

$LogMel_i = log_2(Mel_i)$
</center>

其中 $Mel_i$ 为每个梅尔滤波器收集到的能量。

每个 $LogMel_i$ 组成一帧 LogMel ，把每帧能量按时间顺序拼接就得到整个声音的 LogMel。

把每帧的 LogMel 做 $64$ 阶 DCT 变换，即可得到 MFCC。


# 特征预处理

特征的彩图均为 $99\times 73$ 的 RGB 图片，转化为灰度图后按行展开为 $7227$ 维的向量，灰度的计算方法如下：
<center>

$Grey = 0.299\times R + 0.587\times B + 0.114\times G$

</center>

所有样本进行主成分分析法降维，降维参数为保留样本矩阵的 $85\%$ 的方差。 LogMel、Spectrum、MFCC 分别降至460、198、574维。其中 Spectrum 特征的降维效果最好，说明特征信息集中在更少的维度上。


以下对每一个特征-分类器组合都进行了分析，先对每种特征的三种分类器进行对比分析，再对比每种特征的最好结果。为了描述简洁，以下对本问题中研究的上阻塞、下阻塞分别成为正类、负类。

# 分类

## Spectrum 特征

确定了特征和分类器后，考虑用 sklearn 库的 Gridsearchcv 函数来搜索最优参数，其做法是对给出范围内的各种参数组合进行暴力枚举后测试，所用的测试方法为5折交叉验证，评价标准为测试集上分类正确率。

各个分类器的搜索范围如下

* SVM-rbf

    `C`: 1,2,3,4,5

    `class_weight`: {1: 1, 0: 2}, {1: 1, 0: 1}, {1: 1, 0: 1.5}

* SVM-poly

    `C`: 1,2,3,4,5

    `degree`: 1,2,3,4

    `class_weight`: {1: 1, 0: 2}, {1: 1, 0: 1}, {1: 1, 0: 1.5}

* RandomForest

    `n_estimators`: 50,100,150,200,250,300

    `max_depth`: 5,10,15,20,25,30


SVM 的 `gamma` 参数在手动尝试测试时发现 设置为 'scale' （即根据样本 X 的方差和样本个数进行计算得出的值）都优于所有手动尝试的结果。


搜索的出的最优参数如下

* SVM-rbf

    `C`: 4

    `class_weight`: {1: 1, 0: 1}

* SVM-poly

    `C`: 2

    `degree`: 2

    `class_weight`: {1: 1, 0: 1}

* RandomForest

    `n_estimators`: 200

    `max_depth`: 10

    `class_weight`: {1: 1, 0: 1}

由于两类样本的不均衡，对于最优参数还手动调整两类权重，进行10折交叉验证，用训练集训练以上最优参数的分类器，记录测试集上的正类分类正确率、负类分类正确率和综合分类正确率，结果如下表,其中 class_weight 为 正类 : 负类 。
<center>

### SVM

|核函数|class_weight|up|low|all
|:---:|:---:|:---:|:---:|:---:|
|rbf|1:1|90.50%|68.78%|81.59%|
|rbf|1:1.5|89.10%|69.53%|81.04%|
|rbf|1:2|88.84%|69.44%|80.85%|
|rbf|1:4|88.30%|70.37%|80.93%|
|rbf|1:8|88.30%|70.37%|80.93%|
|poly|1:1|90.83%|69.74%|82.22%|
|poly|1:1.5|85.15%|77.44%|82.02%|
|**poly**|**1:2**|**81.00%**|**80.16%**|**80.65%**|

</center>
<center>

### RandomForests

|class_weight|up|low|all|
|:---:|:---:|:---:|:---:|:---:|
|1:1|96.17%|52.69%|78.42%|
|1:1.5|91.70%|58.99%|78.30%|
|1:2|86.29%|66.46%|78.10%|
|**1:3**|**74.67%**|**76.04%**|**75.04%**|
|1:4|63.98%|81.73%|71.07%|

</center>
从以上结果可以看到，当两类权重为 1:1 时，综合正确率虽然是最高的，但是在两类上的正确率悬殊，这说明综合正确率高并不一定说明分类的综合效果好。

但可以看到随着负类权重的提高，两类正确率的差距在缩小，其中 poly 核 SVM 在 权重为 1:2 时，两类的正确率几乎得到了平衡，并且综合正确率也在 $80.65\%$ 并未下降太多。而 rbf 核 SVM 的两类正确率在负类权重提高到一定程度后，就不在改变了，无法达到平衡。RandomForest 则在 1:3 时接近平衡，但综合正确率只有 $75.04\%$。

现为各个分类器选定最优参数如下

* SVM-rbf

    `C`: 4

    `class_weight`: {1: 1, 0: 2}

* SVM-poly

    `C`: 2

    `degree`: 2

    `class_weight`: {1: 1, 0: 2}

* RandomForest

    `n_estimators`: 200

    `max_depth`: 10

    `class_weight`: {1: 1, 0: 3}

### ROC 曲线分析
用以上测试出的最优权重最优参数对三种分类器进行10折交叉验证，其中作出各自的 ROC 曲线。
<center>

#### **SVM-rbf**
<img src="./Spec/rbf_ROC.png" width = "60%" alt="rbf-ROC" align=center />


#### **SVM-poly**

<img src="./Spec/poly_ROC.png" width = "60%" alt="poly-ROC" align=center />


#### **RandomForest**

<img src="./Spec/ran_ROC.png" width = "60%" alt="ran-ROC" align=center />

</center>

无论是从 ROC 曲线形状还是平均 AUC 值来看，都是 poly 核 SVM 的表现更加出色，与其综合正确率的表现相匹配。

### 样本分布

下面从样本在决策面附近的分布的情况来评估各个分类器的性能。

先以所有数据来训练各个分类器后，画出所有数据的分布直方图，SVM 分类器中横轴为样本到决策超平面的归一化几何距离，RandomForest 分类器中为样本为正类的概率。
<center>

#### **SVM-rbf**

<img src="./Spec/rbf_DataDistri.png" width = "50%" alt="rbf-ROC" align=center /><img src="./Spec/rbf_DataDistri_lim.png" width = "50%" alt="rbf-ROC" align=center />

#### **SVM-poly**

<img src="./Spec/poly_DataDistri.png" width = "50%" alt="rbf-ROC" align=center /><img src="./Spec/poly_DataDistri_lim.png" width = "50%" alt="rbf-ROC" align=center />

#### **RandomForest**

<img src="./Spec/ran_DataDistri.png" width = "60%" alt="rbf-ROC" align=center />

</center>

从以上可以看出各个分类器都把样本显著分开，但这是把所有的样本都用于训练，并未能代表分类器拥有泛化能力。而且从 SVM 中的分布来看，$91.5\%$ 的样本分布在 $-1$ 和 $1$ 附近，说明有过拟合的可能。

以下在把数据随机划分 $75.5\%$ 为训练集，余下为测试集，以训练集训练分类器后，作出测试集的样本分布。每个分类器重复随机划分三次作出三张分布图。

<center>

#### **SVM-rbf**
<img src="./Spec/rbf_DataDistri_test6.png" width = "30%" alt="rbf-ROC" align=center /><img src="./Spec/rbf_DataDistri_test8.png" width = "30%" alt="rbf-ROC" align=center /><img src="./Spec/rbf_DataDistri_test66.png" width = "30%" alt="rbf-ROC" align=center />


#### **SVM-poly**

<img src="./Spec/poly_DataDistri_test1.png" width = "30%" alt="rbf-ROC" align=center /><img src="./Spec/poly_DataDistri_test6.png" width = "30%" alt="rbf-ROC" align=center /><img src="./Spec/poly_DataDistri_test66.png" width = "30%" alt="rbf-ROC" align=center />


#### **RandomForest**

<img src="./Spec/ran_DataDistri_test1.png" width = "30%" alt="rbf-ROC" align=center /><img src="./Spec/ran_DataDistri_test6.png" width = "30%" alt="rbf-ROC" align=center /><img src="./Spec/ran_DataDistri_test66.png" width = "30%" alt="rbf-ROC" align=center />

</center>

从测试集样本的分布来看，poly 核 SVM 的分类能力是最好的，两类的分布明显有各自的分布中心，并且样本在中心比较聚集，两类仅有少部分重叠，与其10折交叉验证的 $80.65\%$ 正确率相匹配。相比之下 rbf 中两类虽也有明显分布中心，但负类分布比较分散，这也与其负类准确率偏低相契合，RandomForest 中的负类分布也是如此。同时三个分类器的正类分布都大致相同。负类的分散是导致效果差距的主要原因。

## LogMel 特征

对 LogMel 特征也进行同上的测试，也得到了相似的结论，但分类效果要稍微和差一些。

在同上的参数搜索范围内搜索出的最优参数如下

* SVM-rbf

    `C`: 4

    `class_weight`: {1: 1, 0: 1}

* SVM-poly

    `C`: 2

    `degree`: 2

    `class_weight`: {1: 1, 0: 1}

* RandomForest

    `n_estimators`: 100

    `max_depth`: 10

    `class_weight`: {1: 1, 0: 1.5}


调整两类权重进行测试

<center>

### SVM

|核函数|class_weight|up|low|all
|:---:|:---:|:---:|:---:|:---:|
|rbf|1:1|89.58%|66.75%|80.30%|
|rbf|1:1.5|89.40%|66.97%|80.25%|
|rbf|1:2|89.26%|66.93%|80.18%|
|rbf|1:4|89.19%|67.02%|80.18%|
|rbf|1:8|89.19%|67.02%|80.18%|
|poly|1:1|91.29%|66.72%|81.24%|
|**poly**|**1:1.5**|**78.09%**|**78.06%**|**77.98%**|
|poly|1:2|73.33%|79.76%|75.82%|


### RandomForestClassifier
|class_weight|up|low|all|
|:---:|:---:|:---:|:---:|:---:|
|1:1|96.10%|45.35%|75.51%|
|1:1.5|89.94%|58.86%|77.28%|
|1:2|81.55%|65.80%|75.12%|
|**1:2.5**|**73.49%**|**71.94%**|**72.76%**|
|1:3|66.18%|77.08%|70.49%|
|1:4|55.18%|84.12%|66.76%|

</center>

最好的是两类正确率平衡后的 poly 核 SVM ，综合正确率为 $77.98\%$

为各个分类器选定最优参数如下

* SVM-rbf

    `C`: 4

    `class_weight`: {1: 1, 0: 2}

* SVM-poly

    `C`: 2

    `degree`: 2

    `class_weight`: {1: 1, 0: 1.5}

* RandomForest

    `n_estimators`: 100

    `max_depth`: 10

    `class_weight`: {1: 1, 0: 2.5}

### ROC 曲线分析


<center>

#### **SVM-rbf**

<img src="./LogMel/rbf_ROC.png" width = "60%" alt="rbf-ROC" align=center />



#### **SVM-poly**

<img src="./LogMel/poly_ROC.png" width = "60%" alt="poly-ROC" align=center />


#### **RandomForest**

<img src="./LogMel/ran_ROC.png" width = "60%" alt="ran-ROC" align=center />

</center>

### 样本分布

<center>

#### **SVM-rbf**
<img src="./LogMel/rbf_DataDistri.png" width = "50%" alt="rbf-ROC" align=center /><img src="./LogMel/rbf_DataDistri_lim.png" width = "50%" alt="rbf-ROC" align=center />

#### **SVM-poly**

<img src="./LogMel/poly_DataDistri.png" width = "50%" alt="rbf-ROC" align=center /><img src="./LogMel/poly_DataDistri_lim.png" width = "50%" alt="rbf-ROC" align=center />

#### **RandomForest**

<img src="./LogMel/ran_DataDistri.png" width = "60%" alt="rbf-ROC" align=center />

</center>


把数据随机划分 $75.5\%$ 为训练集，余下为测试集，以训练集训练分类器后，作出测试集的样本分布。每个分类器重复三次作出三张分布图。

* SVM-rbf
<center>
    <img src="./LogMel/rbf_DataDistri_test1.png" width = "30%" alt="rbf-ROC" align=center /><img src="./LogMel/rbf_DataDistri_test6.png" width = "30%" alt="rbf-ROC" align=center /><img src="./LogMel/rbf_DataDistri_test66.png" width = "30%" alt="rbf-ROC" align=center />
</center>

* SVM-poly
<center>
    <img src="./LogMel/poly_DataDistri_test1.png" width = "30%" alt="rbf-ROC" align=center /><img src="./LogMel/poly_DataDistri_test6.png" width = "30%" alt="rbf-ROC" align=center /><img src="./LogMel/poly_DataDistri_test66.png" width = "30%" alt="rbf-ROC" align=center />
</center>

* RandomForest
<center>
    <img src="./LogMel/ran_DataDistri_test1.png" width = "30%" alt="rbf-ROC" align=center /><img src="./LogMel/ran_DataDistri_test6.png" width = "30%" alt="rbf-ROC" align=center /><img src="./LogMel/ran_DataDistri_test66.png" width = "30%" alt="rbf-ROC" align=center />
</center>

从综合正确率、ROC 曲线、平均AUC值，样本分布来看，用同样的分类方法，LogMel 特征的效果比 Spectrum 特征稍差。

## MFCC 特征

在同上的参数搜索范围内搜索出的最优参数如下

* SVM-rbf

    `C`: 2

    `class_weight`: {1: 1, 0: 1}

* SVM-poly

    `C`: 1

    `degree`: 1

    `class_weight`: {1: 1, 0: 1}

* RandomForest

    `n_estimators`: 250

    `max_depth`: 5

    `class_weight`: {1: 1, 0: 1.5}


调整两类权重进行测试
<center>

### SVM

|核函数|class_weight|up|low|all
|:---:|:---:|:---:|:---:|:---:|
|rbf|1:1|88.82%|60.18%|77.24%|
|rbf|1:1.5|86.80%|62.35%|76.88%|
|rbf|1:2|85.82%|63.02%|76.57%|
|rbf|1:4|85.43%|62.77%|76.22%|
|rbf|1:8|85.43%|62.77%|76.22%|
|poly|1:1|86.66%|64.45%|77.70%|
|**poly**|**1:1.5**|**78.23%**|**73.39%**|**76.29%**|
|poly|1:1.7|75.15%|74.48%|75.00%|
|poly|1:2|71.61%|77.57%|74.02%|

### RandomForestClassifier
|class_weight|up|low|all|
|:---:|:---:|:---:|:---:|:---:|
|1:1|99.35%|16.98%|65.93%|
|1:1.5|82.58%|65.83%|75.55%|
|**1:1.6**|**72.58%**|**73.51%**|**72.72%**|
|1:1.7|62.88%|81.41%|70.21%|
|1:2|34.39%|93.40%|58.16%|

</center>

为各个分类器选定最优参数如下

* SVM-rbf

    `C`: 2

    `class_weight`: {1: 1, 0: 2}

* SVM-poly

    `C`: 1

    `degree`: 1

    `class_weight`: {1: 1, 0: 1.7}

* RandomForest

    `n_estimators`: 250

    `max_depth`: 5

    `class_weight`: {1: 1, 0: 1.6}

### ROC 曲线分析

<center>

#### SVM-rbf

<img src="./MFCC/rbf_ROC.png" width = "60%" alt="rbf-ROC" align=center />

#### **SVM-poly**

<img src="./MFCC/poly_ROC.png" width = "60%" alt="poly-ROC" align=center />

#### **RandomForest**

<img src="./MFCC/ran_ROC.png" width = "60%" alt="ran-ROC" align=center />

</center>

### 样本分布
<center>

#### **SVM-rbf**
<img src="./MFCC/rbf_DataDistri.png" width = "50%" alt="rbf-ROC" align=center /><img src="./MFCC/rbf_DataDistri_lim.png" width = "50%" alt="rbf-ROC" align=center />

#### **SVM-poly**
<img src="./MFCC/poly_DataDistri.png" width = "50%" alt="rbf-ROC" align=center /><img src="./MFCC/poly_DataDistri_lim.png" width = "50%" alt="rbf-ROC" align=center />


#### **RandomForest**

<img src="./MFCC/ran_DataDistri.png" width = "60%" alt="rbf-ROC" align=center />

</center>


把数据随机划分 $75.5\%$ 为训练集，余下为测试集，以训练集训练分类器后，作出测试集的样本分布。每个分类器重复三次作出三张分布图。

<center>

#### **SVM-rbf**
<img src="./MFCC/rbf_DataDistri_test1.png" width = "30%" alt="rbf-ROC" align=center /><img src="./MFCC/rbf_DataDistri_test6.png" width = "30%" alt="rbf-ROC" align=center /><img src="./MFCC/rbf_DataDistri_test66.png" width = "30%" alt="rbf-ROC" align=center />


#### **SVM-poly**

<img src="./MFCC/poly_DataDistri_test1.png" width = "30%" alt="rbf-ROC" align=center /><img src="./MFCC/poly_DataDistri_test6.png" width = "30%" alt="rbf-ROC" align=center /><img src="./MFCC/poly_DataDistri_test66.png" width = "30%" alt="rbf-ROC" align=center />


#### **RandomForest**

<img src="./MFCC/ran_DataDistri_test1.png" width = "30%" alt="rbf-ROC" align=center /><img src="./MFCC/ran_DataDistri_test6.png" width = "30%" alt="rbf-ROC" align=center /><img src="./MFCC/ran_DataDistri_test66.png" width = "30%" alt="rbf-ROC" align=center />

</center>


## 特征比较

在以上的测试中，三种特征中分类最好的都是 poly 核 SVM，以下对比各个特征使用 poly 核 SVM 的表现。

以下图片均以从左往右分别为 Spectrum 、LogMel 、 MFCC 特征的顺序排列。
<center>

### ROC


<img src="./Spec/poly_ROC.png" width = "30%" alt="rbf-ROC" align=center /><img src="./LogMel/poly_ROC.png" width = "30%" alt="rbf-ROC" align=center /><img src="./MFCC/poly_ROC.png" width = "30%" alt="rbf-ROC" align=center />

</center>

可以看到 Spectrum 特征的 ROC 曲线最接近直角折线，同时看到下表的10折交叉验证的平均 AUC 也是 Spectrum 的 0.886 最高。

<center>

### AUC
|Spectrum|LogMel|MFCC|
|:-:|:-:|:-:|
|**0.886**|0.860|0.821|

### 随机测试集样本分布

<img src="./Spec/poly_DataDistri_test1.png" width = "30%" alt="rbf-ROC" align=center /><img src="./LogMel/poly_DataDistri_test66.png" width = "30%" alt="rbf-ROC" align=center /><img src="./MFCC/poly_DataDistri_test66.png" width = "30%" alt="rbf-ROC" align=center />

</center>

从随机划分 $25\%$ 的测试集上的样本分别来看， Spectrum 中的两类更加分离，并且在各自的中心聚集，从重合部分占比更小，相比之下 LogMel 特征中的负类更加分散，从而重合部分占比更大，这个负类分散导致重合部分占比上升在 MFCC 特征中的分布里表现得更加明显。

# 结论

在上面的测试， Spectrum 特征最优分类表现在三种特征的最优分类表现中是最好的，同时 poly 核 SVM 在对三种特征进行分类中表现都优与 rbf 核 SVM和 RandomForest 。

Spectrum 特征用 poly 核 SVM 进行分类的单类和综合分类平均正确率都高于 $80\%$ ，随机划分测试集上的平均 AUC 值有 $0.886$ 。说明这个特征和分类器组合是能够有效分类的。

## Spce EER
rbf_err: 0.2124248496993988
poly_err: 0.19639278557114226
ran_err: 0.24248496993987975

## LogMel EER
rbf_err: 0.22444889779559116
poly_err: 0.21442885771543085
ran_err: 0.26452905811623245

## MFCC EER
rbf_err: 0.25250501002004005
poly_err: 0.24248496993987975
ran_err: 0.2605210420841683