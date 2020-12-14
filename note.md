
### 用 poly 核SVM C=2 degree=2 
|核函数|class_weight|up|low|all
|:---:|:---:|:---:|:---:|:---:|
|poly|1:1|90.83%|69.74%|82.22%|
|poly|1:1.5|85.15%|77.44%|82.02%|
|poly|1:2|81.00%|80.16%|80.65%|
两类正确率的以平衡

### 用 rbf 核SVM C=4 
|核函数|class_weight|up|low|all
|:---:|:---:|:---:|:---:|:---:|
|rbf|1:1|90.50%|68.78%|81.59%|
|rbf|1:1.5|89.10%|69.53%|81.04%|
|rbf|1:2|88.84%|69.44%|80.85%|
|rbf|1:4|88.30%|70.37%|80.93%|
|rbf|1:8|88.30%|70.37%|80.93%|
无论如何加大负类的权值，也没能平衡两类的正确率

### 用综合表现最好的参数画出ROC曲线

参数：`poly` 核，$C=2，degree=2，class\_weight=1:2$ 

进行10折交叉验证，每次利在测试集上作出一条ROC曲线

<img src="./ROC_with_Poly.png" width = "50%" alt="图片名称" align=center />

### 用decision_function() 画出样本分布直方图

用所有样本训练以下参数的SVM后

参数：`poly` 核，$C=2，degree=2，class\_weight=1:2$ 

用decesion_function() 得到各样本到超平面距离，画出分布

<img src="./DataDistri.png" width = "50%" alt="图片名称" align=center />

放大底部

<img src="./DataDistri_lim.png" width = "50%" alt="图片名称" align=center />

随机划分 75% 的样本作为训练集，25%作为测试集，训练同以上参数的SVM，用同样的方法，作出测试集样本的分布

重复3次

<img src="./DataDistri_test.png" width = "50%" alt="图片名称" align=center />

<img src="./DataDistri_test1.png" width = "50%" alt="图片名称" align=center />

<img src="./DataDistri_test66.png" width = "50%" alt="图片名称" align=center />

可见两类的分布是有各自的中心的

