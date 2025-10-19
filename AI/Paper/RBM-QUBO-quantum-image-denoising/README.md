# 【图像去噪】问题

## 课题简介
​	图像去噪是图像处理和机器学习的基础问题，已有众多经典和数据驱动方法。该论文提出一种基于受限玻尔兹曼机（RBM）、QUBO和量子退火（QA）的二值图像去噪框架，将去噪问题转化为QUBO实例，通过平衡训练好的RBM学习分布与噪声图像偏差的惩罚项，得到去噪目标。

​	本课题旨在复现论文《Quantum Image Denoising: A Framework via Boltzmann Machines, QUBO, and Quantum Annealing》提出的量子图像去噪框架，该框架将二值图像去噪问题转化为QUBO实例，通过模拟退火求解器进行求解，并验证其性能。本项目遵循量子开发实验室的代码规范和项目结构要求。

## 项目结构
```
NBMF-applies-image-feature-learning/            # 项目目录
├── README.md                                   # 项目说明文档
├── requirements.txt                            # 依赖包列表
├── params.py                                   # 参数文件
├── data_pretreat.py                            # 数据预处理代码
├── BAS_data.py                                 # 数据生成代码
├── rbm_train.py                                # 模型训练代码
├── noise_process.py                            # 噪声处理代码
├── qubo_inference.py                           # QUBO方法去噪推理代码
├── other_inference.py                          # 其它方法去噪推理代码
├── qubo_evaluate.py                            # QUBO方法评估代码
├── other_evaluate.py                           # 其它方法评估代码
└── main.py                                     # 主程序入口
```

## 使用方法

1. 安装依赖：

```bash
pip install -r requirements.txt
```

2. 运行示例：

​	在params.py中调整路径参数，导向至数据集。

​	以“pretreat”模式运行脚本，执行数据预处理：

```bash
python main.py --mode=pretreat
```

​	或者数据生成（BAS数据集）：

```bash
python main.py --mode=bas
```

​	数据预处理或数据生成后会生成12x12的二值化向量数据，分为训练和测试两种，每种各含特征文件和标签文件，之后通过“train”模式运行脚本，执行受限玻尔兹曼机（RBM）的训练：

```bash
python main.py --mode=train
```

​	RBM训练完成后会生成权重和两层偏置共三个参数文件，首先通过“process”模式运行脚本，对选择序号的数据添加椒盐噪声：

```bash
python main.py --mode=process
```

​	数据添加噪声后会生成原图像和噪声图像，然后通过“inference”模式运行脚本，选择“sigma”参数模式或“rho”参数模式决定去噪惩罚参数，执行数据去噪推理：

```bash
python main.py --mode=inference --rho_mode=sigma/rho
```

​	或者使用其它对比方法推理去噪图像（使用QUBO方法推理时可以省略“推理模式”后缀，使用其它方法推理时不使用“ $\rho$ 模式”后缀）：

```bash
python main.py --mode=inference --inf_mode=median/gaussian/flow/gibbs
```

​	推理出去噪数据后会生成去噪图像，方便使用者进行对比。最后通过“evaluate”模式运行脚本，计算重叠度，定量评估去噪效果（注意：和推理步骤一样，需要选择参数模式以固定在不同的 $\sigma$ 和对应 $\rho$ 下进行比较，评估QUBO方法时可以省略“推理模式”后缀，评估其它方法时不使用“ $\rho$ 模式”后缀）：

```bash
python main.py --mode=evaluate --rho_mode=sigma/rho
python main.py --mode=evaluate --inf_mode=median/gaussian/flow/gibbs
```

​	代码各函数参数可在参数文件params.py中进行统一调整。

## 算法说明

### 惩罚参数 $\rho$ 的最优选择算法

​	因为将通过预期重叠度指标 $d$ 来评估惩罚参数 $\rho$ 的选取效果，所以参数选择问题可以转化为如下优化模型：

$$\rho = \arg \underset{\rho}{\min} d(P,P') = \arg \underset{\rho}{\min} \mathbb{E}_P \mathbb{E}_{P'}[n - \|X-X'\|_1] \tag{1}$$

​	其中 $X∼P,X^′∼P^′$ 。

​	对于 $X∼P_Q^{model}$ ，将对应的去噪图像 $X^∗_{ρ,\tilde{X}_{X,\sigma},Q}$ 简写为 $X^′$ ，同时将对应的噪声图像 $\tilde{X}_{X,σ}$ 表示为 $\tilde{X}$ ，同时用 $d(P,P^′)$ 表示 $X$ 与 $X^′$ 之间的期望重叠度。则关于选择参数 $ρ$ 的算法可归纳为如下定理：

#### 定理1：

​	设 $X∼P_Q^{model}$ ，其中 $\tilde{X}$ 为含噪图像。通过选取参数 $ρ = \log\frac{1−σ}{σ}$ 计算得到 $X^∗_{ρ,\tilde X_{X,\sigma},Q}$ ，该参数组合在最大化 $X$ 与 $X^∗_{ρ,\tilde{X},Q}$ 的期望重叠度方面具有最优性。

##### 证明

​	对于受椒盐噪声影响的 $X∼P_Q^{model}$ ，由于 $\tilde{X}$ 是 $X$ 通过椒盐噪声——以概率 $σ$ 翻转像素获得的，因此得到条件概率分布：

$$\begin{array}{l}P_\sigma(\tilde{X}=\tilde{x}|X=x) & = \prod_{i=1}^v\{\sigma(\tilde{x}_i-x_i)^2 + (1-\sigma)[1-(\tilde{x}_i-x_i)^2]\} \\ & = \frac{\exp[-\beta_\sigma \sum_{i=1}^v (\tilde{x}_i-x_i)^2]}{(1+e^{-\beta_\sigma})^v}\end{array} \tag{2}$$

​	其中 $β_σ := \log \frac{1−σ}{σ}$ 。为了从有噪声的图像 $\tilde{X}$ 中推断出原始图像 $X$ ，利用贝叶斯公式并计算条件概率$P_{β_σ,Q}^{post}(X=x|\tilde{X}=\tilde{x})$ ，则有：

$$\begin{array}{l}P_{β_σ,Q}^{post}(x|\tilde{x}) & = \frac{P_\sigma(\tilde{X}=\tilde{x}|X=x) P_Q^{model}(x)}{\sum_{\{x\}} P_\sigma(\tilde{x}|x) P_Q^{model}(x)} \\ & = \frac{\exp[-\beta_\sigma \sum_{i=1}^v (\tilde{x}_i-x_i)^2 - \sum_{i,j=1}^{v+h}Q_{ij}x_ix_j]}{\sum_{\{x\}} \exp[-\beta_\sigma\sum_{i=1}^v (\tilde{x}_i-x_i)^2 - \sum_{i,j=1}^{v+h}Q_{ij}x_ix_j]}\end{array} \tag{3}$$

​	注意 $x$ 包含隐藏节点的像素，在这里就这样表示。这里的解析方法是通过在QUBO中加入 $β_σ$ 项进行退火求解后得到分布，然后找到在这种分布下最可能的状态。另外，对于两个图像矢量 $x^∗$ 和 $x$ ，其重叠度由下式给出：

$$m(x^*,x) := \frac{1}{v+h} \sum_{i=1}^{v+h}(2x_i-1)(2x_i^*-1) \tag{4}$$

​	即共享条目的比例。考虑去噪解在噪声条件下的平均值，即 $\bar{X}_{ρ,\tilde{x},Q}$ ：

$$(\bar{X}_{ρ,\tilde{x},Q})_i = \theta(\sum_{\{x\}} P_\bar{Q}^{model}(x)x_i - \frac{1}{2}) \tag{5}$$

​	其中当 $x > 0$ 时 $θ(x) = 1$ ，否则为 $0$ ，需要注意的是，右侧表示是基于 $P_\bar{Q}^{model}(x)$ 的期望推断出的像素值。虽然在形式上区分了 $P_\bar{Q}^{model}(x)$ 和 $P_{\rho,Q}^{post}(x|\tilde{x})$ ，但实际上是相同的。请注意：

$$2(\bar{X}_{ρ,\tilde{x},Q})_i - 1 = \text{sign}(\sum_{\{x\}} P_\bar{Q}^{model}(x)(2x_i - 1)) \tag{6}$$

​	其中 $\text{sign}(x)$ 表示 $x$ 的符号。为简洁起见，令 $α_{σ,Q} := −β_σ \sum_i(\tilde{x}_i−x_i)^2 − \sum_{i,j} Q_{ij}x_ix_j$ 。为了评估解析方法在惩罚项系数 $ρ$ 下的统计性能，对重叠度的平均值计算方式如下：

$$\begin{array}{l}M_{\beta_\sigma,Q}(\rho) &:=& \sum_{\{\tilde{x}\},\{x\}} P_\sigma(\tilde{x}|x) P_Q^{model}(x) m(\bar{X}_{ρ,\tilde{x},Q},x) \\ &=& \frac{1}{(1+e^{-\beta_\sigma})^v} \frac{1}{z} \frac{1}{v+h} \sum_i \sum_{\{\tilde{x}\},\{x\}} e^{\alpha_{\sigma,Q}}[2(\bar{X}_{ρ,\tilde{x},Q})_i-1](2x_i-1)\end{array} \tag{7}$$

​	对上述方程右边的其中一个求和有：

$$\begin{array}{l}\sum_{\{x\}} e^{\alpha_{\sigma,Q}}[2(\mathbb{E}(X^*_{ρ,\tilde{x},Q}))_i-1](2x_i-1) &\le& |\sum_{\{x\}} e^{\alpha_{\sigma,Q}}(2x_i-1)| \\ &=& \sum_{\{x\}} e^{\alpha_{\sigma,Q}}(2x_i-1) \text{sign}(\sum_{\{x'\}} P_{\bar{Q}}^{model}(x')(2x'_i-1)) \\ &=& \sum_{\{x\}} e^{\alpha_{\sigma,Q}} [2(\bar{X}_{\beta_\sigma,\tilde{x},Q})_i-1] (2x_i-1)\end{array} \tag{8}$$

​	因此，对平均重叠度的表达式有：

$$M_{\beta_\sigma,Q}(\rho) \le M_{\beta_\sigma,Q}(\beta_\sigma) \tag{9}$$

​	这种不等式表明，当 $ρ=β_σ=\log \frac{1−σ}{σ}$ 时，平均重叠度达到最大值。

​	该定理基于统计物理学中已知的信息处理理论。值得注意的是， $ρ$ 的最佳选择不依赖于数据分布，而仅取决于噪声水平——在许多实际案例中，噪声水平往往有可靠的估计值。定理证明还揭示了以下推论：

#### 推论：

​	在与定理相同的假设条件下，设置 $ρ=\log \frac{1−σ}{σ}$ 可使 $X^∗_{ρ,\tilde{X},Q}$ 成为原始无噪声图像 $X$ 的最大后验估计量。

##### 注意

​	推论的得出基于以下观察：公式（3）后验分布分子中的能量函数与QUBO目标的形式几乎完全等同，其中 $ρ:=\log \frac{1−σ}{σ}$ 。值得注意的是，最小化QUBO目标等价于最大化后验分布。然而，这一框架允许在选择 $ρ$ 参数时具有额外灵活性——这是标准MAP估计所不具备的。事实上，实践中选择更大的 $ρ$ 可能有利于提升方法的鲁棒性。

​	虽然定理推导了 $ρ$ 的最佳取值，但即便在其假设条件下，也不能保证该方法能改善预期重叠度。以下定理表明当可见单元相互独立时，该图像去噪方法在期望重叠度方面实现了严格的去噪改进。对于 $c > 0$ 和前面的 $P_Q^{model}$ ，令 $\mathcal{I}_c$ 为满足 $|Q_{ii}| > c$ 的索引集合。这些索引对应的 $X$ 分量，其为 $0$ 或 $1$ 的概率都至少为 $\frac{1}{(1+e^{−c})}$ ，具体取决于 $Q_{ii}$ 是正数还是负数。

#### 定理2：

​	假设矩阵 $Q$ 是**对角矩阵**， $X∼P_Q$ ，并且 $\tilde{X}$ 是 $X$ 经过了水平为 $σ$ 的椒盐噪声处理。根据前文定义 $\mathcal{I}_c$ （其中 $c>0$ ），当设定参数满足 $\rho \ge \log(\frac{1-\sigma}{\sigma})$ ，并假设 $\mathcal{I}_ρ \ne ∅$ 时，则去噪图像与真实图像的期望重叠度严格大于含噪图像与真实图像的期望重叠度，即：

$$\mathbb{E}[\sum\mathbb{I}((X^∗_{ρ,\tilde{X},Q})_i = X_i)] > \mathbb{E}[\sum\mathbb{I}(\tilde{X}_i = X_i)] \tag{10}$$

##### 证明

​	令 $\mathcal{I}_c^0 := \{i∈\mathcal{I}_c: Q_{ii} > 0\}, \mathcal{I}_c^1 := \{i∈\mathcal{I}_c: Q_{ii} < 0\}$。直观来说，这些索引分别对应可能取 $0$ 或 $1$ 的情况。进一步地，令 $x^{†i}$ 表示通过翻转 $x$ 中第 $i$ 个元素得到的向量，有 $|f_Q(x)−f_Q(x^{†i})|=|Q_{ii}|>c$ 当且仅当 $i∈\mathcal{I}_c$ 。因此，这表明去噪图像 $x^∗$ 通过设置 $x^∗_i = 1 \quad ∀i∈\mathcal{I}_ρ^1,x^∗_i = 0 \quad ∀i∈\mathcal{I}_ρ^0$，以及 $x^∗_i=\tilde{x}_i \quad \text{otherwise}$ 来求解QUBO目标，这是因为 $f_Q$ 的值被减超过 $ρ$ ，所以即使像素翻转带来了 $\rho$ 的惩罚项，整体惩罚目标函数仍能得到改善。

​	现在，设 $X \sim P_Q^{model}$ 。我们需要计算 $P((X^∗_{ρ,\tilde{X},Q})_i=X_i)$ 的概率。这种情况发生的情形包括： $i∈\mathcal{I}_ρ^0$ 且 $X_i = 0$ ， $i∈\mathcal{I}_ρ^1$ 且 $X_i = 1$ ，或 $i \notin \mathcal{I}_ρ$ 且像素 $i$ 未被噪声翻转。已知当 $i∈\mathcal{I}^b_ρ$ 时， $P(X_i=b) \ge \frac{1}{1+e^{-\rho}} \quad b∈\{0,1\}$ ，因此对于这些情况， $P((X^∗_{ρ,\tilde{X},Q})_i = X_i) ≥ \frac{1}{1+e^{-\rho}}$ 成立。对于 $i \notin \mathcal{I}_ρ$ 的情况，$P((X^∗_{ρ,\tilde{X},Q})_i=X_i)=1−σ$ ，其中 $σ$ 表示像素被噪声翻转的概率。另一方面， $P(\tilde{X}_i=X_i)=1−σ \quad \forall i$ 。通过这些情形进行对公式（10）进行特征描述：

$$\sum P((X^∗_{ρ,\tilde{X},Q})_i=X_i) > \sum P(\tilde{X}_i=X_i) = n \cdot (1-\sigma) \tag{11}$$

​	对于左侧，假设 $\mathcal{I}_ρ \ne ∅$ 有：

$$\sum P((X^∗_{ρ,\tilde{X},Q})_i=X_i) > \sum_{i \in \mathcal{I}_\rho}\frac{1}{1+e^{-\rho}} + \sum_{i \notin \mathcal{I}_\rho}(1-\sigma) = |\mathcal{I}_\rho| \cdot \frac{1}{1+e^{-\rho}} + (n - |\mathcal{I}_\rho|)(1-\sigma) \tag{12}$$

​	所以公式（10）在满足以下条件时成立：

$$|\mathcal{I}_\rho| \ne 0 \quad \text{and} \quad \frac{1}{1+e^{-\rho}} \ge 1-\sigma \Leftrightarrow  \rho \ge \log(\frac{1-\sigma}{\sigma}) \quad \text{and} \quad \mathcal{I}_\rho \ne ∅ \tag{13}$$

​	这个定理就被证明了。

​	矩阵 $Q$ 为对角矩阵的假设等同于 $X$ 各分量相互独立，这在真实数据中并不现实。然而由于在受限玻尔兹曼机模型中，**可见单元在给定隐藏单元的条件下是相互独立的，**仍可认为这种独立性对去噪方法具有参考价值。实际上，若隐藏状态被固定（或已知，或正确恢复），定理2将适用。**假设矩阵 $\mathcal{I}_ρ$ 非空是去噪任务的自然前提**——当 $\mathcal{I}_ρ$ 为空时，矩阵 $Q$ 的元素均值不会较大，这相当于 $X$ 的各分量接近均匀分布。直观来看，如果图像本身呈现噪声特征，自然无法保证能有效进行去噪处理。

## 作者信息
- 作者姓名：周澍锦
- 联系方式：Your_beatitude@189.cn

