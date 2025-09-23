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
├── rbm_train.py                                # 模型训练代码
├── noise_process.py                            # 噪声处理代码
├── qubo_inference.py                           # 去噪推理代码
├── evaluate.py                                 # 评估代码
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

​	数据预处理后会生成12x12的二值化向量数据，分为训练和测试两种，每种各含特征文件和标签文件，之后通过“train”模式运行脚本，执行受限玻尔兹曼机（RBM）的训练：

```bash
python main.py --mode=train
```

​	RBM训练完成后会生成权重和两层偏置共三个参数文件，首先通过“process”模式运行脚本，对选择序号的数据添加椒盐噪声：

```bash
python main.py --mode=process
```

​	数据添加噪声后会生成原图像和噪声图像，然后通过“inference”模式运行脚本，选择“sigma”参数模式或“rho”参数模式决定去噪惩罚参数，执行数据去噪推理：

```bash
python main.py --mode=inference --rho_mode=sigma
```

​	推理出去噪数据后会生成去噪图像，方便使用者进行对比。最后通过“evaluate”模式运行脚本，计算重叠比例，定量评估去噪效果（注意：和推理步骤一样，需要选择参数模式以固定在不同的 $\sigma$ 和对应 $\rho$ 下进行比较）：

```bash
python main.py --mode=evaluate --rho_mode=sigma
```

​	代码各函数参数可在参数文件params.py中进行统一调整。

## 算法说明

​	惩罚参数 $\rho$ 的最优选择算法。

​	和

## 作者信息
- 作者姓名：周澍锦
- 联系方式：Your_beatitude@189.cn

