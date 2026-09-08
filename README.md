# RolX-torch for Network Dismantling

本仓库是在原 RolX/ReFeX 代码基础上改成的 PyTorch 版本，主要用于论文

```bibtex
@article{zhou2026dismantling,
  title={Dismantling complex networks based on higher-order graph neural network},
  author={Zhou, Wennan and Tan, Suoyi and Fang, Yang and L{\"u}, Xin and Zhao, Xiang},
  journal={Communications Physics},
  year={2026},
  publisher={Nature Publishing Group UK London}
}
```

相关实验中的数据预处理工作。代码的核心功能是：

- 用 ReFeX 从输入网络中提取节点结构特征；
- 用 PyTorch 版 RolX 对节点-结构特征矩阵做分解，得到节点结构角色嵌入；
- 根据每个节点在不同 role 维度上的最大值，生成节点角色标签和 one-hot 角色矩阵，供后续高阶图神经网络或网络拆解实验使用。

## 目录结构

```text
.
├── input/                  # 输入网络边列表，CSV 格式
├── output/
│   ├── features/           # ReFeX 生成的二值结构特征
│   └── embeddings/         # RolX 嵌入、角色分配和角色-特征矩阵
├── src/
│   ├── main.py             # 主入口：提取特征并训练 RolX
│   ├── par.py              # 命令行参数
│   ├── refex.py            # ReFeX 结构特征提取
│   ├── rolx.py             # 当前 main.py 使用的 PyTorch RolX 实现
│   ├── rolx1.py            # 分阶段训练版本，当前未被 main.py 默认导入
│   ├── layers.py           # PyTorch factorization layer
│   └── print_and_read.py   # 读写和训练日志打印工具
├── process_embedding.py    # 将 embedding 转为 hard role 和 one-hot role
└── README.md
```

## 环境准备

建议使用 Python 3.8+。当前测试机器使用的是 Python 3.9。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install numpy pandas scipy networkx tqdm texttable torch
```

如果要用 GPU，请先根据本机 CUDA 版本安装匹配的 PyTorch。代码会自动检测 CUDA：

```text
Using device: cuda
```

如果没有可用 GPU，则会使用 CPU：

```text
Using device: cpu
```

## 输入数据格式

输入文件放在 `input/` 目录下，格式是带表头的 CSV 边列表。每一行是一条无向边：

```csv
node_1,node_2
282,0
980,0
981,0
```

注意事项：

- 当前实现使用 `networkx.from_edgelist` 读取图，默认按无向图处理。
- 推荐节点编号使用从 `0` 开始的连续整数。
- `src/refex.py` 中会调用 `nx.convert_node_labels_to_integers`，如果原始节点编号不是连续整数，输出行号会对应转换后的内部编号，代码不会额外保存原始 id 到内部 id 的映射。

## 快速运行

在项目根目录运行：

```powershell
python src\main.py
```

当前 `src/par.py` 的默认数据集是 `Douban`，并且默认路径写成了本机绝对路径：

```text
D:\experiment\network dismantling\RolX-torch\input\Douban.csv
```

如果把项目移动到其他目录，建议直接在命令行显式传入路径。例如：

```powershell
python src\main.py `
  --input input\Douban.csv `
  --recursive-features-output output\features\Douban_features.csv `
  --embedding-output output\embeddings\Douban_embedding.csv
```

运行流程包括两步：

1. ReFeX 提取递归结构特征，并保存到 `--recursive-features-output`；
2. PyTorch RolX 训练节点嵌入，并保存到 `--embedding-output`。

## 运行其他数据集

例如运行 `Github_Contest.csv`：

```powershell
python src\main.py `
  --input input\Github_Contest.csv `
  --recursive-features-output output\features\Github_Contest_features.csv `
  --embedding-output output\embeddings\Github_Contest_embedding.csv `
  --dimensions 1000 `
  --epochs 5 `
  --batch-size 320 `
  --bins 40 `
  --pruning-cutoff 0.5
```



## 输出文件说明

主程序 `python src\main.py` 会生成：

```text
output/features/<dataset>_features.csv
output/embeddings/<dataset>_embedding.csv
```

其中：

- `<dataset>_features.csv` 是 ReFeX 生成的二值节点结构特征矩阵，行对应节点，列对应二值结构特征；
- `<dataset>_embedding.csv` 是 RolX 学到的节点角色嵌入矩阵，行对应节点，列为 `x_0, x_1, ...`；
- 当前 `src/rolx.py` 还会额外保存 Douban 的角色-特征矩阵到 `output/embeddings/Douban_role_features.csv`。如果处理其他数据集，需要在代码中把这个硬编码路径改成相应数据集路径，或改用 `rolx1.py` 中基于 `--embedding-output` 自动推导文件名的写法。

## 生成节点角色标签

训练得到 embedding 后，可以运行：

```powershell
python process_embedding.py
```

这个脚本会读取：

```text
output/embeddings/Douban_embedding.csv
```

并生成：

```text
output/embeddings/Douban_node_roles.csv
output/embeddings/Douban_node_roles_one_hot.csv
```

含义如下：

- `Douban_node_roles.csv`：每个节点分配到最大 embedding 维度对应的角色；
- `Douban_node_roles_one_hot.csv`：角色分配的 one-hot 矩阵。

如果要处理其他数据集，需要修改 `process_embedding.py` 顶部三个路径：

```python
embedding_file = r'output\embeddings\YourDataset_embedding.csv'
hard_assignment_file = r'output\embeddings\YourDataset_node_roles.csv'
one_hot_output_file = r'output\embeddings\YourDataset_node_roles_one_hot.csv'
```

## 一个完整例子

以 `DIMACS10` 为例：

```powershell
python src\main.py `
  --input input\DIMACS10.csv `
  --recursive-features-output output\features\DIMACS10_features.csv `
  --embedding-output output\embeddings\DIMACS10_embedding.csv `
  --dimensions 1000 `
  --epochs 5 `
  --batch-size 320
```

然后把 `process_embedding.py` 顶部路径改为 `DIMACS10` 对应文件，再运行：

```powershell
python process_embedding.py
```

最终用于后续网络拆解实验的常用文件是：

```text
output/embeddings/DIMACS10_embedding.csv
output/embeddings/DIMACS10_node_roles.csv
output/embeddings/DIMACS10_node_roles_one_hot.csv
output/embeddings/DIMACS10_role_features.csv
```



## 原始算法背景

ReFeX 和 RolX 分别来自以下工作：

- Keith Henderson et al. *It's who you know: graph mining using recursive structural features*. KDD 2011.
- Keith Henderson et al. *RolX: Structural Role Extraction & Mining in Large Graphs*. KDD 2012.

本仓库的目标不是重新实现高阶 GNN 主模型，而是为网络拆解实验准备结构角色相关输入特征。

## License

本项目保留原仓库的 GNU License，详见 `LICENSE`。
