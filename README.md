# FastORCA - 高通量量子化学特征提取流水线

FastORCA 是一个基于 PySCF-GPU 的高性能量子化学计算框架，专为**大规模分子筛选**和**机器学习数据集构建**设计。与传统量子化学软件"集中力量算大分子"不同，FastORCA 采用**"饱和式人海战术"**，通过 GPU 加速 + 高并发架构，实现百万级分子库的秒级处理。

## 为什么选择 FastORCA？

### 🚀 速度优势：吞吐量提升 10 倍+

| 对比项 | 传统 ORCA (MPI) | FastORCA (高并发) |
|--------|----------------|-------------------|
| **单分子计算** | 128 核并行算 1 个苯环：~5-10 秒 | 单核算 1 个苯环：~21 秒 |
| **并行策略** | 排队串行，CPU 空转等待 | **同时并行 60 个分子** |
| **等效耗时** | - | **~0.35 秒/个** (21s ÷ 60) |
| **100 万分子预估** | 30-50 天 | **4-5 天** |

**核心差异**：传统 ORCA 受限于 Amdahl 定律，MPI 通信开销导致 128 核算小分子效率极低；FastORCA 利用 GPU + 多进程并发，让 128 核真正满负荷运转。

### 💾 IO 优势：内存直通 vs 硬盘杀手

**传统 ORCA 流程**：
```
写 input.inp → 启动进程 → 读写 GB 级临时文件 (.rwf, .int) → 写 output.out → 解析文本
```
- 100 万分子产生 **数百 TB 垃圾读写**
- 硬盘 IO 成为瓶颈，CPU 空等硬盘转圈

**FastORCA 内存直通**：
```
GPU 计算 → RAM 中传递波函数 → CPU 即时提取特征
```
- **零中间文件**：波函数通过 `multiprocessing.Queue` 内存直达
- **CPU 永远满载**：无硬盘等待，算力 100% 用于科学计算

### ⚡ 特征提取：一步到位 vs 繁琐后处理

**传统流程**：
```
DFT 计算 → 生成 .gbw → 启动 Multiwfn → 手动提取电荷/键级 → 拼表整理
```
- 后处理比 DFT 本身还慢
- 多软件切换，容易出错

**FastORCA 原生集成**：
- **IAO 电荷**：NPA 的完美平替，物理意义清晰
- **CM5 电荷**：Hirshfeld 工业级修正
- **Mayer/Wiberg 键级**：毫秒级计算完成
- **DFT 结束瞬间，特征已就绪**

## 核心特性

- **GPU 加速**: 利用 NVIDIA GPU 加速 DFT 计算（通过 gpu4pyscf）
- **多进程并行**: 支持多 Producer + 多 Consumer 架构，充分利用多核 CPU 和 GPU
- **高精度 DFT**: 支持 D3BJ 色散校正、IAO/CM5 高级电荷分析
- **智能回退**: GPU 失败时自动回退到 CPU，确保计算连续性
- **灵活配置**: 支持多种基组、泛函和自定义参数

## 技术架构对比

### 传统量子化学软件（ORCA/Gaussian）
```
MPI 并行单任务架构
┌─────────────────────────────────────┐
│  128 CPU Cores → 1 Molecule         │
│  [||||||||||||||||]  重度并行        │
│                                     │
│  问题：Amdahl 定律限制，通信开销大    │
│        小分子无法有效并行            │
└─────────────────────────────────────┘
```

### FastORCA（高并发多任务架构）
```
GPU + CPU 混合并发架构
┌─────────────────────────────────────┐
│  GPU → Producer 1 → Molecule 1      │
│  GPU → Producer 2 → Molecule 2      │  60 并发
│       ...                           │
│  CPU → Consumer N → Feature N       │
│                                     │
│  优势：零通信开销，线性扩展          │
│        每个核心独立计算              │
└─────────────────────────────────────┘
```

**设计哲学差异**：
- **传统软件**：优化"单个分子的计算速度"
- **FastORCA**：优化"单位时间内处理的分子总数"

## 适用场景

### ✅ 推荐使用 FastORCA

| 场景 | 分子数量 | 说明 |
|------|----------|------|
| **机器学习数据集构建** | 10k - 1M | 训练神经网络需要大量标注数据 |
| **虚拟筛选** | 100k+ | 药物发现、材料筛选的先导化合物评估 |
| **构效关系研究** | 10k - 100k | QSAR/QSPR 模型的特征工程 |
| **分子生成验证** | 1k - 10k | 生成式 AI (VAE, GAN, Diffusion) 的分子合法性验证 |
| **基准测试集构建** | 1k - 10k | 为标准数据集 (QM9, ANI-1x 等) 添加高级特征 |

### ❌ 不推荐使用 FastORCA

| 场景 | 推荐工具 | 原因 |
|------|----------|------|
| **单个大分子 (>200 原子)** | ORCA/Gaussian | FastORCA 针对小分子优化，大分子请用专业软件的 DFT + 线性标度方法 |
| **高精度单点能** | ORCA/MOLPRO | 需要 CCSD(T) 等后 HF 方法 |
| **过渡态搜索** | ORCA/Gaussian | 需要复杂的几何优化和频率分析 |
| **光谱模拟** | ORCA | 需要专业的激发态计算模块 |

## 安装指南

### 系统要求

#### 必需组件
- **Python**: 3.10 或更高版本
- **CUDA**: 11.8 或 12.x（推荐 12.1）
- **NVIDIA GPU**: 计算能力 7.0+（V100/A100/A800/H100）
- **内存**: 至少 32GB RAM（推荐 128GB+）

#### 可选但推荐
- **gfortran**: 用于编译 gpu4pyscf
- **cmake**: 3.18+ 版本

### 详细安装步骤

#### 步骤 1: 创建 Conda 环境

```bash
# 创建 Python 3.10 环境
conda create -n fastorca python=3.10 -y
conda activate fastorca

# 验证 Python 版本
python --version  # 应显示 Python 3.10.x
```

#### 步骤 2: 安装 CUDA 版 PyTorch

根据您的 CUDA 版本选择对应的安装命令：

**CUDA 12.1（推荐）:**
```bash
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
```

**CUDA 11.8:**
```bash
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu118
```

验证 PyTorch 和 CUDA 可用性：
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

#### 步骤 3: 安装基础依赖

使用清华源加速下载（中国用户推荐）：

```bash
# 配置 pip 使用清华源
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn

# 安装核心依赖
pip install pyscf>=2.4.0 pyscf-dftd3 rdkit loguru numpy scipy h5py

# 安装工具依赖
pip install tqdm pyyaml click pandas typing-extensions

# 安装测试工具（可选）
pip install pytest pytest-asyncio black mypy
```

或者使用 requirements.txt 安装：
```bash
pip install -r requirements.txt
```

#### 步骤 4: 安装编译工具（用于 gpu4pyscf）

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install -y gfortran cmake build-essential
```

**CentOS/RHEL:**
```bash
sudo yum install -y gcc-gfortran cmake make gcc-c++
```

验证安装：
```bash
gfortran --version  # 应显示 9.0+
cmake --version     # 应显示 3.18+
```

#### 步骤 5: 从源码安装 gpu4pyscf

**注意**: gpu4pyscf 必须从源码编译安装，无法通过 pip 直接安装。

```bash
# 克隆仓库
cd /tmp
git clone --depth 1 https://github.com/pyscf/gpu4pyscf.git
cd gpu4pyscf

# 编译并安装（需要 5-10 分钟）
pip install .

# 验证安装
python -c "from gpu4pyscf.dft import RKS; print('✅ gpu4pyscf 安装成功')"
```

**常见问题:**

1. **编译错误**: "CMake Error: No CMAKE_Fortran_COMPILER"
   - 解决: 确保已安装 gfortran (`sudo apt-get install gfortran`)

2. **CUDA 错误**: "CUDA not found"
   - 解决: 确保 `nvcc` 在 PATH 中 (`export PATH=/usr/local/cuda/bin:$PATH`)

3. **内存不足**: 编译时卡住
   - 解决: 限制并行编译任务数 `MAKEFLAGS=-j2 pip install .`

#### 步骤 6: 克隆并配置 FastORCA

```bash
# 克隆仓库
cd /home/sulixian
git clone <repository-url> FastORCA
cd FastORCA

# 验证安装
python -c "
from pyscf import gto, dft
from gpu4pyscf.dft import RKS
from rdkit import Chem
from loguru import logger
print('✅ 所有核心依赖安装成功！')
"

# 运行测试
python test_fastorca.py
```

### 配置清华源（中国大陆用户）

```bash
# pip 清华源
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# conda 清华源
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --set show_channel_urls yes
```

### Docker 安装（推荐用于生产环境）

```dockerfile
# Dockerfile
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3.10 python3-pip gfortran cmake git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install torch --index-url https://download.pytorch.org/whl/cu121
RUN pip install pyscf pyscf-dftd3 rdkit loguru numpy scipy h5py

WORKDIR /tmp
RUN git clone --depth 1 https://github.com/pyscf/gpu4pyscf.git && \
    cd gpu4pyscf && pip install . && cd .. && rm -rf gpu4pyscf

WORKDIR /workspace
COPY . /workspace/FastORCA
WORKDIR /workspace/FastORCA

CMD ["python", "main.py", "--help"]
```

构建和运行：
```bash
docker build -t fastorca .
docker run --gpus all -v $(pwd)/data:/data fastorca \
    python main.py --input /data/mol.smi --output /data/out/ --basis 3-21g
```

### 快速开始

### 2. 基本使用

```bash
# 单生产者模式（适合小批量测试）
python main.py --input molecules.smi --output output/ --basis 3-21g

# 多生产者模式（推荐用于量产）
python main.py --input molecules.smi --output output/ \
    --basis 3-21g \
    --n-producers 10 \
    --n-consumers 4

# 高精度计算（def2-svp + D3BJ，CPU 模式）
python main.py --input molecules.smi --output output/ \
    --basis def2-svp \
    --functional B3LYP \
    --n-producers 5 \
    --n-consumers 8
```

### 3. 输入文件格式

创建 `molecules.smi`，每行一个 SMILES：
```
C
CC
C=C
C#C
CO
c1ccccc1
```

### 4. 输出结果

每个分子生成一个 JSON 文件，包含：
```json
{
  "molecule_id": "mol_000000",
  "features": {
    "charge_iao": [...],        // IAO 电荷（NPA 平替）
    "charge_cm5": [...],        // CM5 电荷（Hirshfeld 修正）
    "hirshfeld_charges": [...], // Hirshfeld 电荷
    "mulliken_charges": [...],  // Mulliken 电荷
    "mayer_bond_orders": [...], // Mayer 键级
    "wiberg_bond_orders": [...],// Wiberg 键级
    "energy": -40.4877,         // 总能量（含 D3BJ）
    "dispersion_correction": true
  }
}
```

## 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input` | 必填 | 输入 SMILES 文件路径 |
| `--output` | `output/` | 输出目录 |
| `--basis` | `def2-SVP` | 基组（推荐 3-21g 用于 GPU） |
| `--functional` | `B3LYP` | DFT 泛函 |
| `--n-producers` | `1` | GPU 生产者进程数 |
| `--n-consumers` | `4` | CPU 消费者进程数 |
| `--feature-format` | `json` | 输出格式（json/hdf5） |

## 基组选择指南

### GPU 加速基组（推荐用于量产）

| 基组 | GPU 支持 | 精度 | 适用场景 |
|------|----------|------|----------|
| `sto-3g` | ✅ 完美 | ⭐⭐ | 快速筛选、测试 |
| `3-21g` | ✅ 完美 | ⭐⭐⭐ | **量产推荐** |
| `6-31g` | ✅ 良好 | ⭐⭐⭐ | 平衡精度与速度 |

### CPU 高精度基组（自动回退）

| 基组 | GPU 支持 | 精度 | 适用场景 |
|------|----------|------|----------|
| `6-31g*` | ❌ 失败 | ⭐⭐⭐⭐ | 需要 d 极化函数 |
| `def2-svp` | ❌ 失败 | ⭐⭐⭐⭐ | 高精度量产 |
| `def2-tzvp` | ❌ 失败 | ⭐⭐⭐⭐⭐ | 最高精度 |

## 性能优化

### A800 GPU 推荐配置

对于 80GB 显存的 A800：

```bash
# 小批量测试（<100 分子）
--n-producers 3 --n-consumers 4

# 中等批量（100-1000 分子）
--n-producers 10 --n-consumers 8

# 大规模批量（>1000 分子）
--n-producers 20 --n-consumers 16
```

### 性能对比（苯环分子，def2-svp）

| 配置 | 总耗时 | 吞吐量 |
|------|--------|--------|
| 1 Producer + 2 Consumers | 24s | 0.4 mol/s |
| 10 Producers + 2 Consumers | 21s | 0.5 mol/s |
| CPU-only (密度拟合) | ~30s | 0.3 mol/s |

## 测试与验证

### 运行测试套件

```bash
# 完整测试（sto-3g + def2-svp）
python test_fastorca.py

# 高精度功能测试
python test_high_precision.py

# GPU 兼容性测试
python test_gpu_no_df.py
```

### 验证 D3BJ 色散校正

```python
from producer.dft_calculator import DFTCalculator

calc = DFTCalculator(functional="B3LYP", basis="def2-svp")
mol = calc.from_smiles("c1ccccc1")
mf = calc.run_sp("benzene", mol)

print(f"总能量: {mf.e_tot:.6f} Hartree")
print(f"D3BJ 校正: {mf.disp if hasattr(mf, 'disp') else '未启用'}")
```

## 故障排除

### 常见问题

**1. CUDA Error in MD_build_j**
```
CUDA Error in MD_build_j: invalid argument
MD_build_j kernel for (dp|dp) failed
```
- **原因**: 基组包含 d 极化函数，gpu4pyscf 不支持
- **解决**: 使用 `3-21g` 基组，或等待自动回退到 CPU

**2. GPU 内存不足**
```
cupy.cuda.memory.OutOfMemoryError
```
- **解决**: 减少 `--n-producers` 数量，或使用更小基组

**3. SCF 不收敛**
```
SCF did not converge!
```
- **解决**: 程序会自动尝试二阶收敛，或增大 `max_cycle`

### 调试模式

```bash
# 查看详细日志
python main.py --input test.smi --output output/ --basis 3-21g 2>&1 | tee run.log

# 检查 GPU 利用率
watch -n 1 nvidia-smi
```

## 项目结构

```
FastORCA/
├── main.py                   # 主入口
├── producer/
│   └── dft_calculator.py     # GPU DFT 计算
├── consumer/
│   └── feature_extractor.py  # 特征提取（IAO/CM5/Mayer/Wiberg）
├── taskqueue/
│   └── task_queue.py         # 多进程队列管理
├── test_fastorca.py          # 完整测试套件
├── test_high_precision.py    # 高精度功能测试
└── README_GPU.md             # GPU 详细指南
```


## 运行此脚本以验证物理精度（IAO/CM5/D3BJ） 

```bash
python test/test_high_precision.py
```

## 引用

如果使用 FastORCA，请引用：

- PySCF: [https://pyscf.org](https://pyscf.org)
- gpu4pyscf: [https://github.com/pyscf/gpu4pyscf](https://github.com/pyscf/gpu4pyscf)
- DFT-D3: [https://github.com/dftd3/simple-dftd3](https://github.com/dftd3/simple-dftd3)

## License

MIT License
