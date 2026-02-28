# FastORCA GPU 加速指南

本文档详细介绍如何在 FastORCA 中充分利用 NVIDIA GPU 进行高通量量子化学计算。

## 硬件要求

### 推荐配置
- **GPU**: NVIDIA A100/A800/H100 (80GB 显存)
- **CPU**: 64+ 核 (用于并行特征提取)
- **内存**: 256GB+ RAM
- **存储**: NVMe SSD (用于临时文件)

### 最低配置
- **GPU**: NVIDIA V100 (16GB) 或 RTX 3090 (24GB)
- **CPU**: 16 核
- **内存**: 64GB RAM

## GPU 加速原理

### 计算流程
```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Producer 1 │────▶│             │     │  Consumer 1 │
│  (GPU 0)    │     │   Task      │────▶│  (CPU 1)    │
├─────────────┤     │   Queue     │     ├─────────────┤
│  Producer 2 │────▶│   (MP)      │     │  Consumer 2 │
│  (GPU 0)    │     │             │────▶│  (CPU 2)    │
├─────────────┤     └─────────────┘     ├─────────────┤
│  Producer N │                         │  Consumer N │
│  (GPU 0)    │                         │  (CPU N)    │
└─────────────┘                         └─────────────┘

多进程共享 GPU，通过 CUDA 流并行执行
```

### 显存占用估算

| 分子大小 | sto-3g | 3-21g | def2-svp |
|----------|--------|-------|----------|
| 10 原子  | 0.5GB  | 1GB   | 2GB      |
| 50 原子  | 2GB    | 4GB   | 8GB      |
| 100 原子 | 4GB    | 8GB   | 16GB     |

**A800 80GB 显存可并行**: ~40 个 3-21g 分子

## 基组选择策略

### GPU 友好基组（强烈推荐）

#### 1. STO-3G（最小基组）
```bash
python main.py --input mol.smi --output out/ --basis sto-3g --n-producers 20
```
- **优点**: GPU 兼容性最好，速度最快
- **缺点**: 精度较低，仅含 s,p 轨道
- **适用**: 大规模筛选 (>10,000 分子)

#### 2. 3-21G（分裂价层，推荐）
```bash
python main.py --input mol.smi --output out/ --basis 3-21g --n-producers 10
```
- **优点**: 精度显著提升，仍支持 GPU
- **轨道类型**: s,p 价层分裂
- **适用**: **量产推荐**，平衡精度与速度

#### 3. 6-31G（Pople 基组）
```bash
python main.py --input mol.smi --output out/ --basis 6-31g --n-producers 10
```
- **优点**: 化学精度良好
- **注意**: 不含极化函数（用 6-31g* 会触发 CPU 回退）

### 含 d 极化函数的基组（CPU 回退）

以下基组会触发 GPU → CPU 自动回退：

| 基组 | 极化函数 | GPU 支持 | 回退后性能 |
|------|----------|----------|-----------|
| 6-31g* | d on heavy | ❌ | CPU + density_fit |
| 6-31+g* | d + diffuse | ❌ | CPU + density_fit |
| def2-svp | d on all non-H | ❌ | CPU + density_fit |
| def2-tzvp | d,f | ❌ | CPU + density_fit |
| cc-pvdz | d | ❌ | CPU + density_fit |

### 基组选择决策树

```
需要 GPU 加速？
├── 是 → 使用 3-21g（推荐）或 sto-3g
│        ├── 分子 > 50 原子 → sto-3g
│        └── 分子 < 50 原子 → 3-21g
└── 否 → 使用 def2-svp 或 6-31g*
         ├── 需要最高精度 → def2-tzvp
         └── 平衡精度速度 → def2-svp
```

## 多生产者配置

### A800 优化配置

#### 场景 1: 超大规模筛选（>10,000 小分子）
```bash
# sto-3g，最大并行
python main.py \
    --input large_dataset.smi \
    --output output/ \
    --basis sto-3g \
    --n-producers 40 \
    --n-consumers 16 \
    --functional B3LYP
```
- **显存**: ~40GB (40 × 1GB)
- **吞吐**: ~20-30 分子/秒

#### 场景 2: 标准量产（1,000-10,000 分子）
```bash
# 3-21g，中等并行
python main.py \
    --input production.smi \
    --output output/ \
    --basis 3-21g \
    --n-producers 20 \
    --n-consumers 8 \
    --functional B3LYP
```
- **显存**: ~40GB (20 × 2GB)
- **吞吐**: ~5-10 分子/秒

#### 场景 3: 高精度计算（<1,000 分子）
```bash
# def2-svp，CPU 模式为主
python main.py \
    --input high_quality.smi \
    --output output/ \
    --basis def2-svp \
    --n-producers 5 \
    --n-consumers 32 \
    --functional B3LYP
```
- **模式**: GPU 尝试失败 → CPU + density_fit
- **吞吐**: ~0.5-1 分子/秒

### 生产者数量调优

```python
# 经验公式
n_producers = min(40, max(1, n_molecules // 10))

# 显存限制
n_producers = gpu_memory_gb // memory_per_molecule_gb

# 推荐设置
if basis == 'sto-3g':
    n_producers = 40  # 80GB / 2GB
elif basis == '3-21g':
    n_producers = 20  # 80GB / 4GB
else:
    n_producers = 1   # CPU fallback
```

## D3BJ 色散校正

### 启用方法

```bash
# D3BJ 自动启用（如已安装 pyscf-dftd3）
python main.py --input mol.smi --output out/ --basis def2-svp
```

### 验证色散校正

```python
from producer.dft_calculator import DFTCalculator

calc = DFTCalculator(basis="def2-svp")
mol = calc.from_smiles("c1ccccc1")
mf = calc.run_sp("benzene", mol)

# 检查色散校正
print(f"Dispersion: {mf.disp}")  # 输出: d3bj
print(f"Energy with D3BJ: {mf.e_tot:.6f} Hartree")
```

### 色散校正能量贡献（典型值）

| 分子类型 | D3BJ 贡献 (kcal/mol) |
|----------|---------------------|
| 小分子 (CH4) | -0.1 |
| 芳香族 (苯) | -2.5 |
| 大分子 (C60) | -50.0 |

**注意**: D3BJ 对非共价相互作用（氢键、π-π 堆积）至关重要。

## 监控与调试

### 实时监控 GPU

```bash
# 每秒刷新
watch -n 1 nvidia-smi

# 或使用 gpustat
pip install gpustat
gpustat -i 1
```

### 查看计算日志

```bash
# 过滤关键信息
python main.py --input mol.smi --output out/ --basis 3-21g 2>&1 | tee run.log | grep -E "(GPU|completed|Fallback)"

# 预期输出
# [INFO] Trying GPU acceleration...
# [INFO] GPU DFT completed in 1.50s
# [INFO] D3BJ dispersion correction enabled
```

### 识别 GPU 回退

```
[WARN] GPU calculation failed: MD_build_j kernel for (dp|dp) failed
[INFO] Falling back to CPU...
[INFO] Using density fitting for CPU calculation
[INFO] CPU DFT completed in 10.20s
```

出现此消息说明基组不支持 GPU，已自动切换到 CPU。

## 性能基准

### 单分子计算时间（A800）

| 基组 | GPU 时间 | CPU 时间 | 加速比 |
|------|----------|----------|--------|
| sto-3g | 0.5s | 2.0s | 4x |
| 3-21g | 1.5s | 4.0s | 2.7x |
| 6-31g | 2.0s | 5.0s | 2.5x |
| def2-svp | N/A | 10.0s | 1x (CPU) |

### 吞吐量测试（1000 分子）

```bash
# 生成测试数据
python -c "print('\\n'.join(['C']*1000))" > test_1000.smi

# 测试 sto-3g
time python main.py --input test_1000.smi --output out/ --basis sto-3g --n-producers 40
# 预期: ~30 秒 (33 mol/s)

# 测试 3-21g
time python main.py --input test_1000.smi --output out/ --basis 3-21g --n-producers 20
# 预期: ~2 分钟 (8 mol/s)
```

## 高级功能

### IAO 电荷提取

```python
from consumer.feature_extractor import FeatureExtractor

extractor = FeatureExtractor()
features = extractor.extract_all_features("mol.pkl", "mol_001")

iao_charges = features["features"]["charge_iao"]
cm5_charges = features["features"]["charge_cm5"]

print(f"IAO charges: {iao_charges}")   # NPA 平替
print(f"CM5 charges: {cm5_charges}")   # Hirshfeld 修正
```

### 自定义 DFT 设置

```python
from producer.dft_calculator import DFTCalculator

calc = DFTCalculator(
    functional="B3LYP",
    basis="3-21g",
    verbose=5,           # 输出详细程度
    max_memory=8000,     # 最大内存 (MB)
    scf_conv_tol=1e-9    # SCF 收敛阈值
)
```

## 故障排除

### 问题 1: CUDA 初始化失败

```
CUDA Error: initialization error
```
**解决**: 
- 确保使用 `spawn` 多进程启动方式（已自动配置）
- 检查 `CUDA_VISIBLE_DEVICES` 环境变量

### 问题 2: 显存溢出

```
cupy.cuda.memory.OutOfMemoryError
```
**解决**:
- 减少 `--n-producers` 数量
- 使用更小基组（sto-3g 替代 3-21g）
- 分批处理大数据集

### 问题 3: GPU 计算卡住

```
[无输出，GPU 利用率 100%]
```
**解决**:
- 检查是否有 `MD_build_j` 错误等待回退
- 设置任务超时（当前未实现，建议外部监控）

## 最佳实践

1. **始终使用 3-21g 进行量产**，除非特别需要 d 极化函数
2. **监控 GPU 利用率**，确保 >80% 才算充分利用
3. **合理设置生产者数量**，过多会导致显存溢出，过少会浪费 GPU
4. **保留临时文件**用于调试，计算完成后自动清理
5. **定期运行测试套件**确保环境正常

## 联系与支持

- GitHub Issues: [项目仓库]
- PySCF 文档: https://pyscf.org
- gpu4pyscf 文档: https://github.com/pyscf/gpu4pyscf
