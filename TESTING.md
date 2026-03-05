# FastORCA 测试指南

本文档说明如何运行 FastORCA 的测试套件。

## 环境准备

### 1. 安装依赖

```bash
# 基础依赖
pip install -r requirements.txt

# GPU 支持（可选但推荐）
# 需要从源码安装 gpu4pyscf
git clone https://github.com/pyscf/gpu4pyscf.git
cd gpu4pyscf && pip install .

# 几何优化支持（推荐）
pip install xtb  # GFN2-xTB 快速优化
pip install geometric  # PySCF 几何优化
```

### 2. 验证安装

```bash
python -c "
from producer.dft_calculator import DFTCalculator, XTB_AVAILABLE, GEOMETRIC_AVAILABLE
from consumer.feature_extractor import FeatureExtractor
print('✅ Imports successful')
print(f'   xtb-python available: {XTB_AVAILABLE}')
print(f'   geometric available: {GEOMETRIC_AVAILABLE}')
"
```

## 运行测试

### 快速测试（推荐）

仅测试基础功能，不运行耗时的 DFT 计算：

```bash
# 运行所有快速测试
python test_pipeline.py --quick

# 或运行特定测试
python test_pipeline.py --feature dft --quick
python test_pipeline.py --feature queue --quick
```

### 完整测试

运行所有测试，包括 DFT 计算（耗时较长）：

```bash
# 运行全部测试
python test_pipeline.py

# 运行特定功能测试
python test_pipeline.py --feature dft        # DFT 计算器
python test_pipeline.py --feature queue      # 任务队列
python test_pipeline.py --feature e2e        # 端到端流水线
python test_pipeline.py --feature geometry   # 几何优化
python test_pipeline.py --feature new        # 新功能（ELF、IAO等）
```

### 测试选项说明

| 选项 | 说明 |
|-----|-----|
| `--quick, -q` | 跳过耗时的 DFT 计算，仅测试基础功能 |
| `--feature` | 选择要测试的功能模块 |

## 测试内容

### 1. DFT 计算器测试 (`test_dft_calculator`)

- 测试 SMILES 解析
- 测试分子创建
- 验证原子数量

**预期输出**:
```
Testing DFT Calculator
Testing SMILES: CCO
  Created molecule with 9 atoms
Testing SMILES: c1ccccc1
  Created molecule with 12 atoms
DFT Calculator test completed
```

### 2. 任务队列测试 (`test_queue`)

- 测试队列创建
- 测试任务放入/取出
- 测试队列大小

**预期输出**:
```
Testing Task Queue
Queue size: 5
Got task: test_0
...
Task Queue test completed
```

### 3. 端到端流水线测试 (`test_pipeline_end_to_end`)

- 从 SMILES 创建分子
- 执行 DFT 计算
- 提取特征
- 保存结果

**注意**: 此测试执行实际的 DFT 计算，每个分子需要 10-30 秒。

### 4. 几何优化测试 (`test_geometry_optimization`) ⭐ NEW

- 测试禁用几何优化
- 测试 xTB 几何优化（如果可用）
- 测试 PySCF 几何优化（如果可用）

**预期输出**:
```
Testing Geometry Optimization
xtb-python available: True
geometric available: True
Testing without geometry optimization...
  Created molecule without opt: 9 atoms
Testing with geometry optimization (xtb)...
  Created molecule with opt: 9 atoms
Geometry Optimization test completed
```

### 5. 新功能测试 (`test_new_features`) ⭐ NEW

测试 qcGEM 关键功能：

- **ELF 特征**: 验证 `elf_at_atoms`, `elf_bond_midpoints`, `elf_mean`
- **IAO 矩阵**: 验证 `iao_fock_matrix`, `iao_charges`
- **全局特征**: 验证 `homo_energy`, `lumo_energy`, `homo_lumo_gap`, `dipole_moment`
- **传统特征**: 验证 `charge_iao`, `charge_cm5`, `mayer_bond_orders`

**预期输出**:
```
Testing New Features (ELF, IAO Matrix, Global)
Running DFT calculation...
  DFT completed: E = -153.456789
Extracting features with new methods...
Checking ELF features...
  ELF at atoms: 9 values
  ELF bond midpoints: 8 values
  ELF mean: 0.75
  ELF features extracted successfully
Checking IAO matrix features...
  IAO Fock matrix shape: (13, 13)
  IAO Fock matrix extracted successfully
  IAO charges: 9 atoms
  IAO charges extracted successfully
Checking global features...
  homo_energy: -0.234
  lumo_energy: 0.123
  homo_lumo_gap: 0.357
  dipole_moment: 1.85
  total_energy: -153.456789
  n_electrons: 26
Checking traditional features...
  IAO charges: 9 atoms
  CM5 charges: 9 atoms
  Mayer bond orders: 9x9 matrix
New features test passed!
```

## 特征输出格式

测试成功后会生成 JSON 文件，包含以下结构：

```json
{
  "molecule_id": "test_new_features",
  "features": {
    "elf": {
      "elf_at_atoms": [0.95, 0.87, ...],
      "elf_bond_midpoints": [0.82, 0.78, ...],
      "bond_pairs": [[0, 1], [0, 2], ...],
      "elf_mean": 0.75,
      "elf_max": 0.99,
      "elf_min": 0.23
    },
    "iao_fock_matrix": [[...], ...],
    "iao_charges": [...],
    "homo_energy": -0.234,
    "lumo_energy": 0.123,
    "homo_lumo_gap": 0.357,
    "dipole_moment": 1.85,
    "dipole_vector": [0.0, 0.0, 1.85],
    "total_energy": -153.456789,
    "n_electrons": 26,
    "scf_converged": true,
    "charge_iao": [...],
    "charge_cm5": [...],
    "mayer_bond_orders": [[...], ...]
  }
}
```

## 故障排除

### 问题 1: ImportError

```
ModuleNotFoundError: No module named 'loguru'
```

**解决**: 安装依赖
```bash
pip install -r requirements.txt
```

### 问题 2: xtb 不可用（性能灾难）⚠️ CRITICAL

**现象**:
```
xtb-python NOT AVAILABLE!
Geometry optimization will fall back to PySCF + geometric
This will be 100-1000x SLOWER!

Using PySCF + geometric for geometry optimization
Step 0 ... Step 4 ... Converged! (85.54s)  ← 太慢了！
```

**影响**:
- 苯环几何优化从 0.5 秒变成 85 秒
- 100 万分子需要 ~3 年（vs 3 天）

**解决**: 必须安装 xTB
```bash
# 推荐：conda 安装（最稳定）
conda install -c conda-forge xtb

# 或者 pip 安装
pip install xtb

# 验证安装
python -c "from xtb.interface import Calculator; print('✅ xTB installed')"
```

**注意**: 如果 xTB 安装失败，程序会自动回退到 PySCF geometric，但速度极慢。

### 问题 3: GPU 崩溃（基组不兼容）❌ CRITICAL

**现象**:
```
⚠️ 基组 def2-SVP 包含 d/f 极化函数，可能触发 GPU 回退到 CPU
⚠️ GPU calculation will likely fail and fall back to CPU

❌ GPU calculation failed!
Error: CUDA Error in MD_build_j: invalid argument
MD_build_j kernel for (dd|ds) failed

⚠️ Falling back to CPU mode
⚠️ This is likely due to:
    1. Basis set contains d/f functions (e.g., def2-SVP)
    2. Out of GPU memory
```

**诊断**:
- `def2-SVP`, `def2-TZVP`, `cc-pVDZ` 等基组包含 d 轨道
- 当前 gpu4pyscf 版本对含 d 函数的积分内核支持不完善

**解决**: 使用 GPU 友好基组
```bash
# GPU 完美支持（无 d 函数）
python main.py --basis sto-3g    # 最快，精度较低
python main.py --basis 3-21g     # 推荐：速度和精度的平衡
python main.py --basis 6-31g     # 较好精度，无 d 函数

# GPU 不支持（含 d/f 函数）- 将自动回退到 CPU
python main.py --basis def2-SVP  # ❌ 会触发 GPU 崩溃
python main.py --basis 6-31G*    # ❌ 带星号表示含 d 函数
```

**验证 GPU 是否正常工作**:
```bash
# 使用 3-21g 测试 GPU
python main.py --input test_molecules.smi --output output/ --basis 3-21g

# 如果看到 "GPU DFT completed in X.XXs"，说明 GPU 正常工作
# 如果看到 "Falling back to CPU"，说明有兼容性问题
```

### 问题 4: DFT 计算失败

```
DFT calculation failed: SCF did not converge
```

**可能原因**:
- 分子结构不合理
- 基组/泛函选择不当
- 内存不足

**解决**:
- 启用几何优化: `--geometry-optimization`
- 更换小基组: `--basis 3-21g`
- 增加内存: 修改 `max_memory=16000` in config.yaml

### 问题 5: 键名错误 (Bug Fix)

**现象**:
```
KeyError: 'pkl_path'
```

**原因**: 旧代码中 dft_calculator 返回 `"pkl_file"`，但测试代码使用 `"pkl_path"`

**解决**: 已修复！dft_calculator 现在同时返回两个键名以保证兼容性。

## 性能基准测试

运行吞吐量测试以评估性能：

```bash
# 生成测试分子文件
python test_pipeline.py --feature all

# 运行完整流水线（100 个分子）
python main.py \
    --input test_molecules.smi \
    --output output_test/ \
    --functional B3LYP \
    --basis 3-21g \
    --n-producers 1 \
    --n-consumers 2 \
    --geometry-optimization \
    --geo-opt-method xtb
```

## 持续集成

建议在 CI 中运行快速测试：

```yaml
# .github/workflows/test.yml 示例
test:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v2
    - name: Install dependencies
      run: pip install -r requirements.txt
    - name: Run quick tests
      run: python test_pipeline.py --quick
```

完整测试（含 DFT 计算）建议在本地或有 GPU 的服务器上运行。
