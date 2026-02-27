# qcgem_gpu_pipeline

基于 PySCF-GPU 的高通量量子化学特征异步提取流水线

## 项目背景

复现并加速生成 qcGEM 模型所需的分子图节点/边特征，构建高吞吐量的量子化学计算流水线。

- **计算基准**: DFT (B3LYP/def2-SVP)
- **架构**: 异步流水线（生产者-消费者模式）

## 架构设计

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Producer  │────▶│    Queue    │────▶│  Consumer   │
│  (GPU: DFT) │     │  (Redis/    │     │ (CPU: 特征  │
│             │     │   MP Queue) │     │   提取)     │
└─────────────┘     └─────────────┘     └─────────────┘
```

## 安装

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

## 使用方法

### 1. 启动流水线

```bash
python main.py --input molecules.smi --output features.h5
```

### 2. 运行测试

```bash
python test_pipeline.py
```

## 目录结构

```
qcgem_gpu_pipeline/
├── producer/          # GPU DFT 计算模块
├── consumer/          # CPU 特征提取模块
├── queue/             # 异步队列模块
├── utils/             # 工具函数
├── tests/             # 测试脚本
├── logs/              # 日志文件
└── temp/              # 临时文件
```

## 配置

在 `config.yaml` 中修改以下参数：

- `dft.functional`: DFT 泛函（默认 B3LYP）
- `dft.basis`: 基组（默认 def2-SVP）
- `queue.type`: 队列类型（redis/multiprocessing）
- `output.format`: 输出格式（json/hdf5）

## 许可证

MIT
