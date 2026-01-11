# TRL GRPO Training Guide

完整的 TRL + vLLM + DeepSpeed 训练指南，适用于 8x A100 80GB 配置。

## 目录
- [快速开始](#快速开始)
- [安装依赖](#安装依赖)
- [架构说明](#架构说明)
- [配置文件](#配置文件)
- [启动训练](#启动训练)
- [监控训练](#监控训练)
- [故障排除](#故障排除)
- [性能优化](#性能优化)

---

## 快速开始

### 一键启动（推荐）

```bash
cd /home/ubuntu/jiex/code/research
bash scripts/launch_trl_training.sh configs/grpo_config.yaml
```

这个脚本会自动：
1. 在 GPU 0-1 上启动 vLLM 服务器（生成加速）
2. 在 GPU 2-7 上启动 DeepSpeed 训练（6-GPU 分布式）
3. 管理日志和进程清理

---

## 安装依赖

### 1. 安装 TRL 和 vLLM

```bash
pip install trl[vllm]
```

### 2. 验证安装

```bash
python -c "import trl; print(f'TRL version: {trl.__version__}')"
python -c "import vllm; print(f'vLLM version: {vllm.__version__}')"
```

### 3. 已有依赖（无需重新安装）

- ✅ PyTorch
- ✅ Transformers
- ✅ Accelerate
- ✅ DeepSpeed
- ✅ Flash Attention (通过 Kernels Hub，无需编译)

---

## 架构说明

### GPU 分配策略

```
8x A100 80GB 分配方案:
├── GPU 0-1: vLLM 生成引擎
│   ├── Tensor Parallel (TP=2)
│   ├── 负责快速生成 n_samples × batch_size 个响应
│   └── 内存利用率: 60%
│
└── GPU 2-7: DeepSpeed 训练引擎
    ├── ZeRO-3 参数分片（6-GPU）
    ├── 梯度累积 + 混合精度（BF16）
    ├── Liger Kernel 优化（20% 加速，60% 内存节省）
    └── Flash Attention 2（无需编译）
```

### 训练流程

```
Episode Loop:
  1. [vLLM on GPU 0-1]
     生成 batch_size × n_samples_per_prompt 个代码样本
     (例: 8 prompts × 12 samples = 96 个样本)
     时间: ~5-10 秒（相比原生 generate() 快 5-10x）

  2. [CPU Multiprocessing]
     并行执行代码，计算奖励
     (8 workers × 10s timeout)
     时间: ~10-20 秒

  3. [DeepSpeed on GPU 2-7]
     计算 GRPO 优势函数
     策略梯度更新
     时间: ~5-10 秒

Total: ~20-40 秒/episode
```

---

## 配置文件

### 主要配置文件

1. **训练配置**: `configs/grpo_config.yaml`
   - 模型选择
   - GRPO 超参数
   - 批次大小、学习率等

2. **Accelerate 配置**: `configs/accelerate_trl_8xa100.yaml`
   - DeepSpeed ZeRO-3 设置
   - 6-GPU 分布式配置
   - 混合精度（BF16）

3. **Accelerate 配置（ZeRO-2）**: `configs/accelerate_trl_8xa100_zero2.yaml`
   - 更快但内存占用更高
   - 适合较小模型（1.5B-3B）

### 关键参数

#### GRPO 算法参数
```yaml
grpo:
  n_samples_per_prompt: 12      # 每个 prompt 生成多少个样本
  init_kl_coef: 0.001           # KL 散度惩罚系数
  cliprange: 0.2                # PPO 裁剪范围
  entropy_coef: 0.01            # 熵正则化
```

#### 批次大小优化
```yaml
training:
  train_batch_size: 8           # 每步的唯一 prompt 数量
  micro_train_batch_size: 2     # 每个 GPU 的前向传播批次

# 有效批次大小计算:
# effective_batch = micro_batch × gradient_accumulation × num_gpus
# effective_batch = 2 × 4 × 6 = 48

# 每个 episode 的总样本数:
# total_samples = train_batch × n_samples_per_prompt
# total_samples = 8 × 12 = 96
```

#### 内存优化参数
```python
# 在 train_grpo_trl_enhanced.py 中自动启用:
use_liger_kernel=True              # 20% 加速，60% 内存节省
attn_implementation="flash_attention_2"  # Flash Attention（无需编译）
gradient_checkpointing=True        # 梯度检查点
bf16=True                          # BF16 混合精度
```

---

## 启动训练

### 方法 1: 使用启动脚本（推荐）

```bash
# 使用默认配置
bash scripts/launch_trl_training.sh

# 使用自定义配置
bash scripts/launch_trl_training.sh configs/grpo_config_test.yaml
```

**优点**:
- ✅ 自动管理 vLLM 服务器进程
- ✅ 等待 vLLM 初始化完成
- ✅ 自动清理进程
- ✅ 日志分离（vLLM + 训练）

---

### 方法 2: 手动启动（高级用户）

#### 步骤 1: 启动 vLLM 服务器

```bash
# Terminal 1: vLLM 服务器 (GPU 0-1)
CUDA_VISIBLE_DEVICES=0,1 trl vllm-serve \
    --model Qwen/Qwen2.5-3B-Instruct \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.6 \
    --max-num-seqs 256 \
    --max-model-len 2048
```

等待直到看到: `Uvicorn running on http://...`

#### 步骤 2: 启动训练

```bash
# Terminal 2: 训练 (GPU 2-7)
cd /home/ubuntu/jiex/code/research

CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 accelerate launch \
    --config_file configs/accelerate_trl_8xa100.yaml \
    training/train_grpo_trl_enhanced.py \
    --config configs/grpo_config.yaml
```

---

## 监控训练

### 1. 实时监控脚本

```bash
# 在另一个 terminal 中
python monitor_training.py --log_dir logs/trl_training_<timestamp>
```

监控指标:
- GPU 利用率、内存、温度
- Episode 进度
- Loss、奖励、优势函数
- 生成和训练时间

### 2. TensorBoard

```bash
tensorboard --logdir outputs/qwen-grpo-trl --port 6006
```

访问: `http://localhost:6006`

### 3. 检查 vLLM 服务器状态

```bash
# 查看 vLLM 日志
tail -f logs/trl_training_<timestamp>/vllm_server.log

# 测试 vLLM API
curl http://localhost:8000/health
```

### 4. 实时查看训练日志

```bash
tail -f logs/trl_training_<timestamp>/training.log
```

---

## 故障排除

### 问题 1: vLLM 服务器启动失败

**症状**: `Error: vLLM server failed to start`

**解决方案**:
```bash
# 检查 vLLM 日志
cat logs/trl_training_<timestamp>/vllm_server.log

# 常见原因:
# 1. GPU 内存不足 -> 减少 gpu_memory_utilization
# 2. 端口被占用 -> 杀死旧进程: pkill -f vllm
# 3. 模型下载失败 -> 检查网络连接
```

### 问题 2: OOM (Out of Memory)

**症状**: `CUDA out of memory`

**解决方案**:

1. **减少批次大小**
```yaml
# configs/grpo_config.yaml
training:
  micro_train_batch_size: 1  # 从 2 降到 1
  train_batch_size: 4        # 从 8 降到 4
```

2. **切换到 ZeRO-3**（如果使用 ZeRO-2）
```bash
# 使用 ZeRO-3 配置
accelerate launch --config_file configs/accelerate_trl_8xa100.yaml ...
```

3. **启用 CPU Offload**
```yaml
# configs/accelerate_trl_8xa100.yaml
deepspeed_config:
  offload_optimizer_device: cpu
  offload_param_device: cpu
```

### 问题 3: 训练速度慢

**症状**: 每个 episode > 60 秒

**诊断**:

```bash
# 检查 vLLM 是否正常工作
# 应该看到 "Using vLLM for generation"
grep -i vllm logs/trl_training_<timestamp>/training.log

# 检查 GPU 利用率
nvidia-smi dmon -s u
```

**解决方案**:

1. **确认 vLLM 正在使用**
   - 检查是否有 "use_vllm=True" 的日志
   - 确认 vLLM 服务器在 GPU 0-1 上运行

2. **增加 vLLM 并行度**
```bash
# 启动 vLLM 时增加 max_num_seqs
trl vllm-serve --max-num-seqs 512 ...
```

3. **减少奖励计算开销**
```yaml
# 如果奖励计算慢，增加并行 worker
# 在 reward_func.py 中调整
max_workers: 16  # 默认是 8
```

### 问题 4: 奖励函数报错

**症状**: `Reward computation failed`

**诊断**:
```bash
# 测试奖励函数
python test_reward_speed.py
```

**解决方案**:
```bash
# 检查 Docker 是否可用
docker --version

# 如果 Docker 不可用，修改 code_executor.py
# 使用 subprocess 而不是 Docker
```

### 问题 5: 分布式训练不同步

**症状**: `RuntimeError: Detected mismatch between collectives`

**解决方案**:
```bash
# 清理所有 torch distributed 进程
pkill -f "torch.distributed"

# 重新启动训练
bash scripts/launch_trl_training.sh
```

---

## 性能优化

### 当前配置 (8x A100 80GB)

| 参数 | 值 | 说明 |
|------|-----|------|
| Model | Qwen2.5-3B-Instruct | 通用模型测试 GRPO |
| Samples per prompt | 12 | GRPO 组大小 |
| Train batch size | 8 | 每步 8 个唯一 prompt |
| Micro batch size | 2 | 每 GPU 2 个样本前向 |
| Total samples/step | 96 | 8 × 12 |
| Training GPUs | 6 (GPU 2-7) | DeepSpeed ZeRO-3 |
| Generation GPUs | 2 (GPU 0-1) | vLLM TP=2 |
| Expected time/episode | 20-40s | 生成 + 奖励 + 训练 |

### 激进配置（更大模型/更快训练）

#### 7B 模型 + 更大批次

```yaml
# configs/grpo_config_aggressive.yaml
model:
  pretrain: "Qwen/Qwen2.5-7B-Instruct"

grpo:
  n_samples_per_prompt: 16

training:
  train_batch_size: 16
  micro_train_batch_size: 2
```

**预期**:
- 更大的有效批次: 16 × 16 = 256 samples/step
- 更好的 GRPO 优势估计（更大的组）
- 训练时间: 40-60s/episode

#### 使用 ZeRO-2 加速

```bash
# ZeRO-2 比 ZeRO-3 快 ~20%，但内存占用更高
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 accelerate launch \
    --config_file configs/accelerate_trl_8xa100_zero2.yaml \
    training/train_grpo_trl_enhanced.py
```

**适用于**: 3B 及以下模型

### 保守配置（调试/测试）

```yaml
# configs/grpo_config_test.yaml
grpo:
  n_samples_per_prompt: 4  # 更快的迭代

training:
  train_batch_size: 2
  micro_train_batch_size: 1
  num_episodes: 10  # 快速测试
```

**用途**: 快速验证代码是否工作

---

## 与原有脚本对比

### 代码行数对比

| 文件 | 代码行数 | 说明 |
|------|---------|------|
| `train_grpo_vllm_deepspeed.py` | ~650 行 | 手动实现 |
| `train_grpo_trl_enhanced.py` | ~250 行 | TRL 实现 |
| **减少** | **~60%** | **代码简化** |

### 功能对比

| 功能 | 手动实现 | TRL 实现 |
|------|---------|---------|
| vLLM 集成 | ✅ 手动管理 | ✅ `use_vllm=True` |
| DeepSpeed | ✅ 手动配置 | ✅ Accelerate 管理 |
| Liger Kernel | ❌ 未使用 | ✅ `use_liger_kernel=True` |
| Flash Attention | ⚠️ 需手动编译 | ✅ Kernels Hub（无需编译） |
| 梯度累积 | ✅ 手动实现 | ✅ 自动处理 |
| 分布式同步 | ✅ 手动 broadcast | ✅ 自动同步 |
| 日志 | ⚠️ 自定义 | ✅ TensorBoard/W&B 集成 |

### 性能预期

| 指标 | 手动实现 | TRL 实现 | 提升 |
|------|---------|---------|-----|
| 代码维护 | 困难 | 简单 | ⭐⭐⭐ |
| 生成速度 | 快 (vLLM) | 快 (vLLM) | 相同 |
| 训练速度 | 基准 | +20% (Liger) | ⭐⭐ |
| 内存占用 | 基准 | -60% (Liger) | ⭐⭐⭐ |
| 易用性 | 复杂 | 简单 | ⭐⭐⭐ |

---

## 下一步

### 1. 运行测试训练

```bash
# 快速测试（10 个 episodes）
bash scripts/launch_trl_training.sh configs/grpo_config_test.yaml
```

### 2. 完整训练

```bash
# 完整训练（50 个 episodes）
bash scripts/launch_trl_training.sh configs/grpo_config.yaml
```

### 3. 评估模型

```bash
# 使用 EvalPlus 评估
python evaluation/run_evalplus_vllm.py \
    --model_path outputs/qwen-grpo-trl/final \
    --dataset humaneval

# 对比多个 checkpoint
python evaluation/compare_checkpoints.py \
    --checkpoint_dir outputs/qwen-grpo-trl
```

### 4. 扩展到更大模型

```bash
# 7B 模型训练
# 修改 configs/grpo_config.yaml:
#   model.pretrain: "Qwen/Qwen2.5-7B-Instruct"

bash scripts/launch_trl_training.sh configs/grpo_config.yaml
```

---

## 参考资源

- [TRL Documentation](https://huggingface.co/docs/trl)
- [TRL Speeding Up Training](https://huggingface.co/docs/trl/speeding_up_training)
- [TRL Distributing Training](https://huggingface.co/docs/trl/distributing_training)
- [vLLM Documentation](https://docs.vllm.ai/)
- [DeepSpeed ZeRO](https://www.deepspeed.ai/tutorials/zero/)
- [Accelerate Documentation](https://huggingface.co/docs/accelerate)

---

## 问题反馈

如果遇到问题:

1. **检查日志**: `logs/trl_training_<timestamp>/`
2. **测试组件**:
   ```bash
   # 测试 vLLM
   python test_vllm_generation.py

   # 测试奖励函数
   python test_reward_speed.py
   ```
3. **查看 GPU 状态**:
   ```bash
   nvidia-smi
   watch -n 1 nvidia-smi
   ```

---

**祝训练顺利！** 🚀
