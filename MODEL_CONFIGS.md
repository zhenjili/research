# 模型配置快速参考

针对不同大小的 Qwen 模型，优化的配置参数。

## 📊 模型对比

| 模型 | 参数量 | 模型大小 | 推荐配置 | 用途 |
|------|--------|---------|---------|------|
| **Qwen2.5-0.5B-Instruct** | 0.5B | ~1GB | `grpo_config_0.5b.yaml` | 🚀 快速迭代测试 |
| **Qwen2.5-1.5B-Instruct** | 1.5B | ~3GB | `grpo_config.yaml` (修改模型) | ⚖️ 平衡性能/速度 |
| **Qwen2.5-3B-Instruct** | 3B | ~6GB | 默认原配置 | 🎯 生产级训练 |
| **Qwen2.5-7B-Instruct** | 7B | ~14GB | `grpo_config_7b.yaml` | 💪 最佳性能 |

---

## 🚀 0.5B 模型 - 最快迭代

### 配置文件
```bash
configs/grpo_config_0.5b.yaml
```

### 关键参数
```yaml
model:
  pretrain: "Qwen/Qwen2.5-0.5B-Instruct"

grpo:
  n_samples_per_prompt: 16  # 更多样本，更好的优势估计

training:
  train_batch_size: 16           # 大批次
  micro_train_batch_size: 4      # 大微批次
  gradient_checkpointing: false  # 可以禁用，加速
  num_episodes: 100

distributed:
  zero_stage: 2  # ZeRO-2 足够
  vllm_gpu_memory_utilization: 0.4
```

### 性能预期
- **每个 episode**: 15-30 秒
  - 生成: 2-5 秒 (256 samples with vLLM)
  - 奖励: 10-20 秒
  - 训练: 3-5 秒
- **总训练时间**: 100 episodes ≈ 25-50 分钟
- **内存占用**:
  - vLLM (GPU 0-1): ~10-15GB per GPU
  - 训练 (GPU 2-7): ~10-15GB per GPU

### 优势
✅ **最快的迭代速度** (2x faster than 3B)
✅ **更大的有效批次** (256 vs 96 samples)
✅ **更多的训练 episodes** (100 vs 50)
✅ **内存充足** (可以禁用 gradient checkpointing)

### 劣势
⚠️ **模型能力较弱** (代码质量可能不如 3B/7B)
⚠️ **初始 pass rate 较低** (需要更多 episodes 才能收敛)

### 使用场景
- 🔬 **算法验证**: 快速测试 GRPO 是否工作
- 🐛 **调试代码**: 验证训练流程无误
- 📊 **超参数搜索**: 快速尝试不同的学习率、批次大小等
- 🎓 **学习实验**: 理解 GRPO 训练过程

### 启动命令
```bash
# 使用优化配置
bash scripts/launch_trl_training.sh configs/grpo_config_0.5b.yaml

# 或手动指定
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 accelerate launch \
    --config_file configs/accelerate_trl_8xa100_zero2.yaml \
    training/train_grpo_trl_enhanced.py \
    --config configs/grpo_config_0.5b.yaml
```

---

## ⚖️ 1.5B 模型 - 平衡选择

### 修改默认配置
```yaml
# configs/grpo_config.yaml
model:
  pretrain: "Qwen/Qwen2.5-1.5B-Instruct"

grpo:
  n_samples_per_prompt: 12

training:
  train_batch_size: 12
  micro_train_batch_size: 3
  num_episodes: 75

distributed:
  zero_stage: 2
```

### 性能预期
- **每个 episode**: 25-40 秒
- **总训练时间**: 75 episodes ≈ 30-50 分钟
- **内存占用**: ~15-20GB per GPU

### 优势
✅ 比 0.5B 模型能力更强
✅ 比 3B 模型更快
✅ 内存占用适中

### 使用场景
- 生产前的原型验证
- 中等规模的实验
- 资源受限的环境

---

## 🎯 3B 模型 - 默认配置

### 配置文件
当前 `configs/grpo_config.yaml` 已更新为 0.5B，如需 3B：

```yaml
# 修改 configs/grpo_config.yaml
model:
  pretrain: "Qwen/Qwen2.5-3B-Instruct"

grpo:
  n_samples_per_prompt: 12

training:
  train_batch_size: 8
  micro_train_batch_size: 2
  gradient_checkpointing: true
  num_episodes: 50

distributed:
  zero_stage: 3
  vllm_gpu_memory_utilization: 0.6
```

### 性能预期
- **每个 episode**: 30-50 秒
- **总训练时间**: 50 episodes ≈ 25-40 分钟
- **内存占用**: ~25-35GB per GPU

### 优势
✅ **强大的代码生成能力**
✅ **较好的初始 pass rate**
✅ **稳定的训练过程**

### 使用场景
- 生产级训练
- 发布模型
- Benchmark 对比

---

## 💪 7B 模型 - 最佳性能

### 创建配置文件
```yaml
# configs/grpo_config_7b.yaml
model:
  pretrain: "Qwen/Qwen2.5-7B-Instruct"

grpo:
  n_samples_per_prompt: 16

training:
  train_batch_size: 16
  micro_train_batch_size: 2
  gradient_checkpointing: true
  num_episodes: 50
  actor_learning_rate: 2.0e-6  # 大模型用更小的学习率

distributed:
  zero_stage: 3
  vllm_gpu_memory_utilization: 0.7
```

### 性能预期
- **每个 episode**: 50-80 秒
- **总训练时间**: 50 episodes ≈ 40-65 分钟
- **内存占用**: ~40-60GB per GPU

### 优势
✅ **最强的代码生成能力**
✅ **最高的 pass rate**
✅ **更好的泛化能力**

### 劣势
⚠️ **训练速度最慢**
⚠️ **内存占用最大**
⚠️ **需要 ZeRO-3 + Gradient Checkpointing**

### 使用场景
- 最终模型训练
- 追求最佳性能
- 充足的计算资源

---

## 🔄 如何切换模型

### 方法 1: 修改配置文件（推荐）

```bash
# 1. 编辑配置文件
vim configs/grpo_config.yaml

# 2. 修改模型名称
model:
  pretrain: "Qwen/Qwen2.5-0.5B-Instruct"  # 或 1.5B, 3B, 7B

# 3. 调整对应的批次大小等参数（见上面的推荐配置）

# 4. 启动训练
bash scripts/launch_trl_training.sh configs/grpo_config.yaml
```

### 方法 2: 使用专门的配置文件

```bash
# 使用 0.5B 优化配置
bash scripts/launch_trl_training.sh configs/grpo_config_0.5b.yaml

# 使用 3B 配置（需要创建）
bash scripts/launch_trl_training.sh configs/grpo_config_3b.yaml

# 使用 7B 配置（需要创建）
bash scripts/launch_trl_training.sh configs/grpo_config_7b.yaml
```

### 方法 3: 同时修改启动脚本

```bash
# 编辑 scripts/launch_trl_training.sh
vim scripts/launch_trl_training.sh

# 修改 MODEL_NAME
MODEL_NAME="Qwen/Qwen2.5-0.5B-Instruct"  # 当前已设置
```

---

## 📈 批次大小计算

### 有效批次大小公式
```
effective_batch = train_batch_size × n_samples_per_prompt
```

### 各模型的推荐配置

| 模型 | train_batch | n_samples | effective_batch | 每 episode 总样本 |
|------|------------|-----------|-----------------|------------------|
| 0.5B | 16 | 16 | **256** | 256 |
| 1.5B | 12 | 12 | **144** | 144 |
| 3B   | 8  | 12 | **96**  | 96  |
| 7B   | 16 | 16 | **256** | 256 |

### GPU 内存分配
```
micro_batch_size × num_gpus × gradient_accumulation_steps = train_batch_size
```

示例（0.5B 模型）:
```
4 (micro) × 6 (GPUs) × 1 (no accumulation) = 24 ≠ 16

实际上：
- train_batch_size = 16 个 prompts
- 每个 prompt 生成 n_samples = 16
- 总共 16 × 16 = 256 个样本需要训练
- micro_batch_size = 4，每次前向传播 4 个样本
- 需要 256 / 4 / 6 = 10.67 ≈ 11 次累积
```

TRL 会自动计算 gradient_accumulation_steps。

---

## 🎯 推荐训练策略

### 阶段 1: 快速验证（1-2 小时）
```bash
# 使用 0.5B 模型 + 测试配置
bash scripts/launch_trl_training.sh configs/grpo_config_test.yaml

# 检查:
# - vLLM 是否正常工作
# - 奖励函数是否计算正确
# - 训练是否收敛
# - 没有 OOM 错误
```

### 阶段 2: 完整训练（8-12 小时）
```bash
# 使用 0.5B 模型 + 完整配置
bash scripts/launch_trl_training.sh configs/grpo_config_0.5b.yaml

# 评估 pass rate 是否有提升
python evaluation/run_evalplus_vllm.py \
    --model_path outputs/qwen-0.5b-grpo-trl/final
```

### 阶段 3: 生产训练（24-48 小时）
```bash
# 使用 3B 或 7B 模型
bash scripts/launch_trl_training.sh configs/grpo_config.yaml  # 改为 3B
# 或
bash scripts/launch_trl_training.sh configs/grpo_config_7b.yaml
```

---

## 💡 调试技巧

### 如果遇到 OOM
1. **减小 micro_batch_size**: 4 → 2 → 1
2. **减小 train_batch_size**: 16 → 8 → 4
3. **启用 gradient_checkpointing**: false → true
4. **切换到 ZeRO-3**: zero_stage: 2 → 3
5. **减小模型**: 3B → 1.5B → 0.5B

### 如果训练太慢
1. **增大 micro_batch_size**: 2 → 4
2. **禁用 gradient_checkpointing**: true → false（如果内存充足）
3. **切换到 ZeRO-2**: zero_stage: 3 → 2
4. **使用更小的模型**: 3B → 0.5B

### 如果 pass rate 不提升
1. **增加 n_samples_per_prompt**: 12 → 16 → 24
2. **增加 train_batch_size**: 8 → 16
3. **调整学习率**: 3e-6 → 5e-6 或 1e-6
4. **增加训练 episodes**: 50 → 100 → 200

---

## 🔗 相关文件

- [configs/grpo_config.yaml](configs/grpo_config.yaml) - 默认配置（当前 0.5B）
- [configs/grpo_config_0.5b.yaml](configs/grpo_config_0.5b.yaml) - 0.5B 优化配置
- [configs/grpo_config_test.yaml](configs/grpo_config_test.yaml) - 快速测试配置
- [scripts/launch_trl_training.sh](scripts/launch_trl_training.sh) - 启动脚本
- [TRL_TRAINING_GUIDE.md](TRL_TRAINING_GUIDE.md) - 完整训练指南
