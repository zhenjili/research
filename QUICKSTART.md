# 🚀 快速开始 - TRL GRPO 训练

## 1️⃣ 安装依赖

```bash
pip install trl[vllm]
```

## 2️⃣ 快速测试（2 episodes，~5 分钟）

```bash
cd /home/ubuntu/jiex/code/research
bash scripts/launch_trl_training.sh configs/grpo_config_test.yaml
```

## 3️⃣ 完整训练（100 episodes，~30-50 分钟）

```bash
bash scripts/launch_trl_training.sh configs/grpo_config_0.5b.yaml
```

## 4️⃣ 监控训练

```bash
# 新开一个 terminal
tail -f logs/trl_training_*/training.log

# 或使用监控脚本
python monitor_training.py --log_dir logs/trl_training_*
```

## 5️⃣ 评估模型

```bash
python evaluation/run_evalplus_vllm.py \
    --model_path outputs/qwen-0.5b-grpo-trl/final \
    --dataset humaneval
```

---

## 📚 详细文档

- **完整训练指南**: [TRL_TRAINING_GUIDE.md](TRL_TRAINING_GUIDE.md)
- **模型配置参考**: [MODEL_CONFIGS.md](MODEL_CONFIGS.md)

---

## ⚙️ 当前配置

- **模型**: Qwen2.5-0.5B-Instruct（最小最快）
- **GPU 分配**:
  - GPU 0-1: vLLM 生成
  - GPU 2-7: DeepSpeed 训练
- **批次大小**: 16 prompts × 16 samples = 256 总样本
- **预期速度**: 15-30 秒/episode

---

## 🔄 切换到更大模型

### 3B 模型
```bash
# 修改 configs/grpo_config.yaml
vim configs/grpo_config.yaml
# 改为: pretrain: "Qwen/Qwen2.5-3B-Instruct"

bash scripts/launch_trl_training.sh
```

### 7B 模型
```bash
# 使用 7B 配置（需要先创建 grpo_config_7b.yaml）
bash scripts/launch_trl_training.sh configs/grpo_config_7b.yaml
```

---

## 🐛 故障排除

### vLLM 启动失败
```bash
# 检查日志
cat logs/trl_training_*/vllm_server.log

# 杀死旧进程
pkill -f vllm
```

### OOM 错误
```bash
# 减小批次大小（编辑配置文件）
micro_train_batch_size: 4 → 2 → 1
train_batch_size: 16 → 8 → 4
```

### 训练太慢
```bash
# 确认 vLLM 正在使用
grep -i "use_vllm" logs/trl_training_*/training.log

# 应该看到: use_vllm=True
```

---

## ✨ TRL 优化特性

✅ **Liger Kernel**: 自动启用，20% 加速 + 60% 内存节省
✅ **Flash Attention**: 自动使用（无需编译）
✅ **vLLM 集成**: 生成速度提升 5-10x
✅ **DeepSpeed ZeRO**: 自动参数分片

---

**就是这么简单！** 🎉
