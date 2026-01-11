#!/usr/bin/env python3
"""
Real-time monitoring script for GRPO training.
Monitors GPU utilization, training metrics, and saves logs.
"""

import time
import re
import subprocess
import json
from pathlib import Path
from datetime import datetime
from collections import deque

class TrainingMonitor:
    def __init__(self, log_file, output_file="./outputs/training_metrics.jsonl"):
        self.log_file = log_file
        self.output_file = output_file
        self.metrics_history = []
        self.last_position = 0

        # Create output directory
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    def get_gpu_stats(self):
        """Get current GPU utilization and memory usage."""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu',
                 '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                timeout=5
            )

            gpu_stats = []
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    idx, util, mem_used, mem_total, power, temp = line.split(',')
                    gpu_stats.append({
                        'gpu_id': int(idx.strip()),
                        'utilization': float(util.strip()),
                        'memory_used_mb': float(mem_used.strip()),
                        'memory_total_mb': float(mem_total.strip()),
                        'power_draw_w': float(power.strip()),
                        'temperature_c': float(temp.strip()),
                    })
            return gpu_stats
        except Exception as e:
            print(f"Error getting GPU stats: {e}")
            return []

    def parse_log_file(self):
        """Parse new entries from the log file."""
        try:
            with open(self.log_file, 'r') as f:
                f.seek(self.last_position)
                new_content = f.read()
                self.last_position = f.tell()

            return new_content
        except FileNotFoundError:
            return ""

    def extract_metrics(self, content):
        """Extract training metrics from log content."""
        metrics = {}

        # Extract episode number
        episode_match = re.search(r'\[Episode (\d+)\]', content)
        if episode_match:
            metrics['episode'] = int(episode_match.group(1))

        # Extract timing information
        complete_match = re.search(
            r'\[Episode \d+\] Complete in ([\d.]+)s \(gen: ([\d.]+)s, reward: ([\d.]+)s, train: ([\d.]+)s\)',
            content
        )
        if complete_match:
            metrics['total_time'] = float(complete_match.group(1))
            metrics['gen_time'] = float(complete_match.group(2))
            metrics['reward_time'] = float(complete_match.group(3))
            metrics['train_time'] = float(complete_match.group(4))

        # Extract training metrics
        if 'Training metrics:' in content:
            # Policy loss
            loss_match = re.search(r'Policy loss: ([-\d.]+)', content)
            if loss_match:
                metrics['policy_loss'] = float(loss_match.group(1))

            # Total loss
            total_loss_match = re.search(r'Total loss: ([-\d.]+)', content)
            if total_loss_match:
                metrics['total_loss'] = float(total_loss_match.group(1))

            # Entropy
            entropy_match = re.search(r'Entropy: ([\d.]+)', content)
            if entropy_match:
                metrics['entropy'] = float(entropy_match.group(1))

            # Mean advantage
            adv_match = re.search(r'Mean advantage: ([-\d.]+)', content)
            if adv_match:
                metrics['mean_advantage'] = float(adv_match.group(1))

            # Mean log prob
            logprob_match = re.search(r'Mean log prob: ([-\d.]+)', content)
            if logprob_match:
                metrics['mean_log_prob'] = float(logprob_match.group(1))

            # Learning rate
            lr_match = re.search(r'Learning rate: ([\d.e+-]+)', content)
            if lr_match:
                metrics['learning_rate'] = float(lr_match.group(1))

        # Extract reward information
        reward_match = re.search(r'Rewards computed in [\d.]+s, mean: ([\d.]+)', content)
        if reward_match:
            metrics['mean_reward'] = float(reward_match.group(1))

        return metrics if metrics else None

    def display_metrics(self, metrics, gpu_stats):
        """Display formatted metrics."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print("\n" + "="*80)
        print(f"[{timestamp}] Episode {metrics.get('episode', 'N/A')}")
        print("="*80)

        # Training metrics
        if 'policy_loss' in metrics:
            print("\n📊 Training Metrics:")
            print(f"  Policy Loss:      {metrics.get('policy_loss', 'N/A'):>10.4f}")
            print(f"  Total Loss:       {metrics.get('total_loss', 'N/A'):>10.4f}")
            print(f"  Entropy:          {metrics.get('entropy', 'N/A'):>10.2f}")
            print(f"  Mean Reward:      {metrics.get('mean_reward', 'N/A'):>10.3f}")
            print(f"  Mean Advantage:   {metrics.get('mean_advantage', 'N/A'):>10.4f}")
            print(f"  Mean Log Prob:    {metrics.get('mean_log_prob', 'N/A'):>10.2f}")
            print(f"  Learning Rate:    {metrics.get('learning_rate', 'N/A'):>10.2e}")

        # Timing
        if 'total_time' in metrics:
            print("\n⏱️  Timing:")
            print(f"  Generation:       {metrics.get('gen_time', 'N/A'):>10.2f}s")
            print(f"  Reward Compute:   {metrics.get('reward_time', 'N/A'):>10.2f}s")
            print(f"  Policy Update:    {metrics.get('train_time', 'N/A'):>10.2f}s")
            print(f"  Total:            {metrics.get('total_time', 'N/A'):>10.2f}s")

        # GPU stats
        if gpu_stats:
            print("\n🖥️  GPU Utilization:")
            for gpu in gpu_stats[:4]:  # Show first 4 GPUs
                util = gpu['utilization']
                mem_pct = (gpu['memory_used_mb'] / gpu['memory_total_mb']) * 100
                print(f"  GPU {gpu['gpu_id']}: {util:>5.1f}% | "
                      f"Mem: {mem_pct:>5.1f}% ({gpu['memory_used_mb']/1024:.1f}/{gpu['memory_total_mb']/1024:.1f} GB) | "
                      f"Temp: {gpu['temperature_c']:>4.0f}°C | "
                      f"Power: {gpu['power_draw_w']:>5.0f}W")

        print("="*80 + "\n")

    def save_metrics(self, metrics, gpu_stats):
        """Save metrics to JSONL file."""
        record = {
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'gpu_stats': gpu_stats,
        }

        with open(self.output_file, 'a') as f:
            f.write(json.dumps(record) + '\n')

    def generate_summary(self):
        """Generate training summary from collected metrics."""
        if not Path(self.output_file).exists():
            return

        metrics = []
        with open(self.output_file, 'r') as f:
            for line in f:
                if line.strip():
                    metrics.append(json.loads(line))

        if not metrics:
            return

        print("\n" + "="*80)
        print("📈 Training Summary")
        print("="*80)

        # Extract episode metrics
        episodes = [m['metrics'].get('episode') for m in metrics if 'episode' in m['metrics']]
        rewards = [m['metrics'].get('mean_reward') for m in metrics if 'mean_reward' in m['metrics']]
        losses = [m['metrics'].get('policy_loss') for m in metrics if 'policy_loss' in m['metrics']]

        if episodes:
            print(f"\nTotal Episodes: {max(episodes) + 1}")

        if rewards:
            print(f"\nReward Statistics:")
            print(f"  Initial:  {rewards[0]:.3f}")
            print(f"  Final:    {rewards[-1]:.3f}")
            print(f"  Mean:     {sum(rewards)/len(rewards):.3f}")
            print(f"  Max:      {max(rewards):.3f}")

        if losses:
            print(f"\nPolicy Loss Statistics:")
            print(f"  Initial:  {losses[0]:.4f}")
            print(f"  Final:    {losses[-1]:.4f}")
            print(f"  Mean:     {sum(losses)/len(losses):.4f}")

        # GPU utilization
        all_gpu_utils = []
        for m in metrics:
            if 'gpu_stats' in m and m['gpu_stats']:
                all_gpu_utils.extend([g['utilization'] for g in m['gpu_stats']])

        if all_gpu_utils:
            print(f"\nAverage GPU Utilization: {sum(all_gpu_utils)/len(all_gpu_utils):.1f}%")

        print("="*80 + "\n")

    def monitor(self, check_interval=10):
        """Main monitoring loop."""
        print("🔍 Starting GRPO Training Monitor")
        print(f"📝 Log file: {self.log_file}")
        print(f"💾 Output file: {self.output_file}")
        print(f"⏱️  Check interval: {check_interval}s\n")

        last_episode = -1

        try:
            while True:
                # Parse new log content
                new_content = self.parse_log_file()

                if new_content:
                    metrics = self.extract_metrics(new_content)

                    if metrics and 'episode' in metrics:
                        current_episode = metrics['episode']

                        # Only display when episode completes
                        if current_episode != last_episode and 'total_time' in metrics:
                            gpu_stats = self.get_gpu_stats()
                            self.display_metrics(metrics, gpu_stats)
                            self.save_metrics(metrics, gpu_stats)
                            last_episode = current_episode

                # Check if training is complete
                if 'Training complete!' in new_content:
                    print("\n✅ Training completed!")
                    self.generate_summary()
                    break

                time.sleep(check_interval)

        except KeyboardInterrupt:
            print("\n\n⚠️  Monitor stopped by user")
            self.generate_summary()
        except Exception as e:
            print(f"\n❌ Error during monitoring: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Monitor GRPO training")
    parser.add_argument("--log-file", type=str, required=True, help="Path to training log file")
    parser.add_argument("--output", type=str, default="./outputs/training_metrics.jsonl",
                        help="Path to save metrics")
    parser.add_argument("--interval", type=int, default=10,
                        help="Check interval in seconds")

    args = parser.parse_args()

    monitor = TrainingMonitor(args.log_file, args.output)
    monitor.monitor(args.interval)
