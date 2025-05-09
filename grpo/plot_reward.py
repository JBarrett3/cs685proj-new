#!/usr/bin/env python3
import os
import glob
import json
import argparse

import numpy as np
import matplotlib.pyplot as plt

def load_log_history(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data.get('log_history', [])

def plot_with_trendline(output_dir, pattern=''):
    files = glob.glob(os.path.join(output_dir, 'checkpoint-*', 'trainer_state.json'))
    if pattern:
        files = [f for f in files if pattern in os.path.basename(os.path.dirname(f))]

    def idx(p):
        try:
            return int(os.path.basename(os.path.dirname(p)).split('-')[-1])
        except:
            return float('inf')
    files = sorted(files, key=idx)

    all_steps, all_rewards = [], []
    prev_max = 0
    for path in files:
        history = load_log_history(path)
        new = [(r['step'], r['reward']) for r in history if r.get('step', 0) > prev_max]
        if not new:
            continue
        steps, rewards = zip(*new)
        all_steps.extend(steps)
        all_rewards.extend(rewards)
        prev_max = steps[-1]

    if not all_steps:
        return

    steps = np.array(all_steps)
    rewards = np.array(all_rewards)

    slope, intercept = np.polyfit(steps, rewards, 1)
    trend = slope * steps + intercept

    plt.figure(figsize=(10, 6))
    plt.plot(steps, rewards, linewidth=1, alpha=0.6, label='Raw')
    plt.plot(steps, trend,        linewidth=2)
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.title('GRPO Reward Curve')
    plt.legend(loc='best', fontsize='small')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("reward.png")
    plt.show()

if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description="Rewards with best‐fit line."
    )
    p.add_argument('-d','--dir',    default='epoch1v2/outputs', help='Root outputs folder')
    p.add_argument('-p','--pattern',default='', help='Filter checkpoints by substring')
    args = p.parse_args()
    plot_with_trendline(args.dir, args.pattern)
