from __future__ import print_function

import glob
import os
import sys

from matplotlib import pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from tensorflow.python.summary.event_accumulator import EventAccumulator

RUNS = {
    # Baseline.
    'results-pong': {
        'name': 'Baseline',
        'annotation': 'Training on\nnon-noisy environment'
    },

    # Training on random noise.
    'results-pong-noise-random-0.01': {
        'name': 'Random Noise Training (0.01)',
        'annotation': 'Re-training with\nrandom noise',
        'baselines': ['results-pong']
    },

    'results-pong-noise-random-0.02': {
        'name': 'Random Noise Training (0.02)',
        'annotation': 'Re-training with\nrandom noise',
        'baselines': ['results-pong']
    },

    'results-pong-noise-random-0.05': {
        'name': 'Random Noise Training (0.05)',
        'annotation': 'Re-training with\nrandom noise',
        'baselines': ['results-pong']
    },

    'results-pong-noise-random-0.1': {
        'name': 'Random Noise Training (0.1)',
        'annotation': 'Re-training with\nrandom noise',
        'baselines': ['results-pong']
    },

    'results-pong-noise-random-0.2': {
        'name': 'Random Noise Training (0.2)',
        'annotation': 'Re-training with\nrandom noise',
        'baselines': ['results-pong']
    },

    'results-pong-noise-random-0.5': {
        'name': 'Random Noise Training (0.5)',
        'annotation': 'Re-training with\nrandom noise',
        'baselines': ['results-pong']
    },

    # Evaluation on random noise.
    'results-pong-noise-random-0.01-f': {
        'name': 'Random Noise Evaluation (0.01)',
        'annotation': 'Random Noise\nEvaluation',
        'baselines': ['results-pong']
    },

    'results-pong-noise-random-0.02-f': {
        'name': 'Random Noise Evaluation (0.02)',
        'annotation': 'Random Noise\nEvaluation',
        'baselines': ['results-pong']
    },

    'results-pong-noise-random-0.05-f': {
        'name': 'Random Noise Evaluation (0.05)',
        'annotation': 'Random Noise\nEvaluation',
        'baselines': ['results-pong']
    },

    # Evaluation on FGSM noise.
    'results-pong-noise-fgsm-0.001-f': {
        'name': 'FGSM Evaluation (0.001)',
        'annotation': 'Adversarial\nEvaluation',
        'baselines': ['results-pong']
    },

    'results-pong-noise-fgsm-0.005-f': {
        'name': 'FGSM Evaluation (0.005)',
        'annotation': 'Adversarial\nEvaluation',
        'baselines': ['results-pong']
    },

    'results-pong-noise-fgsm-0.01-f': {
        'name': 'FGSM Evaluation (0.01)',
        'annotation': 'Adversarial\nEvaluation',
        'baselines': ['results-pong']
    },

    # Evaluation on FGSM noise (skip, reuse).
    'results-pong-noise-fgsm-0.001-skip-5': {
        'name': 'FGSM (0.001), Skip 5',
        'annotation': 'Adversarial\nEvaluation',
        'baselines': ['results-pong']
    },

    'results-pong-noise-fgsm-0.001-reuse-5': {
        'name': 'FGSM (0.001), Skip 5 (reuse)',
        'annotation': 'Adversarial\nEvaluation',
        'baselines': ['results-pong']
    },

    'results-pong-noise-fgsm-0.001-vfskip': {
        'name': 'FGSM (0.001), VF Skip',
        'annotation': 'Adversarial\nEvaluation',
        'baselines': ['results-pong']
    },

    'results-pong-noise-fgsm-0.005-skip-10': {
        'name': 'FGSM (0.005), Skip 10',
        'annotation': 'Adversarial\nEvaluation',
        'baselines': ['results-pong']
    },

    'results-pong-noise-fgsm-0.005-reuse-10': {
        'name': 'FGSM (0.005), Skip 10 (reuse)',
        'annotation': 'Adversarial\nEvaluation',
        'baselines': ['results-pong']
    },

    'results-pong-noise-fgsm-0.005-vfskip-1.4': {
        'name': 'FGSM (0.005), VF Skip',
        'annotation': 'Adversarial\nEvaluation',
        'baselines': ['results-pong']
    },

    # Training on random noise, evaluation on FGSM noise.
    'results-pong-noise-fgsm-0.001-f-pt-random-0.05': {
        'name': 'FGSM (0.001) After Random Noise Re-training (0.05)',
        'annotation': 'Adversarial\nEvaluation',
        'baselines': ['results-pong', 'results-pong-noise-random-0.05'],
    },

    'results-pong-noise-fgsm-0.001-f-pt-random-0.1': {
        'name': 'FGSM (0.001) After Random Noise Re-training (0.1)',
        'annotation': 'Adversarial\nEvaluation',
        'baselines': ['results-pong', 'results-pong-noise-random-0.1'],
    },

    'results-pong-noise-fgsm-0.005-f-pt-random-0.05': {
        'name': 'FGSM (0.005) After Random Noise Re-training (0.05)',
        'annotation': 'Adversarial\nEvaluation',
        'baselines': ['results-pong', 'results-pong-noise-random-0.05'],
    },

    'results-pong-noise-fgsm-0.005-f-pt-random-0.1': {
        'name': 'FGSM (0.005) After Random Noise Re-training (0.1)',
        'annotation': 'Adversarial\nEvaluation',
        'baselines': ['results-pong', 'results-pong-noise-random-0.1'],
    },

    'results-pong-noise-fgsm-0.005-f-pt-random-0.2': {
        'name': 'FGSM (0.005) After Random Noise Re-training (0.2)',
        'annotation': 'Adversarial\nEvaluation',
        'baselines': ['results-pong', 'results-pong-noise-random-0.2'],
    },

    'results-pong-noise-fgsm-0.01-f-pt-random-0.1': {
        'name': 'FGSM (0.01) After Random Noise Re-training (0.1)',
        'annotation': 'Adversarial\nEvaluation',
        'baselines': ['results-pong', 'results-pong-noise-random-0.1'],
    },

    # Training on FGSM.
    'results-pong-noise-fgsm-0.001': {
        'name': 'FGSM Training (0.001)',
        'annotation': 'Re-training\nwith FGSM',
        'baselines': ['results-pong']
    },

    'results-pong-noise-fgsm-0.005': {
        'name': 'FGSM Training (0.005)',
        'annotation': 'Re-training\nwith FGSM',
        'baselines': ['results-pong']
    },

    # Training on FGSM, evaluation on FGSM.
    'results-pong-noise-fgsm-0.005-f-pt-fgsm-0.001': {
        'name': 'FGSM (0.005) After FGSM Re-training (0.001)',
        'annotation': 'Adversarial\nEvaluation',
        'baselines': ['results-pong', 'results-pong-noise-fgsm-0.001'],
    },

    'results-pong-noise-fgsm-0.01-f-pt-fgsm-0.005': {
        'name': 'FGSM (0.01) After FGSM Re-training (0.005)',
        'annotation': 'Adversarial\nEvaluation',
        'baselines': ['results-pong', 'results-pong-noise-fgsm-0.005'],
    },

    'results-pong-noise-fgsm-0.001-f-pt-fgsm-0.005': {
        'name': 'FGSM (0.001) After FGSM Re-training (0.005)',
        'annotation': 'Adversarial\nEvaluation',
        'baselines': ['results-pong', 'results-pong-noise-fgsm-0.005'],
    },

    # Episode value function.
    'results-pong-vf': {
        'name': 'Baseline',
        'vf': True,
    },

    'results-pong-noise-fgsm-0.001-vf': {
        'name': 'FGSM (0.001)',
        'vf': True,
    }
}

if len(sys.argv) >= 2:
    filter_runs = {}
    for run in sys.argv[1:]:
        filter_runs[run] = RUNS[run]

    RUNS = filter_runs


def running_mean(x, n):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[n:] - cumsum[:-n]) / n

# Process all runs.
run_rewards = {}
run_episode_vf = {}
run_episode_reward = {}
for run_id, run in RUNS.items():
    rewards = {}

    for worker in glob.glob(os.path.join(run_id, 'train_*')):
        print('Processing {}.'.format(worker))
        ea = EventAccumulator(worker)
        ea.Reload()

        if run.get('vf', False):
            run_episode_vf[run_id] = np.asarray(
                [[scalar.step, scalar.value] for scalar in ea.Scalars('episode/vf')],
            )
            run_episode_reward[run_id] = np.asarray(
                [[scalar.step, scalar.value] for scalar in ea.Scalars('episode/reward')],
            )
        else:
            for scalar in ea.Scalars('global/episode_reward'):
                rewards.setdefault(scalar.step, []).append(scalar.value)

    run_rewards[run_id] = np.asarray(
        sorted([[step, np.mean(value)] for step, value in rewards.items()]),
        dtype=np.int64
    )

plt.style.use('ggplot')
for run_id, run in RUNS.items():
    print('Plotting {}.'.format(run_id))
    if run.get('vf', False):
        vf = run_episode_vf[run_id]
        reward = run_episode_reward[run_id]

        fig, ax1 = plt.subplots()
        plt.title(run['name'])

        # Value function.
        ax1.plot(vf[:, 0], vf[:, 1], color='red', label='Value Function')

        # Rewards.
        ax2 = ax1.twinx()
        ax2.plot(reward[:, 0], reward[:, 1], color='blue', label='Rewards')
        ax2.set_ylim(np.min(reward[:, 1]) - 1, np.max(reward[:, 1]) + 1)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        plt.legend(lines1 + lines2, labels1 + labels2, loc='lower right')

        plt.savefig('figures/{}.pdf'.format(run_id), bbox_inches='tight', pad_inches=0)
        plt.savefig('figures/{}.png'.format(run_id))
        plt.close()
    else:
        rewards = run_rewards[run_id]
        x = rewards[:, 0]
        y = rewards[:, 1]

        # Smooth.
        n = 100
        smooth_x = x[n:]
        smooth_y = running_mean(y, n + 1)
        smooth_y = savgol_filter(smooth_y, 51, 3)

        plt.title(run['name'])
        plt.plot(x, y, alpha=0.4, color='blue')
        plt.plot(smooth_x, smooth_y, color='blue', linewidth=2)

        # Draw annotations.
        prev_baseline_x = 0
        for baseline in run.get('baselines', []) + [run_id]:
            baseline_x = run_rewards[baseline][-1, 0]
            if baseline != run_id:
                plt.axvline(x=baseline_x, color='red', linewidth=3)

            plt.text(
                (baseline_x + prev_baseline_x) / 2.,
                25,
                RUNS[baseline]['annotation'],
                horizontalalignment='center',
                verticalalignment='center',
                bbox=dict(facecolor='red', alpha=0.5),
                wrap=True,
            )
            prev_baseline_x = baseline_x

        plt.savefig('figures/{}.pdf'.format(run_id), bbox_inches='tight', pad_inches=0)
        plt.savefig('figures/{}.png'.format(run_id))
        plt.close()
