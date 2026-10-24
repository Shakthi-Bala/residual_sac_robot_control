import csv
import json
from pathlib import Path
import shutil
from xml.sax.saxutils import escape

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from residual_control.artifacts import save_json
from residual_control.env import ROOT
from residual_control.protocol import CONDITIONS


METHODS = ['controller', 'standalone', 'residual']
LABELS = {'controller': 'Cartesian PI', 'standalone': 'Standalone SAC', 'residual': 'Residual SAC'}
COLORS = {'controller': '#64748b', 'standalone': '#d97706', 'residual': '#0284c7'}
METRICS = ['success', 'final_error_m', 'tracking_rmse_m', 'settling_time_s',
           'settled', 'command_acceleration_rms', 'failure']


def summarize(episodes):
    per_seed = episodes.groupby(['method', 'training_seed', 'condition', 'task'])[METRICS].mean().reset_index()
    per_seed.to_csv(ROOT / 'results/per_seed.csv', index=False, lineterminator='\n')
    summary = per_seed.groupby(['method', 'condition', 'task'])[METRICS].agg(['mean', 'std']).fillna(0.)
    summary.columns = ['_'.join(c) for c in summary.columns]
    summary.reset_index().to_csv(ROOT / 'results/summary.csv', index=False, lineterminator='\n')
    overall = episodes.groupby(['method', 'training_seed'])[METRICS].mean().reset_index()
    overall.to_csv(ROOT / 'results/overall_per_seed.csv', index=False, lineterminator='\n')
    return per_seed, overall


def plot_conditions(per_seed):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    x = np.arange(len(CONDITIONS))
    for row, task in enumerate(['reach', 'tracking']):
        for col, (metric, scale, title) in enumerate([('success', 100, 'Success (%)'), ('tracking_rmse_m', 1000, 'Post-approach RMSE (mm)')]):
            ax = axes[row, col]
            for i, method in enumerate(METHODS):
                values = per_seed[(per_seed.method == method) & (per_seed.task == task)]
                mean = values.groupby('condition')[metric].mean().reindex(CONDITIONS) * scale
                std = values.groupby('condition')[metric].std().fillna(0).reindex(CONDITIONS) * scale
                ax.bar(x + (i - 1) * .24, mean, .23, yerr=std, capsize=2, color=COLORS[method], label=LABELS[method])
            ax.set_title(f'{task.capitalize()}: {title}')
            ax.set_xticks(x, ['nominal', 'delay', 'payload', 'combined', '5 kg + delay', '160 ms + load'], rotation=22, ha='right')
            ax.grid(axis='y', alpha=.2)
            if col == 0:
                ax.set_ylim(0, 110)
    axes[0, 0].legend(fontsize=9)
    fig.suptitle('Held-out UR16e targets • mean ± SD across three training seeds\nPI has no training seeds; all methods share 12 targets per task/condition', fontsize=12)
    fig.tight_layout()
    fig.savefig(ROOT / 'results/conditions.png', dpi=170)
    plt.close(fig)


def learning_curves():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    baseline = json.loads((ROOT / 'results/baseline_validation.json').read_text())
    records, curves = [], []
    archive = ROOT / 'results/learning'
    archive.mkdir(exist_ok=True)
    policies = ROOT / 'results/policies'
    policies.mkdir(exist_ok=True)
    for method in METHODS[1:]:
        frames = []
        for seed in [1, 2, 3]:
            folder = ROOT / 'runs/main' / f'{method}_seed{seed}'
            metadata = json.loads((folder / 'metadata.json').read_text())
            if not metadata['complete']:
                raise RuntimeError(f'Incomplete run: {folder}')
            for checkpoint in ['best', 'final']:
                state = torch.load(folder / f'{checkpoint}.pt', map_location='cpu', weights_only=True)
                path = policies / f'{method}_seed{seed}_{checkpoint}.npz'
                np.savez_compressed(path, **{key: value.cpu().numpy() for key, value in state.items()})
            curve = pd.read_csv(folder / 'validation.csv')
            curve['training_seed'], curve['method'] = seed, method
            frames.append(curve)
            shutil.copyfile(folder / 'validation.csv', archive / f'{method}_seed{seed}.csv')
            save_json(archive / f'{method}_seed{seed}.json', metadata)
            records.append({k: metadata.get(k) for k in ['method', 'seed', 'training_interactions', 'validation_interactions',
                                                        'best_validation_step', 'threshold_interactions', 'improvement_interactions', 'wall_seconds']})
        values = pd.concat(frames)
        curves.append(values)
        for ax, metric, scale in zip(axes, ['tracking_rmse_m', 'success'], [1000, 100]):
            grouped = values.groupby('step')[metric]
            mean, std = grouped.mean() * scale, grouped.std() * scale
            ax.plot(mean.index, mean, label=LABELS[method], color=COLORS[method])
            ax.fill_between(mean.index, mean - std, mean + std, alpha=.18, color=COLORS[method])
    axes[0].axhline(baseline['metrics']['tracking_rmse_m'] * 1000, color=COLORS['controller'], ls='--', label='Frozen PI')
    axes[0].axhline(baseline['improvement_rmse_threshold_m'] * 1000, color='#16a34a', ls=':', label='10% improvement threshold')
    axes[1].axhline(baseline['metrics']['success'] * 100, color=COLORS['controller'], ls='--')
    for ax, title in zip(axes, ['Validation RMSE (mm)', 'Validation success (%)']):
        ax.set(xlabel='Training interactions (validation excluded)', ylabel=title)
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('Deterministic validation • mean ± SD, three seeds • lower RMSE is better')
    fig.tight_layout()
    fig.savefig(ROOT / 'results/learning.png', dpi=170)
    plt.close(fig)
    pd.concat(curves).to_csv(ROOT / 'results/learning_curves.csv', index=False, lineterminator='\n')
    save_json(ROOT / 'results/sample_efficiency.json', records)
    return records


def paired_results(episodes):
    keys = ['condition', 'task', 'target_seed']
    base = episodes[episodes.method == 'controller'][keys + ['tracking_rmse_m']].rename(columns={'tracking_rmse_m': 'baseline_rmse_m'})
    paired = episodes[episodes.method != 'controller'].merge(base, on=keys, validate='many_to_one')
    paired['rmse_improvement_mm'] = (paired.baseline_rmse_m - paired.tracking_rmse_m) * 1000
    paired.to_csv(ROOT / 'results/paired_comparisons.csv', index=False, lineterminator='\n')
    worst = paired[paired.method == 'residual'].sort_values(['rmse_improvement_mm', 'tracking_rmse_m'], ascending=[True, False]).head(20)
    worst.to_csv(ROOT / 'results/worst_residual_cases.csv', index=False, lineterminator='\n')
    episodes[episodes.success < 1].sort_values('tracking_rmse_m', ascending=False).to_csv(
        ROOT / 'results/unsuccessful_episodes.csv', index=False, lineterminator='\n')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, task in zip(axes, ['reach', 'tracking']):
        for i, method in enumerate(METHODS[1:]):
            grouped = paired[(paired.task == task) & (paired.method == method)].groupby(['condition', 'training_seed']).rmse_improvement_mm.mean().reset_index()
            mean = grouped.groupby('condition').rmse_improvement_mm.mean().reindex(CONDITIONS)
            std = grouped.groupby('condition').rmse_improvement_mm.std().reindex(CONDITIONS)
            ax.errorbar(np.arange(6) + (i - .5) * .12, mean, yerr=std, fmt='o', capsize=3, label=LABELS[method], color=COLORS[method])
        ax.axhline(0, color='#64748b', lw=1)
        ax.set_title(task.capitalize())
        ax.set_xticks(np.arange(6), CONDITIONS, rotation=30, ha='right')
        ax.set_ylabel('RMSE reduction versus PI (mm)')
        ax.grid(axis='y', alpha=.2)
    axes[0].legend()
    fig.suptitle('Paired targets • positive values improve on the frozen controller • mean ± training-seed SD')
    fig.tight_layout()
    fig.savefig(ROOT / 'results/paired.png', dpi=170)
    plt.close(fig)


def markdown(overall, records):
    lines = ['## Simulation results', '',
             'Best checkpoints were selected on validation targets before held-out evaluation. Each learned method has three training seeds; each policy is tested on 144 matched episodes. Values below average conditions and tasks equally; RMSE is the mean of episode RMSE values, rather than a pooled squared-error estimate. ± denotes descriptive training-seed SD, not a confidence interval. The conventional controller is deterministic and has no training-seed variation.', '',
             '| Method | Success | RMSE after 1 s | Final error | Command acceleration RMS | Failures |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    totals = {}
    for method in METHODS:
        values = overall[overall.method == method]
        mean, std = values[METRICS].mean(), values[METRICS].std().fillna(0)
        totals[method] = mean.to_dict()
        def fmt(key, scale=1, suffix=''):
            return f'{mean[key] * scale:.1f} ± {std[key] * scale:.1f}{suffix}' if method != 'controller' else f'{mean[key] * scale:.1f}{suffix}'
        lines.append(f"| {LABELS[method]} | {fmt('success', 100, '%')} | {fmt('tracking_rmse_m', 1000, ' mm')} | {fmt('final_error_m', 1000, ' mm')} | {fmt('command_acceleration_rms', 1, ' rad/s²')} | {fmt('failure', 100, '%')} |")
    lines += ['', '![Held-out success and error](results/conditions.png)', '', '![Validation learning curves](results/learning.png)', '', '![Paired comparison](results/paired.png)', '',
              '### Training interactions and checkpoint selection', '',
              '| Method | Seed | Best validation step | Absolute criterion confirmation step | Baseline improvement confirmation step |',
              '| --- | ---: | ---: | ---: | ---: |']
    for r in records:
        absolute = r.get('threshold_interactions')
        improvement = r.get('improvement_interactions')
        lines.append(f"| {LABELS[r['method']]} | {r['seed']} | {r['best_validation_step']:,} | {absolute if absolute is not None else '>100,000 (not reached)'} | {improvement if improvement is not None else '>100,000 (not reached)'} |")
    residual_rmse = totals['residual']['tracking_rmse_m']
    baseline_rmse = totals['controller']['tracking_rmse_m']
    improvement = (baseline_rmse - residual_rmse) / baseline_rmse * 100
    stepzero = sum(r['method'] == 'residual' and r['best_validation_step'] == 0 for r in records)
    lines += ['', '### Interpretation and failures', '',
              f'Residual SAC changes aggregate held-out RMSE by {improvement:+.1f}% relative to PI (positive means improvement). {stepzero}/3 residual runs select the initial zero-residual checkpoint. A step-zero selection means the controller was retained; it is not evidence that reinforcement learning improved it.', '',
              'The absolute criterion measures task competence, which residual SAC can inherit before any training. The stricter criterion requires at least 10% lower validation RMSE than the frozen PI controller with no loss of success, at two consecutive checkpoints. Crossings report the second checkpoint, when the pair is confirmed. Validation is sampled every 10,000 interactions, so crossing times have that resolution.', '',
              'A strong PI controller, local workspace, short episodes, and unnormalized observations can leave little headroom for residual learning. Entropy-driven corrections can also disrupt accurate feedback control. These experiments do not establish superiority across other workspaces, controller gains, training budgets, or reward choices. Checkpoint selection can retain PI even when late training regresses; the learning plots expose those regressions.', '',
              'The aggregate includes out-of-distribution payload/delay conditions. Inspect `summary.csv` for each condition/task and `worst_residual_cases.csv` for the weakest paired comparisons (ties ordered by error). `unsuccessful_episodes.csv` lists every missed target or terminal failure. If all selected residuals equal PI, paired differences are zero rather than learned gains. RMSE uses samples at and after 1 second; final error is endpoint error. Settling time means first entry into 2 cm followed by staying there, capped at 5 seconds; `settled` records whether the cap is censoring. Command acceleration measures issued commands after filtering, not physical joint acceleration. Early failures use observed samples and are separately counted as unsuccessful.', '',
              '### Artifacts', '',
              '- [Per-episode held-out results](results/episodes.csv), [condition summaries](results/summary.csv), and [per-seed aggregates](results/overall_per_seed.csv).',
              '- [Matched target comparisons](results/paired_comparisons.csv), [worst residual cases](results/worst_residual_cases.csv), and [all unsuccessful episodes](results/unsuccessful_episodes.csv).',
              '- [Training interaction thresholds](results/sample_efficiency.json), [validation histories](results/learning_curves.csv), and immutable run metadata in `results/learning/`.',
              '- Portable selected and final model weights in `results/policies/` (NumPy arrays, no pickle); model provenance and mesh license in `models/`; GIF provenance beside every animation.', '']
    tuning = json.loads((ROOT / 'results/controller_tuning.json').read_text())
    total_training = sum(r['training_interactions'] for r in records)
    total_validation = sum(r['validation_interactions'] for r in records)
    evaluations = [json.loads(p.read_text()) for p in (ROOT / 'results/evaluation').glob('*.json')]
    total_test = sum(r['test_interactions'] for r in evaluations)
    lines += ['### Interaction accounting', '',
              f"The controller grid used {tuning['calibration_interactions']:,} calibration interactions, shared by the conventional and residual methods. A separate 4,000-interaction baseline validation fixed the relative criterion. SAC used {total_training:,} training interactions across six runs and {total_validation:,} additional validation interactions, excluded from replay. Held-out evaluation used {total_test:,} interactions. Rendering replays are excluded from these learning counts. Controller calibration cost is not zero and is shared across residual seeds. Any claim about total data efficiency must include that cost; step-zero residual competence alone does not demonstrate cheaper learning.", '']
    result = '\n'.join(lines)
    (ROOT / 'results/results.md').write_text(result)
    root_summary = ['## Residual control study', '',
                    '**Question:** Can bounded residual SAC improve a tuned controller under hidden payload and delay?', '',
                    'A simulation study on a six-joint UR16e compares Cartesian PI, standalone SAC, and residual SAC with three training seeds, equal budgets, and 1,008 matched held-out episodes.', '']
    root_summary += lines[4:9]
    root_summary += ['', f'Residual SAC changes aggregate RMSE by {improvement:+.1f}% relative to PI (positive means improvement). {stepzero}/3 residual seeds retain the initial zero-correction checkpoint. Retaining PI is not learned improvement.', '',
                     '[Full protocol and results](residual-control/README.md) · [Report PDF](residual-control/results/report.pdf) · [Learning curves](residual-control/results/learning.png) · [Per-episode data](residual-control/results/episodes.csv)', '']
    (ROOT / 'results/root_summary.md').write_text('\n'.join(root_summary))
    save_json(ROOT / 'results/aggregate.json', {'metrics': totals, 'residual_rmse_improvement_percent': improvement})
    readme = ROOT / 'README.md'
    text = readme.read_text().split('<!-- GENERATED RESULTS -->')[0].rstrip()
    readme.write_text(text + '\n\n<!-- GENERATED RESULTS -->\n\n' + result)


def pdf_report():
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle

    styles = getSampleStyleSheet()
    styles['BodyText'].fontSize = 10
    styles['BodyText'].leading = 14
    aggregate = json.loads((ROOT / 'results/aggregate.json').read_text())
    story = [Paragraph('Residual SAC for Robust UR16e Control', styles['Title']),
             Paragraph('Simulation study | Three training seeds | Matched held-out targets', styles['Heading2'])]
    paragraphs = [
        'Question: can a bounded SAC correction improve a tuned feedback controller under unknown payload and command delay, using fewer training interactions than SAC controlling the full outer loop?',
        'The free-space UR16e model uses six joints, torque-limited velocity servos, 50 Hz outer control, and 2 ms physics steps. The Cartesian PI controller is calibrated on separate targets. SAC and residual SAC share observations, training budgets, networks, rewards, and command filters. Residual corrections are bounded to 0.2 rad/s per joint before final filtering.',
        'Training randomizes added payload from 0 to 3 kg and delay from 0 to 100 ms. Tests include nominal, delay-only, payload-only, combined, 5 kg with delay, and 160 ms with load. Each policy faces 12 held-out target seeds for reaching and tracking in each condition. Validation selects checkpoints without test feedback.',
        f"Residual SAC changes aggregate held-out RMSE by {aggregate['residual_rmse_improvement_percent']:+.1f}% relative to PI; positive means improvement. Initial zero-residual policies can inherit baseline competence. Validation-selected weights and final training weights are both preserved, and demonstration GIFs show both.",
    ]
    for paragraph in paragraphs:
        story += [Paragraph(escape(paragraph), styles['BodyText']), Spacer(1, .12 * inch)]
    table = [['Method', 'Success', 'RMSE (mm)', 'Final error (mm)', 'Failure']]
    for method in METHODS:
        row = aggregate['metrics'][method]
        table.append([LABELS[method], f"{row['success'] * 100:.1f}%", f"{row['tracking_rmse_m'] * 1000:.1f}",
                      f"{row['final_error_m'] * 1000:.1f}", f"{row['failure'] * 100:.1f}%"])
    t = Table(table, repeatRows=1)
    t.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#dbeafe')),
                           ('BOTTOMPADDING', (0, 0), (-1, -1), 9),
                           ('TOPPADDING', (0, 0), (-1, -1), 9)]))
    story.append(t)
    for title, image, caption in [
        ('Held-out conditions', 'conditions.png', 'Bars average target episodes within each training seed, then across three seeds. Error bars are descriptive seed SD. PI is evaluated once. RMSE uses samples after the one-second approach; early failures are separately unsuccessful.'),
        ('Learning and interaction use', 'learning.png', 'Validation is deterministic and excluded from replay. Initial residual actions are zero. Qualification requires two consecutive checkpoints; baseline-relative improvement additionally requires 10% lower RMSE without reduced success. Missing threshold crossings remain censored.'),
        ('Paired comparisons and limits', 'paired.png', 'Positive values improve on PI for the same target trajectory and perturbation. Conditions beyond the training ranges test limited extrapolation. Three training seeds and one local workspace do not establish general superiority or statistical significance.'),
    ]:
        story += [PageBreak(), Paragraph(title, styles['Heading1']),
                  Image(str(ROOT / 'results' / image), width=7 * inch, height=4.4 * inch if image == 'conditions.png' else 2.5 * inch),
                  Paragraph(escape(caption), styles['BodyText']), Spacer(1, .15 * inch)]
        if image == 'paired.png':
            for text in [
                'Limitations: contacts are disabled; this is not obstacle avoidance or a physical robot validation. The servo and payload inertias are simplified. Recent commands help with hidden delay but do not guarantee a Markov observation. Results depend on reward, initialization, controller tuning, target distribution, and budget.',
                'Failure analysis: worst_residual_cases.csv lists the largest paired regressions. summary.csv records task/condition results, settled flags, capped settling times, issued-command acceleration, and unsuccessful episodes. The learning curves expose regressions masked by retaining an early validation checkpoint.',
                'Reproduction: see residual-control/README.md, configs/experiment.yaml, results/learning/ metadata and CSV histories, and portable best/final weights in results/policies/. Mesh provenance and licensing are in models/. This study extends the repository SAC code; the actor objective uses the minimum of two critics.',
            ]:
                story += [Paragraph(escape(text), styles['BodyText']), Spacer(1, .12 * inch)]
    SimpleDocTemplate(str(ROOT / 'results/report.pdf'), pagesize=(8.5 * inch, 11 * inch),
                      rightMargin=.65 * inch, leftMargin=.65 * inch, title='Residual SAC for Robust UR16e Control').build(story)


def main():
    expected = [ROOT / 'results/evaluation' / f'{m}_seed{s}.csv' for m in METHODS for s in ([0] if m == 'controller' else [1, 2, 3])]
    episodes = pd.concat([pd.read_csv(p) for p in expected], ignore_index=True)
    if len(episodes) != 1008 or episodes.duplicated(['method', 'training_seed', 'condition', 'task', 'target_seed']).any():
        raise RuntimeError('Incomplete or duplicate held-out episodes')
    episodes.to_csv(ROOT / 'results/episodes.csv', index=False, lineterminator='\n')
    per_seed, overall = summarize(episodes)
    plot_conditions(per_seed)
    records = learning_curves()
    paired_results(episodes)
    markdown(overall, records)
    pdf_report()
    print(ROOT / 'results/results.md')


if __name__ == '__main__':
    main()
