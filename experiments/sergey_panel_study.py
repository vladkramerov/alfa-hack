"""SPEC011 sequential panel hypotheses; old control is read-only cached data."""
import argparse
import hashlib
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from automl_data import load_task, score, split_key
from automl_panel_memory import PanelHistory, PanelDynamics, PanelStateForest

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / 'results/2026-10-24/sergey-improvement-v1'


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding='utf8')


def main(number, stage):
    assert subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip() == 'sergey_branch'
    ledger = (ROOT / 'HYPOTHESES.md').read_text(encoding='utf8')
    hypothesis = f'HYP-{number:03d}'
    if hypothesis not in ledger or number not in (70, 71, 72, 73, 74):
        raise ValueError('Unregistered mechanism')
    method = 'stable' if number == 70 else 'reliability'
    folder = ROOT / f'results/2026-10-24/{hypothesis}-{1 if stage == "primary" else 2:02d}-regression-panel-{stage}'
    folder.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    (folder / 'report.md').write_text(f'# {hypothesis}/{stage}\n\nStarted before fit. [SPEC011](../../../docs/specs/SPEC-011-sergey-panel-research.md).\n\nCommand: `.venv/bin/python experiments/sergey_panel_study.py --number {number} --stage {stage}`.\n')
    result = {}
    try:
        bundle = load_task(ROOT / 'data/regression/task.md')
        control = SUITE / f'control-regression-{stage}'
        with np.load(control / 'splits.npz', allow_pickle=False) as z:
            fit, valid = z['fit'], z['valid']
        with np.load(control / 'validation_predictions.npz', allow_pickle=False) as z:
            assert np.array_equal(z['positions'], valid)
            assert np.array_equal(z['y'], bundle.y.iloc[valid])
            baseline, base = z['prediction'], z['base_prediction']
        configuration = {'method': method, 'fields': ['State_Factor', 'building_class', 'facility_type', 'floor_area', 'year_built'],
                         'time_column': 'Year_Factor', 'base': 'read_only_cached_H43_feature_experts'}
        identity = {'hypothesis': hypothesis, 'stage': stage, 'configuration': configuration,
                    'hashes': bundle.hashes, 'split_key': split_key(fit, valid),
                    'ledger_sha256_before_fit': hashlib.sha256(ledger.encode()).hexdigest(),
                    'source_sha256': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                                      for p in ['automl_panel_memory.py', 'experiments/sergey_panel_study.py']}}
        save(folder / 'evidence.json', identity)
        np.savez_compressed(folder / 'splits.npz', fit=fit, valid=valid)
        if number == 73:
            configuration['method'] = 'fixed_panel_dynamics_blend'
            configuration['weights'] = [.5,.5]
            save(folder/'evidence.json', identity)
            parts=[]
            for part in [70,72]:
                path=ROOT/f'results/2026-10-24/HYP-{part:03d}-{1 if stage == "primary" else 2:02d}-regression-panel-{stage}/validation_predictions.npz'
                with np.load(path,allow_pickle=False) as z:
                    assert np.array_equal(z['positions'],valid)
                    parts.append(z['prediction'])
            model = PanelHistory().fit(bundle.X.iloc[fit], bundle.y.iloc[fit])
            prediction = .5*parts[0]+.5*parts[1]
            mean,alpha=np.zeros(len(valid)),np.zeros(len(valid))
            audit={'fixed_blend_weights':[.5,.5], 'reused_components':['HYP-070','HYP-072'], 'new_tree_fits':0}
        elif number in (72,74):
            import os
            os.environ.update(COMBO_RUN_DIR=str(SUITE/'new-fit-log'), COMBO_LEDGER_PATH=str(ROOT/'HYPOTHESES.md'), COMBO_HYPOTHESIS=hypothesis+'/'+stage)
            configuration['method'] = 'causal_panel_dynamics' if number == 72 else 'causal_panel_state_forest'
            configuration['parameters'] = dict(n_estimators=600, learning_rate=.03, num_leaves=15, min_child_samples=40, reg_lambda=10.)
            if number == 74: configuration['parameters'] = dict(n_estimators=300,max_depth=24,min_samples_leaf=4,max_features=1.)
            identity['configuration'] = configuration
            save(folder/'evidence.json', identity)
            model = (PanelDynamics() if number == 72 else PanelStateForest()).fit(bundle.X.iloc[fit], bundle.y.iloc[fit])
            prediction, matched = model.predict_with_fallback(bundle.X.iloc[valid], base)
            mean, alpha = np.zeros(len(valid)), matched.astype(float)
            audit = {'matched_rows':int(matched.sum()), 'query_rows':len(valid)}
        else:
            model = PanelHistory(method=method).fit(bundle.X.iloc[fit], bundle.y.iloc[fit])
            mean, alpha, audit = model.lookup(bundle.X.iloc[valid])
            prediction = alpha * mean + (1-alpha) * base
        value, old = score('rmse', bundle.y.iloc[valid], prediction), score('rmse', bundle.y.iloc[valid], baseline)
        np.savez_compressed(folder / 'validation_predictions.npz', positions=valid, y=np.asarray(bundle.y.iloc[valid]),
                            prediction=prediction, control=baseline, base_prediction=base, memory_mean=mean, memory_weight=alpha)
        save(folder / 'audit.json', {**model.audit_, **audit})
        result = {'status': 'completed', 'metric': 'rmse', 'score': value, 'control': old,
                  'relative_gain': (old-value)/old, 'fit_rows': len(fit), 'valid_rows': len(valid),
                  'seconds': time.monotonic()-started, 'new_tree_model_fits': int(number in (72,74)),
                  'conclusion': 'Подтверждена в проверенных условиях' if value < old else 'Не подтверждена по заданному критерию'}
    except Exception as exc:
        result = {'status': 'failed', 'error': type(exc).__name__ + ': ' + str(exc),
                  'seconds': time.monotonic()-started, 'conclusion': 'Недостаточно данных'}
        (folder / 'traceback.txt').write_text(traceback.format_exc())
    save(folder / 'metrics.json', result)
    with (folder / 'report.md').open('a') as stream:
        stream.write('\n```json\n'+json.dumps(result, ensure_ascii=False, indent=2)+'\n```\n\nPublic data and splits previously examined; not a hidden-platform score. Negative results retained.\n')
    relative = folder.relative_to(ROOT).as_posix()
    with (ROOT / 'HYPOTHESES.md').open('a') as stream:
        stream.write(f'\n- SPEC011 {hypothesis}/{stage}: {json.dumps(result, ensure_ascii=False)}; [report]({relative}/report.md). Не повторять.\n')
    with (ROOT / f'docs/hypotheses/{hypothesis}.md').open('a') as stream:
        stream.write(f'\n- {stage}: {json.dumps(result, ensure_ascii=False)}; [report](../../{relative}/report.md).\n')
    with (ROOT / 'results/README.md').open('a') as stream:
        stream.write(f'\n| {hypothesis} | regression/{stage} | {result} | [report](../{relative}/report.md) |\n')
    print(json.dumps({'hypothesis': hypothesis, 'stage': stage, **result}, ensure_ascii=False), flush=True)
    if result['status'] != 'completed':
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--number', type=int, required=True)
    parser.add_argument('--stage', choices=['primary', 'secondary'], required=True)
    args = parser.parse_args()
    main(args.number, args.stage)
