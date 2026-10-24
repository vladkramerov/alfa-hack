"""Read-only H70/H78 adapter replay; saved models/probabilities only, zero fits.

Writes one new evidence directory. This checks the measured outer graph and
frozen decision transfer, not the predictions of a future full-data refit.
"""
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import sys
import traceback
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
import pandas as pd

from automl_class_combo import ClassCombination, ClassBranch, predict_meta_state
from automl_data import load_task
from automl_reg_combo import FeatureGroupExperts, RegressionComponent
from automl_sprint import SprintDecision, SprintRegression
from automl_temporal_memory import ProfileHistory, TemporalProfileModel
from experiments.sprint_worker import SavedBaseline  # Legacy pickle __main__ lookup.
from sprint_decision import MacroF1Decision
from sprint_posterior import CalibratedMacroDecision
from sprint_regression import TemporalEntityLinker

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        for part in iter(lambda: source.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def arrays(path, fields):
    with np.load(path, allow_pickle=False) as saved:
        return {field: saved[field].copy() for field in fields}


class CachedTeacher:
    def __init__(self, probability):
        self.probability = probability

    def predict_proba(self, X):
        return self.probability[X['row_position'].to_numpy()]


class CachedProfileOutputs:
    """Replay saved H69 memory inference, without constructing/fitting history."""
    def __init__(self, saved):
        self.lookup_ = {int(p): (float(m), float(w)) for p, m, w in
                        zip(saved['positions'], saved['memory_mean'], saved['memory_weight'])}

    def lookup(self, X):
        value = np.asarray([self.lookup_[int(p)] for p in X.index])
        return value[:, 0], value[:, 1], {'source': 'saved_H69_memory_outputs_no_fit'}


def verify(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    result = {'status': 'started', 'model_fits': 0, 'checks': {},
              'limitation': 'Cached outer graph replay; future full-data refit determinism is not established.'}
    fits = []
    def forbidden_fit(*args, **kwargs):
        fits.append(type(args[0]).__name__ if args else 'unknown')
        raise AssertionError('Verification must not fit any model')
    try:
        sources = ['experiments/verify_sprint_adapter.py', 'automl_sprint.py', 'automl_models.py',
                   'automl_history.py', 'automl_data.py', 'automl_class_combo.py', 'automl_reg_combo.py',
                   'automl_temporal_memory.py', 'sprint_regression.py', 'sprint_posterior.py', 'sprint_decision.py']
        result['source_sha256'] = {name: sha(ROOT / name) for name in sources}
        h70 = ROOT / 'results/2026-10-24/HYP-070-04-outer-high-sprint'
        h69 = ROOT / 'results/2026-10-24/HYP-069-01-regression-alternatives'
        h43 = ROOT / 'results/2026-10-23/HYP-043-01-regression-combinations'
        h78 = ROOT / 'results/2026-10-24/HYP-078-02-outer-default-sprint'
        h48 = ROOT / 'results/2026-10-23/HYP-048-01-multiclass-combinations'
        artifacts = [h70 / 'model.joblib', h70 / 'audit.json', h70 / 'validation_predictions.npz',
                     h69 / 'validation_predictions.npz', h69 / 'profile-memory-components.npz',
                     h43 / 'validation_predictions.npz', h78 / 'audit.json', h78 / 'validation_predictions.npz',
                     h48 / 'learned-state.json', h48 / 'validation_predictions.npz']
        result['data_sha256'] = {}
        bundles = {}
        for task, run, historical in [('regression', h70, h69), ('multiclass', h78, h48)]:
            bundles[task] = load_task(ROOT / 'data' / task / 'task.md')
            expected = read(run / 'data.json')['hashes']
            historical_hashes = read(historical / 'evidence.json')['hashes']
            observed = {name: sha(ROOT / 'data' / task / name) for name in expected}
            assert observed == expected == historical_hashes, task + ' data SHA mismatch'
            assert bundles[task].hashes == expected
            result['data_sha256'][task] = observed
        from catboost import CatBoostRegressor, CatBoostClassifier
        from lightgbm import LGBMClassifier
        from sklearn.ensemble import ExtraTreesRegressor
        from xgboost import XGBClassifier
        guarded = [SprintRegression, SprintDecision, ClassCombination, ClassBranch,
                   FeatureGroupExperts, RegressionComponent, TemporalProfileModel, ProfileHistory,
                   TemporalEntityLinker, MacroF1Decision, CalibratedMacroDecision,
                   CatBoostRegressor, CatBoostClassifier, LGBMClassifier, ExtraTreesRegressor, XGBClassifier]
        with ExitStack() as stack:
            for cls in guarded:
                stack.enter_context(patch.object(cls, 'fit', forbidden_fit))
            result['fit_guard_classes'] = [cls.__module__ + '.' + cls.__name__ for cls in guarded]
            s78 = arrays(h78 / 'validation_predictions.npz', ['positions', 'probabilities', 'prediction', 'classes'])
            split78 = arrays(h78 / 'splits.npz', ['fit', 'valid'])
            split48 = arrays(h48 / 'splits.npz', ['fit', 'valid'])
            assert all(np.array_equal(split78[k], split48[k]) for k in ('fit', 'valid'))
            assert np.array_equal(split78['valid'], s78['positions'])
            assert np.array_equal(np.unique(bundles['multiclass'].y.iloc[split78['fit']]).astype(str), s78['classes'])
            meta = read(h48 / 'learned-state.json')['meta_state']
            parts = []
            for name in ['HYP-011-02-multiclass-none', 'HYP-038-01-multiclass-spec006', 'HYP-017-01-multiclass-frequency']:
                path = ROOT / 'results/2026-10-23' / name / 'validation_predictions.npz'
                artifacts.append(path)
                part = arrays(path, ['positions', 'classes', 'probabilities'])
                assert np.array_equal(part['positions'], s78['positions'])
                assert np.array_equal(part['classes'].astype(str), s78['classes'])
                p = np.asarray(part['probabilities'], float)
                parts.append(p / p.sum(axis=1, keepdims=True))
            teacher = predict_meta_state(parts, meta, s78['classes'])
            saved48 = arrays(h48 / 'validation_predictions.npz', ['positions', 'probabilities'])
            assert np.array_equal(saved48['positions'], s78['positions'])
            assert np.allclose(teacher, saved48['probabilities'], atol=1e-12, rtol=0)
            audit78 = read(h78 / 'audit.json')
            state = dict(classes=s78['classes'].tolist(), gain=audit78['decision']['gain'],
                         offset=audit78['decision']['offset'], theta=audit78['theta'])
            model = SprintDecision(dict(strategy='proper_macro_f1_stacking', family='xgboost',
                                        parameters={}, meta_state=meta, decision_state=state))
            model.base_ = CachedTeacher(teacher)
            model.columns_ = ['row_position']
            X = pd.DataFrame({'row_position': np.arange(len(teacher))})
            actual = model.predict_proba(X)
            assert np.allclose(actual, s78['probabilities'], atol=1e-12, rtol=0)
            assert np.array_equal(model.predict(X), s78['prediction'])
            result['checks']['H78'] = {'rows': len(X), 'classes_exact': True, 'all_labels_exact': True,
                'max_probability_delta': float(np.max(np.abs(actual - s78['probabilities']))),
                'probability_tolerance_absolute': 1e-12, 'teacher_rebuilt_from_three_cached_branches': True,
                'split_equal_to_original_H48': True, 'frozen_state': state}
            s70 = arrays(h70 / 'validation_predictions.npz', ['positions', 'prediction'])
            s69 = arrays(h69 / 'validation_predictions.npz', ['positions', 'prediction'])
            split70 = arrays(h70 / 'splits.npz', ['fit', 'valid'])
            split69 = arrays(h69 / 'splits.npz', ['fit', 'valid'])
            assert all(np.array_equal(split70[k], split69[k]) for k in ('fit', 'valid'))
            assert np.array_equal(split70['valid'], s70['positions'])
            assert np.array_equal(s70['positions'], s69['positions'])
            correction = joblib.load(h70 / 'model.joblib')
            assert correction.alpha == .95 and correction.audit_ == read(h70 / 'audit.json')
            query = bundles['regression'].X.iloc[s70['positions']]
            assert np.array_equal(correction.base_predictor_.predict(query), s69['prediction'])
            reg = SprintRegression(dict(strategy='entity_linked_history', family='extra_trees', parameters={}))
            reg.columns_, reg.correction_ = list(query.columns), correction
            actual70 = reg.predict(query)
            assert np.array_equal(actual70, s70['prediction'])
            assert correction.configuration() == reg.configuration()['correction']
            result['checks']['H70'] = {'rows': len(query), 'bit_exact': True, 'max_prediction_delta': 0.,
                'alpha': correction.alpha, 'configuration_exact': True, 'saved_base_is_H69_exact': True,
                'split_equal_to_original_H69': True}
            memory = arrays(h69 / 'profile-memory-components.npz', ['positions', 'memory_mean', 'memory_weight', 'baseline'])
            s43 = arrays(h43 / 'validation_predictions.npz', ['positions', 'prediction'])
            assert np.array_equal(memory['positions'], s43['positions'])
            assert np.array_equal(memory['baseline'], s43['prediction'])
            composed = memory['memory_weight'] * memory['memory_mean'] + (1 - memory['memory_weight']) * memory['baseline']
            assert np.array_equal(composed, s69['prediction'])
            model_path = Path(read(h43 / 'audit.json')['model_cache']['artifact'])
            artifacts.append(model_path)
            base43 = joblib.load(model_path)
            assert isinstance(base43, FeatureGroupExperts)
            assert base43.candidate['feature_numbers'] == [29, 30, 31]
            assert base43.candidate.get('parameters', {}) == {}
            assert base43.audit_['fit_retained_rows'] == len(split70['fit'])
            # Limit inference parallelism; this does not fit or change saved files.
            for component in [base43.global_, *base43.experts_.values()]:
                component.model_.n_jobs = 2
            base = TemporalProfileModel()
            base.base_, base.memory_ = base43, CachedProfileOutputs(memory)
            actual69 = base.predict(query)
            delta69 = float(np.max(np.abs(actual69 - s69['prediction'])))
            assert np.allclose(actual69, s69['prediction'], atol=1e-10, rtol=1e-12)
            result['checks']['H69'] = {'rows': len(query), 'cached_H43_baseline_bit_exact': True,
                'cached_memory_composition_bit_exact': True, 'saved_H43_model_inference_max_delta': delta69,
                'production_base_configuration_matches_H43': True,
                'memory_lookup_replayed_from_cached_outputs_not_refitted': True}
        assert not fits
        result['fit_attempts'] = fits
        result['artifact_sha256'] = {str(p.relative_to(ROOT)): sha(p) for p in artifacts}
        result['status'] = 'passed'
    except Exception as exc:
        result.update(status='failed', error=type(exc).__name__ + ': ' + str(exc),
                      traceback=traceback.format_exc(), fit_attempts=fits)
    (output / 'verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    status = 'PASS' if result['status'] == 'passed' else 'FAIL'
    report = ('# Sprint adapter verification: ' + status + '\n\n'
        'Команда: `.venv-ml/Scripts/python.exe -X utf8 -m experiments.verify_sprint_adapter`.\n\n'
        'Проверены исходные SHA данных, source SHA и exact splits; все fit защищены исключением. '
        'H78 воспроизводится из трёх сохранённых ветвей/H48 meta и frozen H78 theta/gain/offset. '
        'H70 воспроизводится из сохранённого linker alpha.95 и H69; H69 проверен через cached memory '
        'и существующий H43 model inference. Численные итоги и ошибки: [verification.json](verification.json).\n\n'
        'Это replay уже измеренного outer-графа; детерминизм и качество будущего full-data refit '
        'не установлены. Новых гипотез/модельных fit/научных метрик нет.\n')
    if result['status'] == 'failed':
        report += '\nОшибка: `' + result['error'] + '`; незавершённые проверки не объявляются пройденными.\n'
    (output / 'report.md').write_text(report, encoding='utf-8')
    print(json.dumps({'status': result['status'], 'model_fits': 0, 'checks': result['checks'],
                      'error': result.get('error')}, ensure_ascii=False))
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    destination = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'results/2026-10-24/sprint-adapter-verification'
    sys.exit(verify(destination))
