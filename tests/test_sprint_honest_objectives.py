"""H92--94 read-only teacher reuse and unchanged-objective contracts."""
from contextlib import ExitStack
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from research_memory import data_digest


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ReuseTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('experiments.sprint_honest_reuse'), 'honest reuse provider absent')
        self.module = importlib.import_module('experiments.sprint_honest_reuse')
        self.temp = tempfile.TemporaryDirectory(dir=Path.cwd() / 'tmp')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = self.root / 'results/2026-10-24/HYP-088-01-inner-default-sprint'
        self.run.mkdir(parents=True)
        self.bundle = SimpleNamespace(X=pd.DataFrame({'g': np.repeat(np.arange(24), 2), 'proto': ['tcp'] * 48}),
            y=pd.Series(np.repeat(['a', 'b', 'c'] * 8, 2)), hashes={'train.csv': 'synthetic'})
        self.classes = np.array(['a', 'b', 'c'])
        self.outer_fit, self.outer_valid = np.arange(36), np.arange(36, 48)
        groups = pd.util.hash_pandas_object(self.bundle.X.iloc[self.outer_fit], index=False).to_numpy()
        local_fit, local_valid = list(StratifiedGroupKFold(3, shuffle=True, random_state=42).split(
            self.bundle.X.iloc[self.outer_fit], self.bundle.y.iloc[self.outer_fit], groups))[0]
        self.fit, self.valid = self.outer_fit[local_fit], self.outer_fit[local_valid]
        self.raw = {'classes': self.classes, 'directory': self.root / 'raw', 'audit': {'model_fits': 0},
                    'folds': [{'fit_positions': self.fit, 'valid_positions': self.valid,
                               'probabilities': np.tile([.1, .2, .7], (len(self.valid), 1))}]}
        self.raw['directory'].mkdir()
        np.savez(self.raw['directory'] / 'fold-0.npz', **self.raw['folds'][0])
        frozen = self.root / 'results/2026-10-24/series-three-hours-v1/HYP-088'
        self.source = frozen / 'source'
        self.source.mkdir(parents=True)
        names = ['automl_class_combo.py', 'automl_models.py', 'automl_combo_memory.py',
                 'sprint_contextual.py', 'sprint_honest.py', 'experiments/sprint_crossfit_cache.py']
        sources = {}
        for name in names:
            path = self.source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('frozen fixture ' + name)
            sources[name] = digest(path)
        (frozen / 'frozen.json').write_text(json.dumps({'sources': sources}))
        old_hashes = {Path(k).name: v for k, v in sources.items()}
        X, y = self.bundle.X.iloc[self.fit], np.asarray(self.bundle.y)[self.fit]
        groups = pd.util.hash_pandas_object(X, index=False).to_numpy()
        folds = list(StratifiedGroupKFold(3, shuffle=True, random_state=42).split(X, y, groups))
        directory = self.run / 'honest-raw-oof'
        directory.mkdir()
        self.expected = np.zeros((len(X), 3))
        for number, (f, v) in enumerate(folds):
            p = np.tile([.1 * (number + 1), .2, .8 - .1 * (number + 1)], (len(v), 1))
            self.expected[v] = p
            saved = dict(fit=f, valid=v, fit_positions=self.fit[f], valid_positions=self.fit[v],
                         classes=self.classes, probabilities=p)
            np.savez(directory / f'fold-{number}.npz', **saved)
            np.savez(directory / f'fold-{number}.splits.npz', **{k: saved[k] for k in ['fit', 'valid', 'fit_positions', 'valid_positions']})
            config = dict(hypothesis='HYP-088', role='nested_grouped_oof_raw_teacher', family='xgboost',
                component='raw', parameters={'n_estimators':600,'max_depth':6,'learning_rate':.05,'reg_lambda':1,
                    'tree_method':'hist','device':'cpu','objective':'multi:softprob'},
                encoder={'native':False,'scale':False},parameters_override={},threads=2,stage='inner',nested_fold=number,
                fit_positions=self.fit[f].tolist(),valid_positions=self.fit[v].tolist(),
                fit_positions_sha256=hashlib.sha256(np.asarray(self.fit[f],dtype='<i8').tobytes()).hexdigest(),
                valid_positions_sha256=hashlib.sha256(np.asarray(self.fit[v],dtype='<i8').tobytes()).hexdigest(),
                data_hashes=self.bundle.hashes,source_hashes=old_hashes,X_sha256=data_digest(X.iloc[f]),
                y_sha256=data_digest(y[f].reshape(-1,1)),split='SGKF3_seed42_raw_X_groups',raw_groups_disjoint=True)
            (directory / f'fold-{number}.started.json').write_text(json.dumps(dict(status='started',configuration=config,classes=self.classes.tolist())))
            (directory / f'fold-{number}.outcome.json').write_text(json.dumps(dict(status='completed',
                prediction_sha256=digest(directory / f'fold-{number}.npz'),fit_rows=len(f),valid_rows=len(v))))
        self.output = self.root / 'must-not-be-created'

    def load(self):
        with patch.object(self.module, '_original_context', return_value=(self.root,self.outer_fit,self.outer_valid)), \
             patch.object(self.module, '_read_component', return_value=self.raw):
            return self.module.load_reused_honest_teacher(self.bundle,'inner',self.fit,self.valid,
                root=self.root,output=self.output)

    def test_manual_oof_assembly_reuses_all_three_folds_without_writing_or_fitting(self):
        result = self.load()
        np.testing.assert_allclose(result['train_probabilities'], self.expected)
        np.testing.assert_allclose(result['valid_probabilities'], [.1,.2,.7] * np.ones((len(self.valid),3)))
        self.assertTrue(result['audit']['independent_training_oof'])
        self.assertEqual(result['audit']['model_fits'], 0)
        self.assertFalse(self.output.exists())

    def test_old_frozen_source_is_strict_but_current_unrelated_source_is_not_consulted(self):
        (self.root / 'automl_models.py').write_text('unrelated newer production hook')
        self.load()
        (self.source / 'automl_models.py').write_text('mutated historical snapshot')
        with self.assertRaises(ValueError):
            self.load()

    def test_missing_or_unfinished_or_byte_mutated_cache_fails_closed(self):
        directory = self.run / 'honest-raw-oof'
        (directory / 'fold-0.outcome.json').unlink()
        with self.assertRaises((ValueError,FileNotFoundError)):
            self.load()

    def test_completed_outcome_or_artifact_byte_mutation_is_rejected(self):
        directory = self.run / 'honest-raw-oof'
        path = directory / 'fold-0.outcome.json'
        outcome = json.loads(path.read_text())
        path.write_text(json.dumps({**outcome,'status':'started'}))
        with self.assertRaises(ValueError):self.load()
        path.write_text(json.dumps(outcome))
        with (directory / 'fold-0.npz').open('ab') as stream:stream.write(b'changed')
        with self.assertRaises(ValueError):self.load()

    def test_wrong_classes_or_training_digest_or_global_partition_rejected(self):
        path = self.run / 'honest-raw-oof/fold-0.started.json'
        original = json.loads(path.read_text())
        for mutation in ['classes', 'X_sha256', 'valid_positions']:
            changed = json.loads(json.dumps(original))
            if mutation == 'classes': changed['classes'] = ['b','a','c']
            elif mutation == 'X_sha256': changed['configuration'][mutation] = 'bad'
            else: changed['configuration'][mutation] = [999]
            path.write_text(json.dumps(changed))
            with self.assertRaises(ValueError): self.load()
        path.write_text(json.dumps(original))
        self.bundle.X.iloc[self.valid[0]] = self.bundle.X.iloc[self.fit[0]]
        with self.assertRaises(ValueError): self.load()


class ObjectiveWrapperTests(unittest.TestCase):
    def test_preregistered_long_variants_report_actual_rounds_and_epochs(self):
        from sprint_honest_objectives import HonestMacroTeacherBoost, HonestResidualF1Classifier
        self.assertEqual(HonestMacroTeacherBoost(iterations=900).configuration()['parameters']['n_estimators'],900)
        self.assertEqual(HonestResidualF1Classifier(epochs=150).configuration['epochs'],150)

    def test_wrappers_keep_original_objective_and_parameters_and_configuration_shape(self):
        self.assertIsNotNone(importlib.util.find_spec('sprint_honest_objectives'), 'honest objective wrappers absent')
        from sprint_honest_objectives import HonestMacroTeacherBoost, HonestLovaszTeacherBoost, HonestResidualF1Classifier
        from sprint_softboost import MacroTeacherBoost
        from sprint_ambitious import LovaszTeacherBoost
        from sprint_classification import TeacherResidualF1Classifier
        for cls, old, number in [(HonestMacroTeacherBoost,MacroTeacherBoost,92),(HonestLovaszTeacherBoost,LovaszTeacherBoost,93)]:
            model, previous = cls(), old()
            self.assertEqual(model.parameters,previous.parameters)
            self.assertEqual(model.configuration()['hypothesis'],f'HYP-{number:03d}')
            self.assertEqual(cls._objective,old._objective)
        model, previous = HonestResidualF1Classifier(), TeacherResidualF1Classifier()
        self.assertIsInstance(model.configuration,dict)
        self.assertEqual(model.configuration['epochs'],previous.configuration['epochs'])
        self.assertEqual(model.configuration['objective'],previous.configuration['objective'])
        self.assertEqual(model.configuration['hypothesis'],'HYP-094')


if __name__ == '__main__':unittest.main()
