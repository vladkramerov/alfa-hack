"""Data boundary tests: names and lexical values must not leak assumptions."""
import importlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd


class ResearchDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (Path.cwd() / 'tmp').mkdir(exist_ok=True)

    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('research_data'), 'research_data is not implemented')
        return importlib.import_module('research_data')

    def fixture(self, root, classification=False, declared=False):
        root = Path(root)
        nested = root / 'nested' / 'inputs'
        nested.mkdir(parents=True)
        labels = ['001', '+1', 'NA'] * 10 if classification else list(np.arange(30) * 0.5)
        pd.DataFrame({'opaque': [f'{i:03}' for i in range(30)], 'signal': np.arange(30), 'result': labels}).to_csv(nested / ('learning.csv' if declared else 'train.csv'), index=False)
        (nested / ('future.csv' if declared else 'test.csv')).write_text('opaque,signal\n001,3\n+1,4\nNA,5\n', encoding='utf-8')
        (nested / ('answer.csv' if declared else 'sample_submission.csv')).write_text('opaque,result\n001,0\n+1,0\nNA,0\n', encoding='utf-8')
        (nested / 'data_dictionary.csv').write_text('feature,role\nopaque,id\nresult,target\n', encoding='utf-8')
        text = 'Classification.\nPrimary metric: Macro F1.\n' if classification else 'Regression.\nPrimary metric: MAE. Reference RMSE is not primary.\n'
        if declared:
            text += 'Train: `nested/inputs/learning.csv`\nTest: `nested/inputs/future.csv`\nSample submission: `nested/inputs/answer.csv`\nTarget column: `result`\nID column: `opaque`\n'
        (root / 'task.md').write_text(text, encoding='utf-8')
        return root / 'task.md'

    def test_nested_renamed_paths_target_and_primary_metric(self):
        api = self.api()
        with tempfile.TemporaryDirectory(dir=Path.cwd() / 'tmp') as root:
            task = api.load_task(self.fixture(root, declared=True))
            self.assertEqual((task.target_column, task.id_column, task.metric, task.task_type), ('result', 'opaque', 'mae', 'regression'))
            self.assertEqual(task.X.columns.tolist(), ['signal'])
            self.assertTrue(pd.api.types.is_numeric_dtype(task.y))
            self.assertEqual(set(task.hashes), {'train', 'test', 'sample', 'dictionary', 'task'})
            self.assertTrue(all(len(v) == 64 for v in task.hashes.values()))

    def test_lexical_identifiers_and_class_labels_survive_load_and_csv(self):
        api = self.api()
        with tempfile.TemporaryDirectory(dir=Path.cwd() / 'tmp') as root:
            task = api.load_task(self.fixture(root, True))
            self.assertEqual(task.y.iloc[:3].tolist(), ['001', '+1', 'NA'])
            self.assertEqual(task.sample['opaque'].tolist(), ['001', '+1', 'NA'])
            output = api.save_submission(task, np.array(['NA', '001', '+1']))
            actual = pd.read_csv(output, dtype=str, keep_default_na=False)
            self.assertEqual(actual.values.tolist(), [['001', 'NA'], ['+1', '001'], ['NA', '+1']])

    def test_ambiguous_discovery_and_missing_explicit_path_are_rejected(self):
        api = self.api()
        with tempfile.TemporaryDirectory(dir=Path.cwd() / 'tmp') as root:
            path = self.fixture(root)
            other = Path(root) / 'other'
            other.mkdir()
            (other / 'train.csv').write_text('x,y\n1,2\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'ambiguous'):
                api.load_task(path)
        with tempfile.TemporaryDirectory(dir=Path.cwd() / 'tmp') as root:
            path = self.fixture(root, declared=True)
            path.write_text(path.read_text().replace('learning.csv', 'missing.csv'))
            with self.assertRaises((ValueError, FileNotFoundError)):
                api.load_task(path)

    def test_sample_order_and_ambiguous_target_are_rejected(self):
        api = self.api()
        with tempfile.TemporaryDirectory(dir=Path.cwd() / 'tmp') as root:
            path = self.fixture(root)
            sample = path.parent / 'nested/inputs/sample_submission.csv'
            sample.write_text('opaque,result\n+1,0\n001,0\nNA,0\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'order|align'):
                api.load_task(path)
        with tempfile.TemporaryDirectory(dir=Path.cwd() / 'tmp') as root:
            path = self.fixture(root)
            train = path.parent / 'nested/inputs/train.csv'
            frame = pd.read_csv(train)
            frame['other_target'] = 1
            frame.to_csv(train, index=False)
            # Sample uniquely identifies result; another train-only column remains ambiguous leakage.
            with self.assertRaisesRegex(ValueError, 'feature|schema|extra|target'):
                api.load_task(path)

    def test_unknown_metric_rejected_and_supported_directions_scores(self):
        api = self.api()
        for metric in ['rmse', 'mse', 'mae', 'rmsle', 'log_loss']:
            self.assertTrue(api.better(metric, 0.1, 0.2))
            self.assertFalse(api.better(metric, 0.2, 0.2))
        for metric in ['r2', 'macro_f1', 'weighted_f1', 'accuracy', 'balanced_accuracy', 'roc_auc']:
            self.assertTrue(api.better(metric, 0.2, 0.1))
        self.assertAlmostEqual(api.score('rmse', [1, 3], [2, 3]), 2 ** -0.5)
        self.assertAlmostEqual(api.score('mae', [1, 3], [2, 3]), 0.5)
        self.assertAlmostEqual(api.score('mse', [1, 3], [2, 3]), 0.5)
        self.assertAlmostEqual(api.score('macro_f1', ['001', 'NA'], ['001', 'NA']), 1.)
        self.assertAlmostEqual(api.score('roc_auc', ['001', 'NA'], [.1, .8]), 1.)
        self.assertAlmostEqual(api.score('log_loss', ['001', 'NA'], [[.9, .1], [.1, .9]]), -np.log(.9))
        with self.assertRaises(ValueError):
            api.better('custom_metric', 1, 2)
        with tempfile.TemporaryDirectory(dir=Path.cwd() / 'tmp') as root:
            path = self.fixture(root)
            path.write_text('Regression. Primary metric: mystery. RMSE is a reference only.')
            with self.assertRaisesRegex(ValueError, 'metric'):
                api.load_task(path)

    def test_temporal_split_uses_generic_future_period_and_no_overlap(self):
        api = self.api()
        with tempfile.TemporaryDirectory(dir=Path.cwd() / 'tmp') as root:
            task = api.load_task(self.fixture(root))
            task.X['observation_time'] = np.repeat([10, 20, 30], 10)
            task.X_test['observation_time'] = [40, 40, 40]
            fit, valid, description = api.split_data(task)
            self.assertEqual(len(fit), 20)
            self.assertEqual(len(valid), 10)
            self.assertLess(task.X.iloc[fit]['observation_time'].max(), task.X.iloc[valid]['observation_time'].min())
            self.assertEqual(description['kind'], 'temporal')
            json.dumps(description, allow_nan=False)

    def test_duplicate_classification_groups_are_disjoint_and_folds_fixed(self):
        api = self.api()
        with tempfile.TemporaryDirectory(dir=Path.cwd() / 'tmp') as root:
            task = api.load_task(self.fixture(root, True))
            task.X = pd.DataFrame({'raw': np.repeat(np.arange(15), 2)})
            task.y = pd.Series(np.repeat(['001', '+1', 'NA'] * 5, 2))
            fit, valid, description = api.split_data(task)
            self.assertTrue(set(task.X.iloc[fit]['raw']).isdisjoint(task.X.iloc[valid]['raw']))
            self.assertEqual(description['kind'], 'stratified_group')
            self.assertEqual(description['n_splits'], 5)
            np.testing.assert_array_equal(valid, api.split_data(task)[1])

    def test_rare_class_group_has_clear_error_instead_of_row_leakage(self):
        api = self.api()
        with tempfile.TemporaryDirectory(dir=Path.cwd() / 'tmp') as root:
            task = api.load_task(self.fixture(root, True))
            task.X = pd.DataFrame({'raw': [1, 1, 2, 2]})
            task.y = pd.Series(['a', 'a', 'b', 'b'])
            with self.assertRaisesRegex(ValueError, 'group|class'):
                api.split_data(task)

    def test_exact_lexical_duplicate_groups_and_small_group_split(self):
        api = self.api()
        with tempfile.TemporaryDirectory(dir=Path.cwd() / 'tmp') as root:
            path = self.fixture(root, True)
            representations = ['001', '+1', '1.0', '01', '01.0', '1e0']
            rows = []
            for context, label in [(2, '001'), (3, 'NA')]:
                for lexical in representations:
                    for _ in range(2):
                        rows.append([str(len(rows)), lexical, context, label])
            pd.DataFrame(rows, columns=['opaque', 'signal', 'context', 'result']).to_csv(path.parent / 'nested/inputs/train.csv', index=False)
            (path.parent / 'nested/inputs/test.csv').write_text('opaque,signal,context\n001,2,2\n+1,3,2\nNA,4,3\n')
            task = api.load_task(path)
            fit, valid, description = api.split_data(task)
            self.assertEqual(description['groups'], 12)
            self.assertEqual(set(task.y.iloc[fit]), {'001', 'NA'})
            self.assertEqual(set(task.y.iloc[valid]), {'001', 'NA'})
            task.raw_X = None
            task.X = pd.DataFrame({'raw': [1, 1, 2, 2, 3, 3, 4, 4]})
            task.y = pd.Series(['a', 'a', 'a', 'a', 'b', 'b', 'b', 'b'])
            fit, valid, description = api.split_data(task)
            self.assertEqual(description['n_splits'], 2)
            self.assertTrue(set(task.X.iloc[fit]['raw']).isdisjoint(task.X.iloc[valid]['raw']))

    def test_train_only_feature_type_and_missing_lexical_category(self):
        api = self.api()
        with tempfile.TemporaryDirectory(dir=Path.cwd() / 'tmp') as root:
            path = self.fixture(root)
            train_path = path.parent / 'nested/inputs/train.csv'
            frame = pd.read_csv(train_path, dtype={'opaque': str})
            frame['category'] = ['NA', '', 'known'] * 10
            frame['empty'] = ''
            frame.to_csv(train_path, index=False)
            (path.parent / 'nested/inputs/test.csv').write_text('opaque,signal,category,empty\n001,broken,NA,1\n+1,4,new,2\nNA,5,,3\n')
            task = api.load_task(path)
            self.assertEqual(task.X['category'].iloc[0], 'NA')
            self.assertTrue(pd.isna(task.X['category'].iloc[1]))
            self.assertTrue(pd.isna(task.X_test['signal'].iloc[0]))
            self.assertFalse(pd.api.types.is_numeric_dtype(task.X_test['empty']))

    def test_submission_rejects_shape_unknown_classes_nonfinite_and_reordering(self):
        api = self.api()
        with tempfile.TemporaryDirectory(dir=Path.cwd() / 'tmp') as root:
            task = api.load_task(self.fixture(root, True))
            for bad in [['NA'], ['NA', 'unknown', '001'], [['NA'], ['001'], ['+1']]]:
                with self.assertRaises(ValueError):
                    api.save_submission(task, bad)
            reversed_frame = pd.DataFrame({'opaque': ['NA', '+1', '001'], 'result': ['NA'] * 3})
            with self.assertRaisesRegex(ValueError, 'order|identifier'):
                api.save_submission(task, reversed_frame)
            self.assertFalse((Path(root) / 'submission.csv').exists())
        with tempfile.TemporaryDirectory(dir=Path.cwd() / 'tmp') as root:
            task = api.load_task(self.fixture(root))
            for bad in [[0, np.nan, 1], [0, np.inf, 1]]:
                with self.assertRaises(ValueError):
                    api.save_submission(task, bad)
            out = api.save_submission(task, [1.1, 2.2, 3.3])
            self.assertTrue(Path(out).is_file())
            original = Path(out).read_bytes()
            with self.assertRaises(ValueError):
                api.save_submission(task, [1., np.inf, 2.])
            self.assertEqual(Path(out).read_bytes(), original)
            self.assertEqual(list(Path(root).glob('*.tmp')), [])

    def test_diagnostics_are_bounded_json_with_handchecked_errors(self):
        api = self.api()
        classified = api.diagnostics('classification', ['a', 'a', 'b'], ['a', 'b', 'b'])
        self.assertEqual(classified['confusion_pairs'][0], {'actual': 'a', 'predicted': 'b', 'count': 1})
        self.assertAlmostEqual(classified['per_class'][0]['recall'], .5)
        regression = api.diagnostics('regression', [1., 3.], [2., 3.])
        self.assertAlmostEqual(regression['rmse'], 2 ** -.5)
        self.assertAlmostEqual(regression['bias'], .5)
        many = [str(i) for i in range(200)]
        bounded = api.diagnostics('classification', many, many[::-1])
        self.assertLessEqual(len(bounded['per_class']), 30)
        self.assertLessEqual(len(bounded['confusion_pairs']), 20)
        json.dumps([classified, regression, bounded], allow_nan=False)

    def test_binary_probability_csv_uses_primary_objective(self):
        api = self.api()
        with tempfile.TemporaryDirectory(dir=Path.cwd() / 'tmp') as root:
            path = self.fixture(root, True)
            train = path.parent / 'nested/inputs/train.csv'
            frame = pd.read_csv(train, dtype=str, keep_default_na=False)
            frame['result'] = ['001', 'NA'] * 15
            frame.to_csv(train, index=False)
            path.write_text('Classification. Primary metric: ROC AUC.')
            task = api.load_task(path)
            self.assertEqual(task.probability_columns, ['result'])
            self.assertEqual(task.class_order, ['001', 'NA'])
            output = api.save_submission(task, [.1, .9, .5])
            actual = pd.read_csv(output, dtype={'opaque': str}, keep_default_na=False)
            self.assertEqual(actual['result'].tolist(), [.1, .9, .5])
            for bad in [[-.1, .2, .3], [0., 1.1, .3], [0., np.nan, .3]]:
                with self.assertRaises(ValueError):
                    api.save_submission(task, bad)

    def test_multiclass_probability_order_and_sums_match_sample(self):
        api = self.api()
        with tempfile.TemporaryDirectory(dir=Path.cwd() / 'tmp') as root:
            path = self.fixture(root, True)
            path.write_text('Classification. Primary metric: Log Loss.')
            sample = path.parent / 'nested/inputs/sample_submission.csv'
            sample.write_text('opaque,NA,001,+1\n001,0,0,0\n+1,0,0,0\nNA,0,0,0\n')
            task = api.load_task(path)
            self.assertEqual(task.target_column, 'result')
            self.assertEqual(task.probability_columns, ['NA', '001', '+1'])
            self.assertEqual(task.class_order, ['+1', '001', 'NA'])
            predictions = np.array([[.1, .7, .2], [.2, .1, .7], [.8, .1, .1]])
            output = api.save_submission(task, predictions)
            actual = pd.read_csv(output, dtype={'opaque': str}, keep_default_na=False)
            self.assertEqual(actual.columns.tolist(), ['opaque', 'NA', '001', '+1'])
            np.testing.assert_allclose(actual.iloc[:, 1:], predictions)
            for bad in [predictions[:, :2], predictions * .5, np.array([[1.1, -.1, 0.]] * 3)]:
                with self.assertRaises(ValueError):
                    api.save_submission(task, bad)
            sample.write_text('opaque,NA,001,unknown\n001,0,0,0\n+1,0,0,0\nNA,0,0,0\n')
            with self.assertRaisesRegex(ValueError, 'class|probability'):
                api.load_task(path)

    def test_generated_research_runs_are_not_input_candidates(self):
        api = self.api()
        with tempfile.TemporaryDirectory(dir=Path.cwd() / 'tmp') as root:
            path = self.fixture(root)
            output = Path(root) / 'research_runs' / 'one'
            output.mkdir(parents=True)
            (output / 'train.csv').write_text('a,b\n1,2\n')
            task = api.load_task(path)
            self.assertEqual(len(task.y), 30)


if __name__ == '__main__':
    unittest.main()
