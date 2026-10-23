import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from automl_data import (
    TaskData, atomic_submission, fallback_submission, infer_metric, load_task,
    make_submission, resolve_paths, score, split_key, validate_submission,
    validation_split,
)


class TaskLoadingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / 'nested').mkdir()

    def tearDown(self):
        self.temp.cleanup()

    def write_csv(self, name, frame):
        path = self.root / 'nested' / name
        frame.to_csv(path, index=False)
        return path

    def test_role_links_dictionary_schema_and_sample_order(self):
        train = pd.DataFrame({'record_key': [1, 2, 3, 4], 'signal': [2, 3, 4, 5], 'response': ['a', 'b', 'a', 'b']})
        test = pd.DataFrame({'record_key': [8, 7], 'signal': [9, 8]})
        sample = pd.DataFrame({'record_key': [7, 8], 'response': ['', '']})
        dictionary = pd.DataFrame({'column': ['record_key', 'response'], 'role': ['identifier', 'target']})
        paths = [self.write_csv('observations.csv', train), self.write_csv('future.csv', test),
                 self.write_csv('answer_format.csv', sample), self.write_csv('schema.csv', dictionary)]
        task = self.root / 'task.md'
        task.write_text(
            f'Training data: [data]({paths[0].relative_to(self.root)})\n'
            f'Test data: [data]({paths[1].relative_to(self.root)})\n'
            f'Sample submission: [format]({paths[2].relative_to(self.root)})\n'
            f'Data dictionary: [schema]({paths[3].relative_to(self.root)})\n'
            'Target: response\nPositive class: b\nPrimary metric: macro F1\n', encoding='utf-8')
        bundle = load_task(task)
        self.assertEqual(bundle.target, 'response')
        self.assertEqual(bundle.id_column, 'record_key')
        self.assertEqual(bundle.classes, ('a', 'b'))
        self.assertEqual(bundle.positive_class, 'b')
        self.assertEqual(bundle.X_test['signal'].tolist(), [8, 9])
        submission = make_submission(bundle, np.array(['a', 'b']))
        self.assertEqual(submission['record_key'].tolist(), [7, 8])

    def test_arbitrary_filenames_and_optional_dictionary_fallback(self):
        self.write_csv('learn.csv', pd.DataFrame({'x': [1, 2, 3], 'y': [2, 4, 6]}))
        self.write_csv('holdout.csv', pd.DataFrame({'x': [4]}))
        self.write_csv('format.csv', pd.DataFrame({'y': [0]}))
        text = 'train: nested/learn.csv\ntest: nested/holdout.csv\nsample submission: nested/format.csv\nPrimary metric: MAE'
        paths = resolve_paths(self.root / 'task.md', text)
        self.assertIsNone(paths['dictionary'])
        task = self.root / 'task.md'
        task.write_text(text, encoding='utf-8')
        bundle = load_task(task)
        self.assertEqual(bundle.target, 'y')
        self.assertEqual(bundle.metric, 'mae')
        self.assertEqual(bundle.task_type, 'regression')

    def test_unknown_and_ambiguous_primary_metric_fail(self):
        with self.assertRaisesRegex(ValueError, 'Unsupported explicit metric'):
            infer_metric('Primary metric: SMAPE')
        with self.assertRaisesRegex(ValueError, 'Ambiguous metric'):
            infer_metric('Primary metric: RMSE and MAE')

    def test_declared_id_missing_from_sample_fails_clearly(self):
        train = self.write_csv('train.csv', pd.DataFrame({'id': [1, 2], 'x': [0, 1], 'target': [2, 3]}))
        test = self.write_csv('test.csv', pd.DataFrame({'id': [3], 'x': [2]}))
        sample = self.write_csv('sample_submission.csv', pd.DataFrame({'target': [0]}))
        dictionary = self.write_csv('data_dictionary.csv', pd.DataFrame({'feature': ['id', 'target'], 'role': ['identifier', 'target']}))
        task = self.root / 'task.md'
        task.write_text(f'train: {train.relative_to(self.root)}\ntest: {test.relative_to(self.root)}\nsample submission: {sample.relative_to(self.root)}\ndata dictionary: {dictionary.relative_to(self.root)}\nPrimary metric: MAE', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Sample submission is missing the declared ID'):
            load_task(task)

    def test_missing_required_role_fails(self):
        (self.root / 'one.csv').write_text('a\n1\n')
        with self.assertRaisesRegex(ValueError, 'required CSV roles'):
            resolve_paths(self.root / 'task.md', 'Primary metric: MAE')


class SplitTests(unittest.TestCase):
    def bundle(self, X, y, task_type='classification', test=None, text=''):
        train = X.copy()
        train['target'] = y
        test = test if test is not None else X.iloc[:2].copy()
        sample = pd.DataFrame({'target': [0] * len(test)})
        return TaskData(Path('/tmp/task.md'), text, {}, train, test, sample, pd.DataFrame(),
                        'target', None, task_type, 'macro_f1' if task_type == 'classification' else 'rmse',
                        tuple(sorted(pd.unique(y))) if task_type == 'classification' else None)

    def test_duplicate_rows_never_cross_split_for_classification_or_regression(self):
        X = pd.DataFrame({'signal': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]})
        for kind, y in [('classification', np.array(['a', 'a', 'b', 'b', 'a', 'a', 'b', 'b', 'a', 'a'])),
                        ('regression', np.arange(10, dtype=float))]:
            bundle = self.bundle(X, y, kind)
            fit, valid, description = validation_split(bundle)
            fit_rows = set(map(tuple, bundle.X.iloc[fit].to_numpy()))
            valid_rows = set(map(tuple, bundle.X.iloc[valid].to_numpy()))
            self.assertTrue(fit_rows.isdisjoint(valid_rows), description)
            self.assertEqual(len(set(fit) | set(valid)), len(y))

    def test_date_split_accepts_high_cardinality_dates_when_test_is_later(self):
        dates = pd.date_range('2020-01-01', periods=120, freq='D').astype(str)
        X = pd.DataFrame({'event_date': dates, 'signal': range(120)})
        test = pd.DataFrame({'event_date': ['2020-05-01'], 'signal': [121]})
        bundle = self.bundle(X, np.arange(120, dtype=float), 'regression', test)
        fit, valid, desc = validation_split(bundle)
        self.assertIn('temporal:event_date', desc)
        self.assertEqual(len(valid), 1)
        self.assertEqual(X.iloc[valid[0]]['event_date'], '2020-04-29')

    def test_classification_can_use_temporal_split_without_crossing_groups(self):
        X = pd.DataFrame({'period': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5], 'x': range(10)})
        test = pd.DataFrame({'period': [6, 6], 'x': [10, 11]})
        y = np.array(['a', 'b'] * 5)
        bundle = self.bundle(X, y, 'classification', test)
        fit, valid, desc = validation_split(bundle)
        self.assertIn('temporal:period', desc)
        self.assertEqual(set(X.iloc[valid]['period']), {5})
        self.assertEqual(set(X.iloc[fit]['period']), {1, 2, 3, 4})

    def test_public_temporal_split_requires_later_test_period(self):
        X = pd.DataFrame({'Year_Factor': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5], 'signal': range(10)})
        test = pd.DataFrame({'Year_Factor': [6, 6], 'signal': [10, 11]})
        bundle = self.bundle(X, np.arange(10, dtype=float), 'regression', test)
        fit, valid, desc = validation_split(bundle)
        self.assertIn('temporal:Year_Factor', desc)
        self.assertEqual(set(bundle.X.iloc[valid]['Year_Factor']), {5})
        self.assertEqual(set(bundle.X.iloc[fit]['Year_Factor']), {1, 2, 3, 4})

    def test_all_singleton_classes_are_retained_when_no_safe_validation_exists(self):
        X = pd.DataFrame({'x': range(5)})
        y = np.array(['a', 'b', 'c', 'd', 'e'])
        bundle = self.bundle(X, y)
        fit, valid, desc = validation_split(bundle)
        self.assertEqual(set(fit), set(range(5)))
        self.assertEqual(len(valid), 0)
        self.assertIn('no-validation-safe-rows', desc)

    def test_singleton_class_is_kept_in_fit_and_limitation_is_reported(self):
        X = pd.DataFrame({'x': range(20)})
        y = np.array(['common'] * 19 + ['singleton'])
        bundle = self.bundle(X, y)
        fit, valid, desc = validation_split(bundle)
        self.assertIn(19, fit)
        self.assertNotIn(19, valid)
        self.assertIn('singleton-classes-kept-in-fit=1', desc)
        self.assertNotEqual(split_key(fit, valid), split_key(valid, fit))


class MetricAndSubmissionTests(unittest.TestCase):
    def test_binary_auc_respects_explicit_positive_class(self):
        y = np.array(['no', 'yes', 'no', 'yes'])
        p_yes = np.array([0.1, 0.9, 0.2, 0.8])
        self.assertAlmostEqual(score('roc_auc', y, p_yes, classes=('no', 'yes'), positive_class='yes'), 1.0)
        self.assertAlmostEqual(score('roc_auc', y, 1 - p_yes, classes=('no', 'yes'), positive_class='no'), 1.0)
        p_columns = np.column_stack([1 - p_yes, p_yes])
        self.assertAlmostEqual(score('roc_auc', y, p_columns, classes=('no', 'yes'), positive_class='yes'), 1.0)

    def test_metric_aliases_and_directions(self):
        self.assertEqual(infer_metric('Primary metric: F1 weighted'), 'weighted_f1')
        self.assertEqual(infer_metric('Primary metric: micro-F1'), 'micro_f1')
        self.assertAlmostEqual(score('micro_f1', ['a', 'b', 'a', 'b'], ['a', 'b', 'b', 'b'], classes=('a', 'b')), 0.75)
        self.assertEqual(infer_metric('Primary metric: R²'), 'r2')
        self.assertEqual(infer_metric('Primary metric: AUC'), 'roc_auc')
        self.assertGreater(score('r2', [0, 1, 2], [0, 1, 2]), .99)

    def test_log_loss_uses_named_class_order(self):
        y = np.array(['yes', 'no'])
        pred = np.array([[0.9, 0.1], [0.1, 0.9]])
        self.assertLess(score('log_loss', y, pred, classes=('yes', 'no')), 0.11)

    def test_submission_validates_sample_order_and_atomic_write(self):
        test = pd.DataFrame({'id': [2, 1], 'x': [20, 10]})
        sample = pd.DataFrame({'id': [1, 2], 'target': [0.0, 0.0]})
        train = pd.DataFrame({'id': [3, 4], 'x': [30, 40], 'target': [1.0, 2.0]})
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        bundle = TaskData(root / 'task.md', '', {}, train, test, sample, pd.DataFrame(),
                          'target', 'id', 'regression', 'mae')
        frame = make_submission(bundle, np.array([10.0, 20.0]))
        self.assertEqual(frame['target'].tolist(), [10.0, 20.0])
        self.assertEqual(frame['id'].tolist(), [1, 2])
        output = atomic_submission(bundle, frame)
        self.assertEqual(output, None)
        validate_submission(bundle, pd.read_csv(root / 'submission.csv'))
        bad = frame.iloc[::-1].reset_index(drop=True)
        with self.assertRaisesRegex(ValueError, 'ID values or order'):
            validate_submission(bundle, bad)

    def test_single_column_probability_fallback_uses_positive_class(self):
        train = pd.DataFrame({'x': [0, 1, 2, 3], 'target': ['no', 'yes', 'yes', 'yes']})
        test = pd.DataFrame({'x': [4, 5]})
        sample = pd.DataFrame({'target': [0.0, 0.0]})
        bundle = TaskData(Path('/tmp/task.md'), '', {}, train, test, sample, pd.DataFrame(),
                          'target', None, 'classification', 'roc_auc', ('no', 'yes'), 'no')
        result = fallback_submission(bundle)
        self.assertEqual(result['target'].tolist(), [0.25, 0.25])

    def test_multiclass_probability_columns_are_remapped_once_to_sample_order(self):
        train = pd.DataFrame({'x': [0, 1, 2], 'target': ['a', 'b', 'c']})
        test = pd.DataFrame({'x': [3, 4]})
        sample = pd.DataFrame({'c': [0.0, 0.0], 'a': [0.0, 0.0], 'b': [0.0, 0.0]})
        bundle = TaskData(Path('/tmp/task.md'), '', {}, train, test, sample, pd.DataFrame(),
                          'target', None, 'classification', 'log_loss', ('a', 'b', 'c'))
        result = make_submission(bundle, np.array([[0.7, 0.2, 0.1], [0.1, 0.2, 0.7]]))
        self.assertEqual(result.iloc[0].tolist(), [0.1, 0.7, 0.2])
        self.assertEqual(result.iloc[1].tolist(), [0.7, 0.1, 0.2])
        bad = result.copy()
        bad.iloc[0, :] = [0.1, 0.7, 0.7]
        with self.assertRaisesRegex(ValueError, 'sum to one'):
            validate_submission(bundle, bad)

    def test_fallback_submission_has_valid_rows(self):
        train = pd.DataFrame({'x': [0, 1, 2], 'target': ['n', 'y', 'y']})
        test = pd.DataFrame({'x': [3, 4]})
        sample = pd.DataFrame({'target': ['', '']})
        bundle = TaskData(Path('/tmp/task.md'), '', {}, train, test, sample, pd.DataFrame(),
                          'target', None, 'classification', 'macro_f1', ('n', 'y'))
        result = fallback_submission(bundle)
        self.assertEqual(result['target'].tolist(), ['y', 'y'])


if __name__ == '__main__':
    unittest.main()
