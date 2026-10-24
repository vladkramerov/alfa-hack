"""Export the complete requested preprocessing separately from model selection.

This performs no model fitting or metric comparison. The full-train export
is for future training/inference, and must not be reused to score validation.
"""
import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from experiments.preprocessing import DataPreprocessor
from experiments.run_suite import RECIPES, ROOT, save_json


def export(task, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    data = ROOT / 'data' / task
    train, test = pd.read_csv(data/'train.csv'), pd.read_csv(data/'test.csv')
    target = 'site_eui' if task=='regression' else 'attack_cat'
    X, y = train.drop(columns=['id',target]), train[target]
    preprocessor = DataPreprocessor(task, **RECIPES['engineered'])
    A, B = preprocessor.fit_transform(X), None
    B = preprocessor.transform(test.drop(columns='id'))
    assert np.isfinite(A).all() and np.isfinite(B).all()
    joblib.dump(preprocessor, destination/'preprocessor.joblib', compress=3)
    np.savez_compressed(destination/'preprocessed_data.npz', X_train=A, X_test=B,
                        y_train=y.to_numpy(), train_id=train.id.to_numpy(), test_id=test.id.to_numpy(),
                        feature_names=preprocessor.get_feature_names_out().astype(str))
    save_json(destination/'preprocessing_audit.json',preprocessor.audit_)
    save_json(destination/'manifest.json',dict(
        task=task, recipe='engineered', fit_scope='full original train; no test fit; not for retrospective validation',
        created_at=datetime.now(timezone(timedelta(hours=3))).isoformat(),
        shape_train=A.shape, shape_test=B.shape, dtype=str(A.dtype),
        original_data_sha256={name:hashlib.sha256((data/name).read_bytes()).hexdigest() for name in ['train.csv','test.csv']},
        preprocessor_sha256=hashlib.sha256((ROOT/'experiments/preprocessing.py').read_bytes()).hexdigest()))
    (destination/'README.md').write_text('# Полный запрошенный вариант предобработки\n\n'
        'Включены imputation, correlation, VIF<=10, StandardScaler, train-fit Yeo–Johnson и feature engineering.\n'
        'Статистики обучены только на полном исходном train; test только преобразован.\n'
        'Этот экспорт предназначен для дальнейшего полного обучения и inference, не для оценки сохранённых validation.\n'
        'Он сохранён независимо от того, улучшил ли полный pipeline метрику и выбран ли он для финальной модели.\n\n'
        '[Массивы](preprocessed_data.npz), [препроцессор](preprocessor.joblib), [аудит](preprocessing_audit.json), [manifest](manifest.json).\n'
        'NPZ содержит X_train, X_test, y_train, train_id, test_id, feature_names; классы y_train хранятся строками/объектами.\n'
        'Для безопасного чтения числовых массивов используйте numpy.load(..., allow_pickle=False); y_train при необходимости берите из исходного CSV.\n',encoding='utf-8')
    return dict(task=task,train=A.shape,test=B.shape,path=str(destination))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',required=True,choices=['regression','multiclass'])
    parser.add_argument('--destination',required=True,type=Path)
    args=parser.parse_args()
    print(json.dumps(export(args.task,args.destination),ensure_ascii=False))
