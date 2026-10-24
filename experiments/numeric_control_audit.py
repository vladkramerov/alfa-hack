"""Single necessary regression control: current preprocessing differs from legacy H12."""
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from automl_data import load_task, validation_split, split_key, score
from automl_models import build_pipeline


def run():
    output = ROOT / 'results/numeric-control-audit.json'
    if output.exists():
        raise RuntimeError('Audit exists; reuse it')
    study = json.loads((ROOT / 'results/numeric-views-v1.json').read_text())
    if study.get('status') != 'completed':
        raise RuntimeError('Finish frozen study first')
    if study['training_seconds'] > 1080:
        raise RuntimeError('Training budget reserved for this audit exhausted')
    record = study['tasks']['regression']
    bundle = load_task(ROOT / 'drive-download-20261023T201455Z-1-001/Регрессия/task.md')
    fit, valid, description = validation_split(bundle)
    assert split_key(fit, valid) == record['validation']['audit_key']
    candidate = record['control_candidate']
    start = time.monotonic()
    model = build_pipeline('regression', candidate, 4).fit(bundle.X.iloc[fit], bundle.y.iloc[fit])
    fit_seconds = time.monotonic() - start
    prediction = model.predict(bundle.X.iloc[valid])
    result = {'reason': 'Legacy H12 used dense onehot without the current scaling; current-code paired control required to isolate numerical-view effect.',
              'candidate': candidate, 'metric': 'rmse', 'score': score('rmse', bundle.y.iloc[valid].to_numpy(), prediction),
              'split_key': split_key(fit, valid), 'description': description, 'data_hashes': bundle.hashes,
              'fit_seconds': fit_seconds, 'study_plus_control_fit_seconds': study['training_seconds'] + fit_seconds,
              'model_source_sha256': hashlib.sha256((ROOT/'automl_models.py').read_bytes()).hexdigest()}
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps(result))


if __name__ == '__main__':
    run()
