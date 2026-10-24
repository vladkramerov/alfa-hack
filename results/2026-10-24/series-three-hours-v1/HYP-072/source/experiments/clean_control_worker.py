"""Explicitly authorized old-leader comparison; never imported by the new agent."""
import json
import os
from pathlib import Path
import sys
import time
import traceback

# Frozen offline worker lives one directory below the flat source snapshot.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from research_data import load_task, score, diagnostics
from research_memory import Memory, atomic_json


def run(request):
    output = Path(request['output'])
    started = time.monotonic()
    history = Memory(request['history'], request['global_ledger'])
    fingerprint = None
    try:
        task = load_task(request['task_path'])
        with np.load(request['split_path'], allow_pickle=False) as split:
            fit, valid = split['fit'], split['valid']
        os.environ.update(COMBO_RUN_DIR=request['control_fit_log'], COMBO_LEDGER_PATH=request['global_ledger'],
                          COMBO_HYPOTHESIS='SPEC010-authorized-control-' + task.task_type)
        if task.task_type == 'regression':
            from automl_temporal_memory import TemporalProfileModel
            model = TemporalProfileModel({'parameters': {}}, threads=4)
        else:
            from automl_class_combo import ClassCombination
            # Do NOT inject public frozen meta_state trained with confirmation labels.
            model = ClassCombination({'strategy': 'probability_stacking', 'parameters': {}}, threads=4)
        fingerprint = history.claim(request['recipe'], task.X.iloc[fit], task.y.iloc[fit], 'control')
        model.fit(task.X.iloc[fit], task.y.iloc[fit])
        prediction = model.predict(task.X.iloc[valid])
        saved = {'positions': valid, 'y': task.y.iloc[valid].to_numpy(), 'prediction': prediction}
        if task.task_type == 'classification':
            saved.update(y=saved['y'].astype(str), prediction=prediction.astype(str),
                         classes=np.asarray(model.classes_, dtype=str), probabilities=model.predict_proba(task.X.iloc[valid]))
        np.savez_compressed(output / 'validation_predictions.npz', **saved)
        atomic_json(output / 'audit.json', getattr(model, 'audit_', {}))
        result = {'status': 'completed', 'metric': task.metric, 'score': score(task.metric, task.y.iloc[valid], prediction),
                  'fit_rows': len(fit), 'valid_rows': len(valid), 'seconds': time.monotonic() - started,
                  'diagnostics': diagnostics(task.task_type, task.y.iloc[valid], prediction, getattr(model, 'classes_', None)),
                  'fingerprint': fingerprint, 'frozen_public_meta_reused': False}
        history.finish(fingerprint, result)
    except Exception as exc:
        result = {'status': 'failed', 'error': type(exc).__name__ + ': ' + str(exc), 'seconds': time.monotonic() - started}
        (output / 'traceback.txt').write_text(traceback.format_exc(), encoding='utf8')
        if fingerprint is not None:
            history.finish(fingerprint, result)
    atomic_json(output / 'metrics.json', result)
    return 0 if result['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(run(json.loads(Path(sys.argv[-1]).read_text(encoding='utf8'))))
