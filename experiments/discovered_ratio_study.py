"""No learner refits: screen generic H83 transfer using saved causal control predictions."""
import json
from pathlib import Path
import resource
import sys
import time
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from automl_data import load_task, validation_split, split_key, score
from automl_ratio import DiscoveredRatio
OUT = ROOT / 'results/team-runtime/discovered-ratio.json'
WORK = ROOT / 'local_data/team-runtime'


def main():
    if OUT.exists() and json.loads(OUT.read_text())['status'] == 'completed':
        raise RuntimeError('Reuse result; no repeated fits')
    WORK.mkdir(parents=True, exist_ok=True)
    b = load_task(ROOT / 'drive-download-20261023T201455Z-1-001/Регрессия/task.md')
    periods = b.X['Year_Factor'].to_numpy()
    fit, valid = np.flatnonzero(periods < 4), np.flatnonzero(periods == 4)
    r = {'status': 'running', 'hashes': b.hashes, 'fits': [], 'base_learner_fits': 0,
         'protocol': 'HT2 generic H83 transfer; finite-pair driver discovery rather than ST4 missing-zero deltas, inferred percent support, hierarchical category curves. Base frozen generic history ET. Latest temporal inner because older period has zero history; gate>=.5 RMSE improvement. One outer only after gate. Same repeatedly viewed outer; not independent.'}
    OUT.write_text(json.dumps(r, indent=2))
    def stage(fit, valid, cache, name):
        with np.load(cache) as z:
            assert np.array_equal(z['positions'], valid)
            base = z['prediction']
        destination = WORK / ('ratio-'+name+'.npz')
        if destination.exists():
            with np.load(destination) as z:
                assert np.array_equal(z['positions'], valid)
                pred = z['prediction']
            audit = {'recovered_prediction_after_receipt_bool_serialization_failure': True,
                     'model_not_refit': True}
            seconds, rss = None, None
        else:
            started = time.monotonic()
            model = DiscoveredRatio('Year_Factor').fit(b.X.iloc[fit], b.y.iloc[fit])
            pred = model.predict(b.X.iloc[valid], base)
            audit = model.audit_
            seconds = time.monotonic()-started
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
        result = {'stage': name, 'key': split_key(fit, valid), 'base': score('rmse', b.y.iloc[valid], base),
                  'score': score('rmse', b.y.iloc[valid], pred), 'audit': audit,
                  'seconds': seconds, 'rss_mb': rss}
        if not destination.exists():
            np.savez_compressed(destination, positions=valid, prediction=pred)
        r['fits'].append(result)
        OUT.write_text(json.dumps(r, indent=2, ensure_ascii=False))
        print(json.dumps(result), flush=True)
        return result
    inner = stage(fit, valid, '/private/tmp/alfa-score-recovery-panel/candidate-3-fold-0.npz', 'inner')
    if inner['score'] <= inner['base']-.5:
        fit, valid, _ = validation_split(b)
        stage(fit, valid, '/private/tmp/alfa-score-recovery-panel/selected-outer.npz', 'outer')
    else:
        r['decision'] = 'rejected_inner'
    r['status'] = 'completed'
    OUT.write_text(json.dumps(r, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
