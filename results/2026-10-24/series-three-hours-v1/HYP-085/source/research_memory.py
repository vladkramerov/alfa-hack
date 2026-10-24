"""Persistent experiment identity and explicitly agreed metric trade-offs."""
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

IGNORED = {'seed', 'random_state', 'random_seed', 'name', 'hypothesis', 'n_jobs',
           'thread_count', 'threads', 'verbose', 'verbosity'}


def timestamp():
    return datetime.now(timezone(timedelta(hours=3))).isoformat()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=path.name + '-', dir=path.parent)
    try:
        with os.fdopen(handle, 'w', encoding='utf8') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, default=str, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def semantic_recipe(value):
    if isinstance(value, dict):
        return {str(k): semantic_recipe(v) for k, v in value.items() if k not in IGNORED}
    if isinstance(value, (list, tuple)):
        return [semantic_recipe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def data_digest(value):
    frame = value if isinstance(value, pd.DataFrame) else pd.DataFrame(np.asarray(value))
    digest = hashlib.sha256(json.dumps([(str(c), str(t)) for c, t in zip(frame.columns, frame.dtypes)]).encode())
    digest.update(pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes())
    return digest.hexdigest()


class Memory:
    def __init__(self, path, global_ledger=None):
        self.path = Path(path)
        self.global_ledger = Path(global_ledger) if global_ledger else None

    def records(self):
        if not self.path.exists():
            return []
        result = json.loads(self.path.read_text(encoding='utf8'))
        if not isinstance(result, list):
            raise ValueError('History must contain a list')
        return result

    def claim(self, recipe, X, y, stage='inner'):
        # Read the COMPLETE human ledger before every real fit, never a tail.
        ledger = self.global_ledger.read_text(encoding='utf8') if self.global_ledger else ''
        identity = {'recipe': semantic_recipe(recipe), 'X_sha256': data_digest(X),
                    'y_sha256': data_digest(np.asarray(y).reshape(-1, 1))}
        fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode()).hexdigest()
        rows = self.records()
        if any(row['fingerprint'] == fingerprint for row in rows):
            raise RuntimeError('Configuration and data already attempted: ' + fingerprint)
        rows.append({**identity, 'fingerprint': fingerprint, 'status': 'started', 'stage': stage,
                     'started': timestamp(), 'ledger_sha256': hashlib.sha256(ledger.encode()).hexdigest() if ledger else None})
        atomic_json(self.path, rows)
        return fingerprint

    def finish(self, fingerprint, result):
        rows = self.records()
        record = next((r for r in rows if r['fingerprint'] == fingerprint), None)
        if record is None or record['status'] != 'started':
            raise RuntimeError('No unfinished claim for result')
        record.update(result, finished=timestamp())
        atomic_json(self.path, rows)


def promotion(control, candidate):
    values = [control['rmse'], candidate['rmse'], control['macro_f1'], candidate['macro_f1']]
    if not all(math.isfinite(v) for v in values) or control['rmse'] <= 0 or control['macro_f1'] <= 0:
        raise ValueError('Finite positive control metrics are required')
    gains = {'regression': (control['rmse'] - candidate['rmse']) / control['rmse'],
             'classification': (candidate['macro_f1'] - control['macro_f1']) / control['macro_f1']}
    reg, cls = gains.values()
    eps = 1e-12  # Absorb float boundary representation; don't turn strict > into >=.
    no_regression = min(reg, cls) >= -eps and max(reg, cls) > eps
    tradeoff = (reg > .15 + eps and cls > -.10 + eps) or (cls > .15 + eps and reg > -.10 + eps)
    return {'accepted': bool(no_regression or tradeoff), 'relative_gains': gains,
            'rule': 'nonnegative_both_with_one_gain_or_gain_gt15pct_loss_lt10pct'}
