"""Standalone NumPy TabM-mini research implementation; not the official package.

Architecture reference: yandex-research/tabm/tabm_reference.py, arch_type=tabm-mini.
Trainable input adapters, shared MLP and separate heads. No pretrained weights.
"""
import time

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, QuantileTransformer
from threadpoolctl import threadpool_limits


def softmax(values):
    shifted = values - values.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


class MiniEnsemble:
    def __init__(self, inputs, outputs, members=8, width=64, seed=42, dtype=np.float32,
                 sections=None):
        rng = np.random.default_rng(seed)
        self.outputs = outputs
        self.members = members
        def weight(shape, fan):
            return (rng.normal(size=shape) / np.sqrt(fan)).astype(dtype)
        self.params = {
            'r': rng.choice([-1., 1.], size=(members, inputs)).astype(dtype),
            'W0': weight((inputs, width), inputs), 'b0': np.zeros(width, dtype=dtype),
            'W1': weight((width, width), width), 'b1': np.zeros(width, dtype=dtype),
            'Wo': weight((members, width, outputs), width),
            'bo': np.zeros((members, outputs), dtype=dtype),
        }
        if sections:
            start = 0
            for length in sections:
                self.params['r'][:, start:start+length] = rng.choice([-1., 1.], (members, 1))
                start += length
            if start != inputs: raise ValueError('Adapter sections do not match encoded features')

    def forward(self, X):
        p = self.params
        a = X[:, None, :] * p['r'][None, :, :]
        h0 = np.maximum(a @ p['W0'] + p['b0'], 0)
        h1 = np.maximum(h0 @ p['W1'] + p['b1'], 0)
        z = np.einsum('bkh,kho->bko', h1, p['Wo'], optimize=True) + p['bo']
        return z, (X, a, h0, h1)

    def loss_grad(self, X, y):
        z, (X, a, h0, h1) = self.forward(X)
        divisor = len(X) * self.members
        if self.outputs == 1:
            diff = z[:, :, 0] - y[:, None]
            loss = np.mean(diff**2)
            dz = (2 * diff / divisor)[:, :, None]
        else:
            probabilities = softmax(z)
            rows = np.arange(len(X))[:, None]
            heads = np.arange(self.members)[None, :]
            loss = -np.log(np.maximum(probabilities[rows, heads, y[:, None]], 1e-30)).mean()
            dz = probabilities.copy()
            dz[rows, heads, y[:, None]] -= 1
            dz /= divisor
        p = self.params
        g = {'Wo': np.einsum('bkh,bko->kho', h1, dz, optimize=True), 'bo': dz.sum(axis=0)}
        dh1 = np.einsum('bko,kho->bkh', dz, p['Wo'], optimize=True)
        dh1 *= h1 > 0
        g['W1'] = h0.reshape(-1, h0.shape[-1]).T @ dh1.reshape(-1, dh1.shape[-1])
        g['b1'] = dh1.sum(axis=(0, 1))
        dh0 = (dh1 @ p['W1'].T) * (h0 > 0)
        g['W0'] = a.reshape(-1, a.shape[-1]).T @ dh0.reshape(-1, dh0.shape[-1])
        g['b0'] = dh0.sum(axis=(0, 1))
        da = dh0 @ p['W0'].T
        g['r'] = (da * X[:, None, :]).sum(axis=0)
        return float(loss), g


class TabMMini(BaseEstimator):
    def __init__(self, kind='regression', members=8, width=64, epochs=60, batch_size=256,
                 learning_rate=.003, weight_decay=.0001, seconds=240, seed=42, threads=4):
        self.kind = kind
        self.members = members
        self.width = width
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.seconds = seconds
        self.seed = seed
        self.threads = threads

    def configuration(self):
        return {'family': 'tabm-mini-numpy', 'implementation': 'research-v1',
                'parameters': self.get_params(), 'depth': 2, 'loss': 'MSE' if self.kind == 'regression' else 'CE',
                'preprocessing': 'fit-only-median-missing-quantilenormal256-ohe64-v1',
                'checkpoint': 'lowest-completed-epoch-training-loss', 'no_row_filter': True}

    def _frame(self, X, fitting=False):
        if not isinstance(X, pd.DataFrame) or not X.columns.is_unique:
            raise ValueError('Unique-column DataFrame required')
        if fitting:
            self.feature_names_in_ = list(X.columns)
            self.numeric_ = list(X.select_dtypes(include='number').columns)
            self.categorical_ = [c for c in X.columns if c not in self.numeric_]
        if set(X.columns) != set(self.feature_names_in_): raise ValueError('Feature schema mismatch')
        frame = X.loc[:, self.feature_names_in_].copy()
        for col in self.numeric_:
            frame[col] = pd.to_numeric(frame[col], errors='coerce').replace([np.inf, -np.inf], np.nan)
        for col in self.categorical_:
            # Distinguish a missing value from any observed string, including '<NA>'.
            frame[col] = frame[col].map(lambda v: 'M:' if pd.isna(v) else 'V:' + str(v))
        return frame

    def fit(self, X, y):
        started = time.monotonic()
        for value, maximum in ((self.members,32),(self.width,512),(self.epochs,200),
                               (self.batch_size,8192),(self.threads,8)):
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
                raise ValueError('Resource setting outside bounds')
        if (not np.isfinite(self.seconds) or not 0 < self.seconds <= 900
                or not np.isfinite(self.learning_rate) or not 0 < self.learning_rate <= .1
                or not np.isfinite(self.weight_decay) or not 0 <= self.weight_decay <= 1):
            raise ValueError('Invalid training setting')
        if self.kind not in ('regression','classification'): raise ValueError('Invalid task kind')
        target = np.asarray(y)
        if target.ndim != 1 or len(target) != len(X) or len(X) < 2 or not len(X.columns):
            raise ValueError('Invalid training data or target')
        if pd.isna(target).any(): raise ValueError('Missing target')
        if self.kind == 'classification':
            encoder = LabelEncoder().fit(target)
            self.classes_ = encoder.classes_
            if len(self.classes_) < 2: raise ValueError('At least two classes required')
            labels = encoder.transform(target)
            outputs = len(self.classes_)
        else:
            target = np.asarray(target, dtype=float)
            if not np.isfinite(target).all(): raise ValueError('Nonfinite target')
            self.target_mean_ = float(target.mean())
            self.target_scale_ = float(target.std()) or 1.
            labels = ((target-self.target_mean_) / self.target_scale_).astype(np.float32)
            outputs = 1
        frame = self._frame(X, fitting=True)
        transformers = []
        if self.numeric_:
            transformers.append(('num', Pipeline([
                ('impute', SimpleImputer(strategy='median', keep_empty_features=True, add_indicator=True)),
                ('quantile', QuantileTransformer(n_quantiles=min(256,len(X)), output_distribution='normal',
                                                 subsample=1000000, random_state=self.seed))]), self.numeric_))
        if self.categorical_:
            transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', max_categories=64,
                                                       sparse_output=False, dtype=np.float32), self.categorical_))
        self.preprocessor_ = ColumnTransformer(transformers, sparse_threshold=0)
        values = self.preprocessor_.fit_transform(frame).astype(np.float32)
        if not np.isfinite(values).all(): raise ValueError('Nonfinite prepared input')
        numeric_size = self.preprocessor_.output_indices_.get('num', slice(0,0)).stop or 0
        sections = [1]*numeric_size
        if self.categorical_:
            onehot = self.preprocessor_.named_transformers_['cat']
            sections += [min(64,len(c)) for c in onehot.categories_]
        self.network_ = MiniEnsemble(values.shape[1], outputs, self.members, self.width, self.seed,
                                     sections=sections)
        p = self.network_.params
        m = {n: np.zeros_like(v) for n,v in p.items()}
        v = {n: np.zeros_like(a) for n,a in p.items()}
        rng = np.random.default_rng(self.seed)
        best, best_loss, step = None, float('inf'), 0
        history = []
        timed_out = False
        with threadpool_limits(limits=self.threads):
            for epoch in range(self.epochs):
                order = rng.permutation(len(values))
                total, seen = 0., 0
                for start in range(0,len(values),self.batch_size):
                    if time.monotonic()-started >= self.seconds:
                        timed_out = True
                        break
                    batch = order[start:start+self.batch_size]
                    loss, grads = self.network_.loss_grad(values[batch], labels[batch])
                    if not np.isfinite(loss): raise FloatingPointError('Nonfinite training loss')
                    total += loss*len(batch)
                    seen += len(batch)
                    step += 1
                    for name in p:
                        gradient = grads[name]
                        m[name] *= .9
                        m[name] += .1*gradient
                        v[name] *= .999
                        v[name] += .001*gradient**2
                        update = (m[name]/(1-.9**step)) / (np.sqrt(v[name]/(1-.999**step))+1e-8)
                        if name.startswith('W'): p[name] *= 1-self.learning_rate*self.weight_decay
                        p[name] -= self.learning_rate*update
                if timed_out: break
                training_loss = total/seen
                history.append(training_loss)
                if training_loss < best_loss:
                    best_loss = training_loss
                    best = {n:a.copy() for n,a in p.items()}
                    self.selected_epoch_ = epoch+1
                if (epoch+1)%10 == 0:
                    print(f'TabM-mini epoch={epoch+1} train_loss={training_loss:.7f} sec={time.monotonic()-started:.1f}',flush=True)
        if best is None: raise TimeoutError('No complete training epoch within budget')
        self.network_.params = best
        self.audit_ = {'implementation':'standalone-numpy-tabm-mini', 'official_package':False,
                       'rows':len(X), 'encoded_features':values.shape[1], 'members':self.members,
                       'epochs_completed':len(history), 'selected_epoch':self.selected_epoch_,
                       'training_losses':history, 'checkpoint_criterion':'training_loss_only',
                       'time_limited':timed_out, 'seconds':time.monotonic()-started,
                       'target_transform':'standardize_original_y' if self.kind=='regression' else 'label_encode',
                       'parameter_count':sum(a.size for a in p.values()), 'fit_only_preprocessing':True}
        return self

    def _predict(self, X):
        if not hasattr(self,'network_'): raise ValueError('Model is not fitted')
        if not len(X): return np.empty((0,len(self.classes_))) if self.kind=='classification' else np.empty(0)
        frame = self._frame(X)
        values = self.preprocessor_.transform(frame).astype(np.float32)
        result = []
        with threadpool_limits(limits=self.threads):
            for start in range(0,len(X),1024):
                z = self.network_.forward(values[start:start+1024])[0]
                result.append(softmax(z).mean(axis=1) if self.kind=='classification' else z[:,:,0].mean(axis=1))
        predictions = np.concatenate(result)
        if not np.isfinite(predictions).all(): raise ValueError('Nonfinite prediction')
        return predictions

    def predict_proba(self, X):
        if self.kind != 'classification': raise ValueError('Classification only')
        return self._predict(X)

    def predict(self, X):
        p = self._predict(X)
        return self.classes_[p.argmax(axis=1)] if self.kind=='classification' else p*self.target_scale_+self.target_mean_
