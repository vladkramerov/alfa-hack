"""Small original CPU-only PLE/BatchEnsemble adaptations, SPEC012.

Uses only organizer-permitted packages; no upstream tabm/embedding dependency.
These experiments are not an exact implementation of TabM or RealMLP.
"""
import math
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import f1_score


CONFIG = dict(bins=16, embedding=8, width=128, depth=2, dropout=0.1,
              learning_rate=0.002, weight_decay=0.0003, batch_size=512,
              max_epochs=30, patience=6, category_cap=64, threads=4, seed=42)


class DeadlineExceeded(TimeoutError):
    pass


def check_deadline(deadline):
    if time.monotonic() >= deadline:
        raise DeadlineExceeded('900-second total hypothesis budget exhausted')


class PLEPreprocessor:
    """Fit-only medians, quantile knots and capped lexical categories.

    Each feature has at most 16 nondegenerate intervals; padded intervals
    remain zero. Three category codes distinguish missing, rare and unseen.
    """
    def __init__(self, bins=16, category_cap=64):
        self.bins = bins
        self.category_cap = category_cap

    @staticmethod
    def lexical(series):
        return series.map(lambda v: None if pd.isna(v) else str(v))

    def fit(self, frame):
        self.columns_ = list(frame.columns)
        self.numeric_ = list(frame.select_dtypes(include=[np.number]).columns)
        self.categorical_ = [c for c in self.columns_ if c not in self.numeric_]
        self.medians_, self.knots_, self.category_maps_, self.seen_ = {}, {}, {}, {}
        for column in self.numeric_:
            x = pd.to_numeric(frame[column], errors='coerce').to_numpy(dtype=float)
            observed = x[np.isfinite(x)]
            median = float(np.median(observed)) if len(observed) else 0.0
            self.medians_[column] = median
            filled = np.where(np.isfinite(x), x, median)
            self.knots_[column] = np.unique(np.quantile(filled, np.linspace(0, 1, self.bins+1)))
        self.category_sizes_ = []
        for column in self.categorical_:
            values = self.lexical(frame[column])
            counts = values.dropna().value_counts()
            # Stable lexical tie-break makes the mapping reproducible.
            ranked = sorted(counts.index, key=lambda v: (-int(counts[v]), v))
            retained = ranked[:max(0, self.category_cap-3)]
            self.category_maps_[column] = {v: j+3 for j, v in enumerate(retained)}
            self.seen_[column] = set(ranked)
            self.category_sizes_.append(len(retained)+3)
        return self

    def transform(self, frame):
        frame = frame[self.columns_]
        encoded = np.zeros((len(frame), len(self.numeric_), self.bins), dtype=np.float32)
        for j, column in enumerate(self.numeric_):
            raw = pd.to_numeric(frame[column], errors='coerce').to_numpy(dtype=float)
            x = np.where(np.isfinite(raw), raw, self.medians_[column])
            knots = self.knots_[column]
            if len(knots) > 1:
                encoded[:, j, :len(knots)-1] = np.clip(
                    (x[:, None]-knots[:-1])/(knots[1:]-knots[:-1]), 0.0, 1.0)
        categorical = np.zeros((len(frame), sum(self.category_sizes_)), dtype=np.float32)
        offset = 0
        for column, size in zip(self.categorical_, self.category_sizes_):
            mapping, seen = self.category_maps_[column], self.seen_[column]
            values = self.lexical(frame[column])
            codes = np.array([0 if v is None else mapping.get(v, 1 if v in seen else 2)
                              for v in values], dtype=np.int64)
            categorical[np.arange(len(frame)), offset+codes] = 1.0
            offset += size
        assert np.isfinite(encoded).all() and np.isfinite(categorical).all()
        return encoded, categorical

    def fit_transform(self, frame):
        return self.fit(frame).transform(frame)

    def state(self):
        return dict(columns=self.columns_, numeric=self.numeric_, categorical=self.categorical_,
                    medians=self.medians_, knots={k:v.tolist() for k,v in self.knots_.items()},
                    category_maps=self.category_maps_, category_sizes=self.category_sizes_,
                    bins=self.bins, category_cap=self.category_cap)


class NumericEmbedding(nn.Module):
    def __init__(self, n_numeric, bins, embedding):
        super().__init__()
        self.n_numeric, self.embedding = n_numeric, embedding
        self.weight = nn.Parameter(torch.empty(n_numeric, bins, embedding))
        self.bias = nn.Parameter(torch.zeros(n_numeric, embedding))
        if n_numeric:
            nn.init.uniform_(self.weight, -1/math.sqrt(bins), 1/math.sqrt(bins))

    def forward(self, x):
        return (torch.einsum('bfi,fie->bfe', x, self.weight)+self.bias).flatten(1)


class BatchEnsembleLinear(nn.Module):
    def __init__(self, d_in, d_out, members, first_chunks=None):
        super().__init__()
        self.linear = nn.Linear(d_in, d_out, bias=False)
        self.r = nn.Parameter(torch.ones(members, d_in))
        self.s = nn.Parameter(torch.ones(members, d_out))
        self.bias = nn.Parameter(torch.zeros(members, d_out))
        if first_chunks is not None:
            assert sum(first_chunks) == d_in
            with torch.no_grad():
                offset = 0
                for size in first_chunks:
                    self.r[:, offset:offset+size] = torch.randn(members, 1)
                    offset += size

    def forward(self, x):
        return self.linear(x*self.r)*self.s+self.bias


class PaperNeural(nn.Module):
    def __init__(self, n_numeric, category_sizes, n_classes, members=1, **configuration):
        super().__init__()
        c = {**CONFIG, **configuration}
        self.members = members
        self.embedding = NumericEmbedding(n_numeric, c['bins'], c['embedding'])
        chunks = [c['embedding']]*n_numeric + list(category_sizes)
        d_in = sum(chunks)
        if d_in == 0:
            raise ValueError('At least one predictor is required')
        layers = []
        for depth in range(c['depth']):
            if members == 1:
                layers.append(nn.Linear(d_in, c['width']))
            else:
                layers.append(BatchEnsembleLinear(d_in, c['width'], members,
                                                  first_chunks=chunks if depth == 0 else None))
            layers.extend([nn.ReLU(), nn.Dropout(c['dropout'])])
            d_in = c['width']
        self.backbone = nn.Sequential(*layers)
        if members == 1:
            self.head = nn.Linear(d_in, n_classes)
        else:
            self.head_weight = nn.Parameter(torch.empty(members, d_in, n_classes))
            self.head_bias = nn.Parameter(torch.zeros(members, n_classes))
            nn.init.uniform_(self.head_weight, -1/math.sqrt(d_in), 1/math.sqrt(d_in))

    def forward(self, numeric, categorical):
        x = torch.cat([self.embedding(numeric), categorical], dim=1)
        if self.members > 1:
            x = x[:, None, :].expand(-1, self.members, -1)
        x = self.backbone(x)
        if self.members == 1:
            return self.head(x)[:, None, :]
        return torch.einsum('bkd,kdc->bkc', x, self.head_weight)+self.head_bias


def class_weights(y, n_classes):
    counts = np.bincount(y, minlength=n_classes)
    weights = 1/np.sqrt(np.maximum(counts, 1))
    weights /= np.mean(weights[y])
    return torch.tensor(weights, dtype=torch.float32)


@torch.inference_mode()
def probabilities(model, tensors, deadline=float('inf')):
    model.eval()
    pieces = []
    numeric, categorical = tensors
    for start in range(0, len(numeric), 1024):
        check_deadline(deadline)
        logits = model(numeric[start:start+1024], categorical[start:start+1024])
        pieces.append(logits.softmax(-1).mean(1).numpy())
    return np.concatenate(pieces)


def tensors(preprocessor, frame):
    return tuple(torch.from_numpy(v) for v in preprocessor.transform(frame))


def train(frame, y, classes, members, deadline, trace, valid_frame=None,
          valid_y=None, epochs=None):
    """Select epochs only on an explicit inner holdout; refit receives epochs.

    No outer target is accepted by this function during fitting.
    """
    check_deadline(deadline)
    torch.manual_seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])
    torch.set_num_threads(CONFIG['threads'])
    preprocessor = PLEPreprocessor(CONFIG['bins'], CONFIG['category_cap']).fit(frame)
    train_tensors = tensors(preprocessor, frame)
    target = torch.tensor(y, dtype=torch.long)
    valid_tensors = tensors(preprocessor, valid_frame) if valid_frame is not None else None
    model = PaperNeural(len(preprocessor.numeric_), preprocessor.category_sizes_, len(classes), members)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'],
                                 weight_decay=CONFIG['weight_decay'])
    weights = class_weights(y, len(classes))
    best_score, best_epoch = -float('inf'), 1
    max_epochs = int(epochs or CONFIG['max_epochs'])
    for epoch in range(1, max_epochs+1):
        check_deadline(deadline)
        model.train()
        ordering = torch.randperm(len(y))
        loss_sum = 0.0
        epoch_start = time.monotonic()
        for start in range(0, len(y), CONFIG['batch_size']):
            check_deadline(deadline)
            index = ordering[start:start+CONFIG['batch_size']]
            logits = model(train_tensors[0][index], train_tensors[1][index])
            labels = target[index, None].expand(-1, members).reshape(-1)
            # Fixed normalized class weights and explicit mean across members.
            losses = F.cross_entropy(logits.reshape(-1, len(classes)), labels,
                                     weight=weights, reduction='none')
            loss = losses.mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite training loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach())*len(index)
        row = dict(epoch=epoch, loss=loss_sum/len(y), seconds=time.monotonic()-epoch_start)
        if valid_tensors is not None:
            prob = probabilities(model, valid_tensors, deadline)
            inner_f1 = float(f1_score(valid_y, prob.argmax(1), labels=np.arange(len(classes)),
                                     average='macro', zero_division=0))
            row['inner_macro_f1'] = inner_f1
            if inner_f1 > best_score:
                best_score, best_epoch = inner_f1, epoch
        trace(row)
        if valid_tensors is not None and epoch-best_epoch >= CONFIG['patience']:
            break
    return model, preprocessor, dict(best_epoch=best_epoch if valid_tensors is not None else max_epochs,
                                    inner_score=best_score if valid_tensors is not None else None,
                                    epochs_run=epoch,
                                    parameters=sum(p.numel() for p in model.parameters()),
                                    normalized_class_weights=weights.tolist())
