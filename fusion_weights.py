"""Training-only convex regression pooling; no dataset-specific learned state."""
import numpy as np
from scipy.optimize import minimize


def fit_simplex(predictions, target):
    matrix=np.asarray(predictions,dtype=float)
    y=np.asarray(target,dtype=float)
    if (matrix.ndim!=2 or matrix.shape[0]==0 or matrix.shape[1]<2
            or y.shape!=(len(matrix),) or not np.isfinite(matrix).all() or not np.isfinite(y).all()):
        raise ValueError('Finite aligned nonempty predictions and target required')
    residual=matrix-y[:,None]
    gram=residual.T@residual/len(y)
    result=minimize(lambda w:float(w@gram@w), np.full(matrix.shape[1],1/matrix.shape[1]),
        jac=lambda w:2*gram@w, method='SLSQP', bounds=[(0.,1.)]*matrix.shape[1],
        constraints=[{'type':'eq','fun':lambda w:w.sum()-1.,'jac':lambda w:np.ones_like(w)}],
        options={'maxiter':200,'ftol':1e-12})
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError('Simplex solver failed: '+str(result.message))
    weights=np.clip(result.x,0.,1.)
    return weights/weights.sum()


def combine(predictions, weights):
    matrix=np.asarray(predictions,dtype=float);w=np.asarray(weights,dtype=float)
    if (matrix.ndim!=2 or w.shape!=(matrix.shape[1],) or not np.isfinite(matrix).all()
            or not np.isfinite(w).all() or (w<0).any() or not np.isclose(w.sum(),1.,rtol=0.,atol=1e-10)):
        raise ValueError('Finite predictions and nonnegative simplex weights required')
    return matrix@w
