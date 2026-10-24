"""Three bounded refinements; training and research architecture stay intact."""
import numpy as np
import ag_finalize as F
import ag_ml_metrics as M
from ag_schemas import REGRESSION
BASE_POST=F.fit_postprocess
BASE_GREEDY=F.greedy_ensemble

def bounded_postprocess(metric,y,oof,A,B):
    if metric.name!='macro_f1' or np.asarray(oof).ndim!=2 or oof.shape[1]<3:
        return BASE_POST(metric,y,oof,A,B)
    w=np.ones(oof.shape[1]);cur=M.score(metric,y[A],oof[A])
    for _ in range(3):
        moved=False
        for c in range(len(w)):
            best,score=w[c],cur
            for value in (.5,.75,1.,1.25,1.5,2.):
                trial=w.copy();trial[c]=value
                s=M.score(metric,y[A],oof[A],{'weights':trial})
                if metric.better(s,score) and abs(s-score)>1e-6:best,score=value,s
            if best!=w[c]:w[c],cur,moved=best,score,True
        if not moved:break
    if metric.better(M.score(metric,y[B],oof[B],{'weights':w}),M.score(metric,y[B],oof[B])):
        return {'weights':(w/w.max()).tolist()}
    return {}

def refined_regression_ensemble(metric,y,runs,A,B,max_models,steps=25,step_gain=1e-3):
    old=BASE_GREEDY(metric,y,runs,A,B,max_models,steps,step_gain)
    runs=[r for r in runs if r.oof is not None][:max_models]
    if metric.name!='rmse' or len(runs)<2 or np.asarray(runs[0].oof).ndim!=1:return old
    from scipy.optimize import minimize
    matrix=np.column_stack([r.oof[A] for r in runs]);actual=np.asarray(y)[A]
    scale=max(float(np.std(actual)),1e-6);matrix=matrix/scale;target=actual/scale
    w0=np.zeros(len(runs))
    if old is not None:
        for i,r in enumerate(runs):w0[i]=old.members.get(r.exp_id,0.)
    else:w0[0]=1.
    w0/=w0.sum()
    def fun(w):
        residual=matrix@w-target
        return float(residual@residual/len(target)+1e-4*np.sum((w-w0)**2))
    def jac(w):return 2*matrix.T@(matrix@w-target)/len(target)+2e-4*(w-w0)
    fitted=minimize(fun,w0,jac=jac,method='SLSQP',bounds=[(0.,1.)]*len(w0),constraints={'type':'eq','fun':lambda w:w.sum()-1.,'jac':lambda w:np.ones(len(w))},options={'maxiter':100,'ftol':1e-10})
    if not fitted.success or not np.isfinite(fitted.x).all():return old
    weights=.5*w0+.5*fitted.x
    weights=np.maximum(weights,0);weights/=weights.sum()
    if np.count_nonzero(weights>1e-8)<2:return old
    oof=sum(w*r.oof for w,r in zip(weights,runs));test=sum(w*np.asarray(r.test_pred) for w,r in zip(weights,runs))
    post=BASE_POST(metric,y,oof,A,B)
    candidate=F.Solution('refined-ensemble',oof,test,post,F._cv(metric,y,oof,post,A,B),M.score(metric,y[B],oof[B],post),{r.exp_id:float(w) for w,r in zip(weights,runs) if w>1e-8})
    reference=old if old is not None else F.Solution('reference',runs[0].oof,runs[0].test_pred,BASE_POST(metric,y,runs[0].oof,A,B))
    reference.holdout_b=M.score(metric,y[B],reference.oof[B],reference.post)
    # Preserve the source agent's significance and minimum-gain requirements.
    return F.choose(metric,y,reference,candidate,B,min_gain=.0005,confidence=.8).solution if old is not None else (candidate if F.choose(metric,y,reference,candidate,B,min_gain=.0005,confidence=.8).solution is candidate else None)

def macro_aware_ensemble(metric,y,runs,A,B,max_models,steps=25,step_gain=1e-3):
    if metric.name!='macro_f1' or np.asarray(runs[0].oof).ndim!=2:return BASE_GREEDY(metric,y,runs,A,B,max_models,steps,step_gain)
    runs=[r for r in runs if r.oof is not None][:max_models]
    if len(runs)<2:return None
    k=runs[0].oof.shape[1];labels=np.asarray(y[A],int)
    counts=np.bincount(labels,minlength=k);sample=1/np.sqrt(np.maximum(counts[labels],1))
    def utility(p):
        p=np.asarray(p,dtype=float);p=p/np.maximum(p.sum(1,keepdims=True),1e-15)
        return float(np.average(np.log(np.clip(p[np.arange(len(labels)),labels],1e-15,1.)),weights=sample))
    counts=np.zeros(len(runs));blend=np.zeros_like(runs[0].oof,dtype=float);best=-np.inf
    for _ in range(steps):
        n=counts.sum();scores=[utility(((blend*n+r.oof)/(n+1))[A]) for r in runs];i=int(np.argmax(scores));score=scores[i]
        if n and score<=best+step_gain*abs(best):break
        blend=(blend*n+runs[i].oof)/(n+1);counts[i]+=1;best=score
    if (counts>0).sum()<2:return None
    weights=counts/counts.sum();oof=sum(w*r.oof for w,r in zip(weights,runs));test=sum(w*np.asarray(r.test_pred) for w,r in zip(weights,runs));post=BASE_POST(metric,y,oof,A,B)
    return F.Solution('macro-aware-ensemble',oof,test,post,F._cv(metric,y,oof,post,A,B),M.score(metric,y[B],oof[B],post),{r.exp_id:float(w) for w,r in zip(weights,runs) if w>0})
