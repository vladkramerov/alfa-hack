"""HT6: target-aware numerical cuts; same neural backbone and class control.

Mechanism reference: Gorishniy et al., arXiv:2203.05556, section3.2.2.
Independent small implementation using permitted sklearn; no rtdl dependency.
"""
import json
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import automl_torch_tabm as neural
from sklearn.tree import DecisionTreeClassifier
from automl_data import load_task,validation_split,split_key,score
from automl_engine import _search_windows
from automl_models import build_pipeline
from automl_worker import train_priors,class_decisions


class TreeBinEncoder(neural.TorchFrameEncoder):
    def __init__(self, labels, bins=8):
        super().__init__(bins)
        self.labels=labels

    def fit(self,X):
        super().fit(X)
        frame=self._frame(X)
        matrix=self.imputer_.transform(frame[self.numeric_]) if self.imputer_ is not None else np.empty((len(X),0))
        labels,encoded=np.unique(np.asarray(self.labels),return_inverse=True)
        counts=np.bincount(encoded)
        weights=1/np.sqrt(counts[encoded]);weights/=weights.mean()
        original=self.pwl_.mask_.sum(axis=1).tolist()
        self.pwl_.left_.fill(0);self.pwl_.width_.fill(1);self.pwl_.mask_.fill(False)
        fitted=0
        for j in range(matrix.shape[1]):
            values=matrix[:,j]
            edges=[values.min(),values.max()]
            if edges[0]!=edges[1]:
                tree=DecisionTreeClassifier(max_leaf_nodes=self.bins,min_samples_leaf=64,random_state=42)
                tree.fit(values.reshape(-1,1),encoded,sample_weight=weights)
                branches=tree.tree_.children_left!=tree.tree_.children_right
                edges.extend(tree.tree_.threshold[branches].tolist())
                fitted+=1
            edges=np.unique(edges)
            count=len(edges)-1
            if count:
                self.pwl_.left_[j,:count]=edges[:-1]
                self.pwl_.width_[j,:count]=np.diff(edges)
                self.pwl_.mask_[j,:count]=True
        self.audit_={'quantile_active_bins':original,'supervised_active_bins':self.pwl_.mask_.sum(axis=1).tolist(),
                     'tree_fits':fitted,'max_leaf_nodes':self.bins,'min_samples_leaf':64,
                     'sample_weight':'same inverse-sqrt as neural CE','statistics_source':'FIT X/y only',
                     'query_used':False,'thresholds_published':False}
        return self


def main():
    out=ROOT/'results/neural-transfer/tree-bins.json'
    if out.exists():raise RuntimeError('Inspect receipt before any repeated fit')
    work=ROOT/'local_data/neural-transfer';work.mkdir(exist_ok=True)
    b=load_task(ROOT/'drive-download-20261023T201455Z-1-001/Многоклассовая классификация/task.md')
    of,ov,_=validation_split(b);window=_search_windows(b,of,1800)[0][0]
    candidate=next(r['selected'] for r in json.loads((ROOT/'results/strongest-team/e2e-v2.json').read_text())['runs'] if r['task']=='classification' and r['scope']=='shadow')
    report={'status':'running','candidate':candidate,'hashes':b.hashes,'stages':[],
            'protocol':'HT6 predeclared: replace only numeric bin edges with inverse-sqrt label-weighted 1Dtrees8leaves/minleaf64; same FullTabM8x96/PWL16d/60epochs/120sec/4CPU. Reference arXiv2203.05556. HT5 class rejected quantile baseline, do not refit it. Own teacher cached. Same inner, gate>=.002 F1 and positiveweight, grid0/.05/.1/.2/.4/.7/1, freeze before outer. Tree threshold fits counted separately; no new package/pretrained state.'}
    def save():out.write_text(json.dumps(report,indent=2,ensure_ascii=False))
    save()
    for stage in ['inner','outer']:
        if stage=='outer' and not report.get('admitted_outer'):break
        fit,valid=(window['fit'],window['valid']) if stage=='inner' else (of,ov)
        path=ROOT/'local_data/team-runtime/inner-teacher.npz' if stage=='inner' else work/'tree-bins-outer-teacher.npz'
        if path.exists():
            with np.load(path) as z:
                assert np.array_equal(z['fit'],fit) and np.array_equal(z['valid'],valid)
                base=z['probabilities']
        else:
            teacher=build_pipeline('classification',candidate,8).fit(b.X.iloc[fit],b.y.iloc[fit])
            assert np.array_equal(teacher.classes_,b.classes)
            base=teacher.predict_proba(b.X.iloc[valid])
            np.savez_compressed(path,fit=fit,valid=valid,probabilities=base)
        encoders=[]
        def encoder_factory(bins=8):
            encoder=TreeBinEncoder(b.y.iloc[fit],bins);encoders.append(encoder);return encoder
        with patch.object(neural,'TorchFrameEncoder',side_effect=encoder_factory):
            model=neural.TorchTabM(kind='classification',seconds=120,threads=4).fit(b.X.iloc[fit],b.y.iloc[fit])
        assert np.array_equal(model.classes_,b.classes)
        prediction=model.predict_proba(b.X.iloc[valid])
        audit={'model':model.audit_,'encoding':encoders[0].audit_}
        (work/('tree-bins-'+stage+'.audit.json')).write_text(json.dumps(audit,indent=2))
        np.savez_compressed(work/('tree-bins-'+stage+'.npz'),fit=fit,valid=valid,probabilities=prediction)
        priors=train_priors(b.y.iloc[fit],b.classes)
        def value(weight):return score('macro_f1',b.y.iloc[valid],class_decisions((1-weight)*base+weight*prediction,b.classes,priors,candidate.get('decision_power',0)))
        grid=[{'weight':w,'score':value(w)} for w in [0,.05,.1,.2,.4,.7,1]] if stage=='inner' else []
        weight=max(grid,key=lambda x:x['score'])['weight'] if grid else report['weight']
        result={'stage':stage,'key':split_key(fit,valid),'base':value(0),'score':value(weight),
                'standalone':value(1),'weight':weight,'audit':audit}
        report['stages'].append(result)
        if grid:report.update(grid=grid,weight=weight,admitted_outer=weight>0 and result['score']>=result['base']+.002)
        save();print(json.dumps({k:v for k,v in result.items() if k!='audit'}),flush=True)
    report['status']='completed';save()


if __name__=='__main__':main()
