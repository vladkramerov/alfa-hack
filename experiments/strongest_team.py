"""Finite, frozen inner-first transport and leaf-retrieval screening."""
import sys,json,time,resource,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from automl_data import load_task,validation_split,split_key,score
from automl_transfer import TemporalDonor,LeafRetrieval
from automl_engine import _search_windows
from automl_models import build_pipeline
from automl_worker import class_decisions,train_priors
OUT=ROOT/'results/strongest-team';LOCAL=Path('/private/tmp/alfa-strongest-team');LOCAL.mkdir(exist_ok=True)
def dump(name,record):
    (OUT/name).write_text(json.dumps(record,indent=2,ensure_ascii=False))
def regression():
    if (OUT/'transport.json').exists():raise RuntimeError('Reuse result, no repeated fits')
    b=load_task(ROOT/'drive-download-20261023T201455Z-1-001/Регрессия/task.md')
    outerfit,outervalid,_=validation_split(b)
    old=json.loads((ROOT/'results/score-recovery/panel.json').read_text())
    record={'status':'running','hashes':b.hashes,'hypotheses':[],'outer_previously_viewed':True}
    for mode in [False,True]:
        rec={'transport':mode,'folds':[]};record['hypotheses'].append(rec)
        for j,period in enumerate([4,3]):
            t=b.X['Year_Factor'].to_numpy();fit=np.flatnonzero(t<period);valid=np.flatnonzero(t==period)
            assert split_key(fit,valid)==old['recipes'][3]['folds'][j]['key']
            with np.load(f'/private/tmp/alfa-score-recovery-panel/candidate-3-fold-{j}.npz') as z:
                assert np.array_equal(z['positions'],valid);base=z['prediction']
            start=time.monotonic();m=TemporalDonor('Year_Factor',mode).fit(b.X.iloc[fit],b.y.iloc[fit]);duration=time.monotonic()-start
            pred=m.predict(b.X.iloc[valid],base)
            np.savez_compressed(LOCAL/f'transport-{int(mode)}-inner-{j}.npz',positions=valid,prediction=pred)
            rec['folds'].append({'key':split_key(fit,valid),'base':score('rmse',b.y.iloc[valid],base),'score':score('rmse',b.y.iloc[valid],pred),'seconds':duration,'audit':m.audit_,'linked':m.last_linked_ if hasattr(m,'last_linked_') else 0})
            dump('transport.json',record)
        rec['score']=.65*rec['folds'][0]['score']+.35*rec['folds'][1]['score']
        print('TRANSPORT INNER',mode,rec['score'],flush=True)
    winner=min(record['hypotheses'],key=lambda x:x['score']);record['frozen_mode']=winner['transport'];dump('transport.json',record)
    if winner['score']<old['recipes'][3]['score']:
        with np.load('/private/tmp/alfa-score-recovery-panel/selected-outer.npz') as z:
            assert np.array_equal(z['positions'],outervalid);base=z['prediction']
        start=time.monotonic();m=TemporalDonor('Year_Factor',winner['transport']).fit(b.X.iloc[outerfit],b.y.iloc[outerfit]);pred=m.predict(b.X.iloc[outervalid],base)
        record['outer']={'base':score('rmse',b.y.iloc[outervalid],base),'score':score('rmse',b.y.iloc[outervalid],pred),'seconds':time.monotonic()-start,'audit':m.audit_,'key':split_key(outerfit,outervalid)}
        np.savez_compressed(LOCAL/'transport-selected-outer.npz',positions=outervalid,prediction=pred)
    record['status']='completed';record['rss_mb']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2;dump('transport.json',record);print(json.dumps(record.get('outer')),flush=True)
def classification():
    if (OUT/'leaf-retrieval.json').exists():raise RuntimeError('Reuse result')
    b=load_task(ROOT/'drive-download-20261023T201455Z-1-001/Многоклассовая классификация/task.md');of,ov,_=validation_split(b);w=_search_windows(b,of,1800)[0][0];fit,valid=w['fit'],w['valid']
    cache=next(Path('/private/tmp/alfa-work-v5/multiclass-adaptive/agent_runs').glob('*/experiment-09-fold01/probability_cache.npz'))
    with np.load(cache,allow_pickle=False) as z:
        assert np.array_equal(z['fit_positions'],fit) and np.array_equal(z['valid_positions'],valid);base=z['valid_probabilities']
    priors=train_priors(b.y.iloc[fit],b.classes)
    c={'family':'lightgbm','parameters':{'n_estimators':200,'num_leaves':31,'min_child_samples':20,'reg_lambda':3},'class_weight_power':.5}
    rec={'status':'running','candidate':c,'key':split_key(fit,valid),'hashes':b.hashes,'outer_previously_viewed':True};dump('leaf-retrieval.json',rec)
    start=time.monotonic();m=build_pipeline('classification',c,8).fit(b.X.iloc[fit],b.y.iloc[fit]);rec['teacher_seconds']=time.monotonic()-start
    start=time.monotonic();retrieval=LeafRetrieval().fit(m,b.X.iloc[fit],b.y.iloc[fit]);p=retrieval.predict_proba(b.X.iloc[valid]);rec['retrieval_seconds']=time.monotonic()-start
    assert np.array_equal(retrieval.classes_,b.classes)
    rec['grid']=[]
    for weight in [0,.25,.5,.75,1]:
        pred=class_decisions((1-weight)*base+weight*p,b.classes,priors,.3)
        rec['grid'].append({'weight':weight,'score':score('macro_f1',b.y.iloc[valid],pred)})
    np.savez_compressed(LOCAL/'leaf-inner.npz',positions=valid,probabilities=p,classes=np.asarray(b.classes))
    rec['status']='completed';rec['rss_mb']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2;dump('leaf-retrieval.json',rec);print(json.dumps(rec),flush=True)
if __name__=='__main__':
    regression() if sys.argv[1]=='regression' else classification()
