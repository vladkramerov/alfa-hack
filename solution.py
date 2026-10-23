"""Universal bounded AutoResearch entry point: python solution.py /path/task.md."""
import datetime
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import pandas as pd
from automl_data import load_task,validation_split,split_key,better,fallback_submission,atomic_submission,validate_submission,read_submission,LOWER_IS_BETTER
from automl_history import candidate_key,known_task,already_known,ledger_case,read_ledger,write_ledger,compatible_policy
from automl_llm import propose
from automl_models import canonical_candidate


def save(path,value): path.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str),encoding='utf-8')


def refit_candidate(candidate,result):
    """Freeze fit-only calibration for refit without changing attempted identity."""
    selected=copy.deepcopy(candidate)
    if selected.get('strategy')=='macro_f1_calibrated' and result.get('calibration_multipliers'):
        selected['class_multipliers']=dict(result['calibration_multipliers'])
    allowed={'reg_convex_blend':'blend_weights','smote_macro_f1':'class_multipliers','frequency_blend_macro_f1':'class_multipliers','probability_stacking':'meta_state'}
    key=allowed.get(selected.get('strategy'))
    if key and key in result.get('learned_state',{}):selected[key]=copy.deepcopy(result['learned_state'][key])
    return canonical_candidate(selected)


def portfolio_candidates(kind,metric,training,training_y):
    """Offer domain-independent strategies only when their fit-only inputs apply."""
    target=np.asarray(training_y)
    log_valid=kind=='regression' and np.isfinite(target).all() and np.all(target>-1)
    candidates=[]
    if log_valid:
        profile={'State_Factor','building_class','facility_type','floor_area','year_built','ELEVATION','Year_Factor'}
        if profile.issubset(training.columns) and pd.api.types.is_numeric_dtype(training['Year_Factor']) and training['Year_Factor'].nunique()>=2:
            candidates.append({'family':'extra_trees','parameters':{},'strategy':'temporal_profile_memory',
                'hypothesis':'Strictly past targets of a fixed physical profile with feature-expert fallback'})
        grouping=[c for c in training.select_dtypes(exclude='number') if 2<=training[c].nunique()<=6]
        if 'building_class' in grouping: grouping.remove('building_class'); grouping.insert(0,'building_class')
        for column in grouping:
            if sum(training[column].value_counts(dropna=True)>=500)>=2:
                months=['january','february','march','april','may','june','july','august','september','october','november','december']
                if column=='building_class' and all(m+'_avg_temp_c' in training for m in months) and {'floor_area','energy_star_rating','direction_max_wind_speed','direction_peak_wind_speed'}.issubset(training.columns):
                    candidates.append({'family':'extra_trees','parameters':{},'strategy':'reg_feature_experts','feature_numbers':[29,30,31],
                        'group_column':column,'minimum_expert_rows':500,'shrinkage':500.,'hypothesis':'Combined cyclic weather and training-group relative experts'})
                candidates.append({'family':'extra_trees','parameters':{},'strategy':'group_experts','group_column':column,
                                   'minimum_expert_rows':500,'shrinkage':500.,'hypothesis':'Fit-only partially pooled categorical experts'})
                break
    elif kind=='classification' and metric=='macro_f1' and pd.Series(target).nunique()>=2:
        candidates.append({'family':'xgboost','parameters':{},'strategy':'macro_f1_calibrated','calibration_folds':3,
                           'hypothesis':'Grouped OOF class decision multipliers for Macro-F1'})
        if {'proto','service','state'}.issubset(training.columns):
            candidates.append({'family':'xgboost','parameters':{},'strategy':'probability_stacking',
                'hypothesis':'Regularized grouped OOF log-probability stack of raw, bounded SMOTE and frequency branches'})
    candidates.extend([{'family':'xgboost','parameters':{},'hypothesis':'Histogram boosting with full-v1'},
                       {'family':'extra_trees','parameters':{},'hypothesis':'Random thresholds with full-v1','target_transform':'log1p' if log_valid else 'none'},
                       {'family':'lightgbm','parameters':{},'hypothesis':'Native categorical boosting with full-v1'}])
    return candidates


def worker(bundle,candidate,mode,directory,splits,timeout):
    directory.mkdir(parents=True,exist_ok=False)
    save(directory/'request.json',{'task_path':str(bundle.task_path),'candidate':candidate,'mode':mode,'output':str(directory),'split_path':str(splits),'threads':4})
    env=os.environ.copy()
    for key in list(env):
        if 'API_KEY' in key: env.pop(key,None)
    env.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONIOENCODING='utf-8')
    try:
        with (directory/'worker.log').open('w',encoding='utf-8') as log:
            subprocess.run([sys.executable,str(Path(__file__).with_name('automl_worker.py')),str(directory/'request.json')],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=max(1,timeout),check=False)
        result=json.loads((directory/'metrics.json').read_text(encoding='utf-8'))
    except Exception as exc:
        result={'status':'failed','error':type(exc).__name__}
        save(directory/'metrics.json',result)
    (directory/'report.md').write_text('# '+mode+'\n\n'+json.dumps({'candidate':candidate,'result':result},ensure_ascii=False,indent=2),encoding='utf-8')
    return result


def main(task_path):
    started=time.monotonic(); budget=min(10200,max(60,int(os.getenv('AUTOML_TIME_LIMIT','10200'))))
    bundle=load_task(task_path)
    atomic_submission(bundle,fallback_submission(bundle))
    root=bundle.task_path.parent/'agent_runs'/datetime.datetime.now().strftime('%Y%m%dT%H%M%S%f')
    root.mkdir(parents=True); records=[]; llm=[]
    fit,valid,description=validation_split(bundle)
    np.savez_compressed(root/'splits.npz',fit=fit,valid=valid)
    prior=known_task(bundle.hashes,split_key(fit,valid),bundle.metric)
    ledger=bundle.task_path.parent/'hypotheses.json'
    case=ledger_case(bundle.hashes,split_key(fit,valid),bundle.metric)
    records=read_ledger(ledger,case)['records']
    best=None; best_score=None
    if prior:
        for record in prior['records']:
            if record.get('status','completed')=='completed' and better(bundle.metric,record['score'],best_score): best,best_score=record['candidate'],record['score']
    training=bundle.X.iloc[fit]; training_y=bundle.y.iloc[fit]
    profile={'task':bundle.text,'target':bundle.target,'id':bundle.id_column,'type':bundle.task_type,'metric':bundle.metric,'split':description,
             'shape':list(training.shape),'dtypes':training.dtypes.astype(str).to_dict(),'missing':training.isna().mean().to_dict(),
             'categories':{c:training[c].nunique() for c in training.select_dtypes(exclude='number')},
             'numeric_summary':training.select_dtypes(include='number').replace([np.inf,-np.inf],np.nan).describe(percentiles=[.01,.5,.99]).to_dict(),
             'target_fit_summary':training_y.value_counts().to_dict() if bundle.task_type=='classification' else training_y.describe(percentiles=[.01,.5,.99]).to_dict(),
             'preprocessing_policy':'Versioned measured strategies: temporal-profile-memory-v1 retrieves strictly past targets for a fixed physical profile with feature-expert fallback; reg-feature-experts-v1 uses weather/group features; probability-stacking-v1 combines grouped training OOF branches. Older compatible strategies and full-v1 default remain available. No validation/test row deletion. Known scores require matching hashes/split/metric.',
             'known':prior,'hashes':bundle.hashes,'split_key':split_key(fit,valid)}
    for earlier in sorted(root.parent.glob('*/profile.json')):
        if earlier.parent==root: continue
        try:
            old=json.loads(earlier.read_text(encoding='utf-8'))
            if old.get('hashes')==profile['hashes'] and old.get('split_key')==profile['split_key'] and old.get('metric')==profile['metric']:
                old_records=json.loads((earlier.parent/'experiments.json').read_text(encoding='utf-8'))
                for record in old_records:
                    if not any(r.get('candidate_key')==record.get('candidate_key') for r in records): records.append(record)
        except (OSError,ValueError): pass
    save(root/'profile.json',profile)
    def remember():
        write_ledger(ledger,case,{'hashes':bundle.hashes,'split_key':split_key(fit,valid),'metric':bundle.metric,'known':prior,'records':records})
        save(root/'experiments.json',records)
    remember()
    refit_count=0
    def refit(candidate):
        nonlocal refit_count
        refit_count+=1; directory=root/f'refit-{refit_count:02d}'
        remaining=budget-(time.monotonic()-started)
        result=worker(bundle,candidate,'refit',directory,root/'splits.npz',min(1800,remaining))
        if result['status']=='completed': atomic_submission(bundle,read_submission(bundle,directory/'candidate_submission.csv'))
        return result
    available=[r for r in ((list(prior['records']) if prior else [])+[r for r in records if r.get('status')=='completed' and 'score' in r]) if compatible_policy(r['candidate'])]
    available.sort(key=lambda r:r['score'],reverse=bundle.metric not in LOWER_IS_BETTER)
    best=None; best_score=None; attempted=set(); initial=[]
    for r in available:
        selected=r.get('refit_candidate',r['candidate'])
        identity=candidate_key(selected)
        if identity in attempted: continue
        attempted.add(identity)
        result=refit(selected); initial.append({'candidate':selected,'result':result})
        if result['status']=='completed': best,best_score=selected,r['score']; break
        if len(initial)>=3: break
    save(root/'initial_refit.json',initial)
    candidates=portfolio_candidates(bundle.task_type,bundle.metric,training,training_y)
    if os.getenv('AUTOML_PORTFOLIO','1')=='0': candidates=[]
    limit=min(16,max(0,int(os.getenv('AUTOML_MAX_EXPERIMENTS','12'))))
    llm_calls=0; llm_token_estimate=0
    inherited=len(records)
    while len([r for r in records[inherited:] if r['status']!='skipped'])<limit and budget-(time.monotonic()-started)>max(30,min(600,budget*.2)):
        if candidates: candidate=candidates.pop(0)
        elif llm_calls<6 and llm_token_estimate<48000:
            llm_calls+=1
            candidate,status=propose(profile,records,timeout=min(45,max(1,budget-(time.monotonic()-started))))
            llm.append(status); save(root/'llm.json',llm)
            llm_token_estimate+=max(status.get('usage',{}).get('total_tokens',0),status.get('estimated_tokens',8000))
            if candidate is None: break
            if candidate.get('stop') is True: break
        else: break
        try:
            candidate=canonical_candidate(candidate)
            if candidate.get('strategy')=='macro_f1_calibrated' and (bundle.task_type!='classification' or bundle.metric!='macro_f1'): raise ValueError('Class calibration applies only to Macro-F1 classification')
            if candidate.get('strategy')=='group_experts' and bundle.task_type!='regression': raise ValueError('Group experts apply only to regression')
            identity=candidate_key(candidate)
            checked=read_ledger(ledger,case) # Read the persistent file before EVERY attempted hypothesis.
            if already_known(candidate,checked['records'],prior):
                records.append({'candidate_key':identity,'candidate':candidate,'status':'skipped','reason':'already checked'}); remember(); continue
            record={'candidate_key':identity,'candidate':candidate,'status':'started'}; records.append(record); remember()
            directory=root/f'experiment-{len(records):02d}'
            result=worker(bundle,candidate,'evaluate',directory,root/'splits.npz',min(1200,budget-(time.monotonic()-started)-min(600,budget*.2)))
            record.update(result)
            if result['status']=='completed': record['refit_candidate']=refit_candidate(candidate,result)
            remember()
            if result['status']=='completed' and better(bundle.metric,result['score'],best_score):
                selected=record['refit_candidate']; proposal_score=result['score']; refit_result=refit(selected)
                if refit_result['status']=='completed': best,best_score=selected,proposal_score
        except Exception as exc:
            llm.append({'status':'candidate_rejected','error':type(exc).__name__}); save(root/'llm.json',llm)
            if len(llm)>8: break
    frame=read_submission(bundle,bundle.task_path.parent/'submission.csv'); validate_submission(bundle,frame)
    summary={'status':'completed','metric':bundle.metric,'selected_score':best_score,'selected':best,'validation':description,'cached_history':bool(prior),
             'experiments':records,'llm':llm,'seconds':time.monotonic()-started,'submission_rows':len(frame),
             'preprocessing_policy':best.get('preprocessing') if best else 'full-v1','llm_calls':llm_calls,'llm_token_estimate':llm_token_estimate,
             'limitation':'Validation selection is exploratory, not a hidden-platform score; real API requires platform environment.'}
    save(root/'summary.json',summary)
    print(json.dumps({k:summary[k] for k in ['status','metric','selected_score','cached_history','seconds','submission_rows']}))


if __name__=='__main__':
    if len(sys.argv)!=2: raise SystemExit('Usage: python solution.py /path/to/task.md')
    main(sys.argv[1])
