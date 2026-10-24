"""Promote only independently verified SPEC011 outer winners; no model fits."""
import hashlib
import json
import os
import math
from pathlib import Path
import pprint

from automl_models import canonical_candidate
from automl_history import candidate_key,compatible_policy
from research_memory import atomic_json,timestamp

ROOT=Path(__file__).resolve().parents[1]
SUITE=ROOT/'results/2026-10-24/series-three-hours-v1'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def paired_write(baseline_path,known_path,baseline,known,backup_dir):
    """Preserve both originals and restore the pair if either replacement fails."""
    before={Path(baseline_path):Path(baseline_path).read_bytes(),Path(known_path):Path(known_path).read_bytes()}
    backups={Path(baseline_path):Path(backup_dir)/'baseline-before-promotion.json',
             Path(known_path):Path(backup_dir)/'automl-known-before-promotion.py'}
    for path in backups.values():assert not path.exists()
    for path,content in before.items():backups[path].write_bytes(content)
    pending={Path(baseline_path):json.dumps(baseline,ensure_ascii=False,indent=2).encode('utf8'),
             Path(known_path):known.encode('utf8')}
    temporary=[]
    try:
        for path,content in pending.items():
            tmp=path.with_name(path.name+'.promotion-tmp')
            assert not tmp.exists()
            tmp.write_bytes(content);temporary.append((tmp,path))
        for tmp,path in temporary:os.replace(tmp,path)
    except Exception:
        for path,content in before.items():path.write_bytes(content)
        raise
    finally:
        for tmp,_ in temporary:
            if tmp.exists():tmp.unlink()


def main():
    pointer=json.loads((SUITE/'verification.json').read_text(encoding='utf8'))
    verification=ROOT/pointer['latest_verification']
    assert sha(verification)==pointer['sha256'] and pointer['passed']
    verified=json.loads(verification.read_text(encoding='utf8'))
    assert verified['passed'] and verified['all_attempts_terminal'] and verified['all_completed_runs_verified']
    adapter=ROOT/'results/2026-10-24/sprint-pool-adapter-verification/verification.json'
    replay=json.loads(adapter.read_text(encoding='utf8'))
    assert replay['status']=='passed' and replay['model_fits']==0 and not replay['fit_attempts']
    for name,digest in replay['source_sha256'].items():assert sha(ROOT/name)==digest, name
    for name,digest in replay['artifact_sha256'].items():assert sha(ROOT/name)==digest, name
    old=json.loads((ROOT/'baseline.json').read_text(encoding='utf8'))
    assert old['regression']['hypothesis']=='HYP-069' and old['multiclass']['hypothesis']=='HYP-048'
    rollback=SUITE/'baseline-before-promotion.json'
    assert not rollback.exists()
    results=ROOT/'results/2026-10-24'
    pool=json.loads((results/'HYP-091-02-outer-multiblend-sprint/metrics.json').read_text(encoding='utf8'))
    decision=json.loads((results/'HYP-078-02-outer-default-sprint/audit.json').read_text(encoding='utf8'))
    with __import__('numpy').load(results/'HYP-078-02-outer-default-sprint/validation_predictions.npz',allow_pickle=False) as z:
        classes=z['classes'].tolist()
    reg=canonical_candidate({'strategy':'temporal_transport_pool','family':'extra_trees','parameters':{},
        'pool_weights':pool['weights'],'hypothesis':'HYP-091: inner-frozen temporal transport pool; final refit only'})
    cls=canonical_candidate({'strategy':'proper_macro_f1_stacking','family':'xgboost','parameters':{},
        'meta_state':old['multiclass']['candidate']['meta_state'],
        'decision_state':{'classes':classes,'theta':decision['theta'],
            'gain':decision['decision']['gain'],'offset':decision['decision']['offset']},
        'hypothesis':'HYP-078: proper posterior calibration and frozen expected Macro-F1 decisions; final refit only'})
    assert reg['pool_weights']==replay['checks']['H91']['frozen_inner_weights']
    assert cls['decision_state']==replay['checks']['H78']['frozen_state']
    measured_meta=json.loads((ROOT/'results/2026-10-23/HYP-048-01-multiclass-combinations/learned-state.json').read_text(encoding='utf8'))
    assert cls['meta_state']==measured_meta.get('meta_state',measured_meta)
    target=verified['target']['tasks']
    wanted={'regression':('rmse',reg,'HYP-091','HYP-091-02-outer-multiblend-sprint'),
            'multiclass':('macro_f1',cls,'HYP-078','HYP-078-02-outer-default-sprint')}
    new={}
    for kind,(metric,candidate,hypothesis,directory) in wanted.items():
        best=target[metric]['best_new_outer']
        assert best['directory'].replace('\\','/').endswith(directory),best
        assert math.isfinite(best['score']) and (best['score']<old[kind]['score'] if metric=='rmse' else best['score']>old[kind]['score'])
        assert compatible_policy(candidate)
        new[kind]={'metric':metric,'score':best['score'],'candidate':candidate,
            'source':f'results/2026-10-24/{directory}/report.md','hypothesis':hypothesis,
            'previous_best':old[kind]['score'],'relative_gain':target[metric]['relative_gain'],
            'assessment':'Repeatedly viewed public outer; exploratory, hidden platform score unknown'}
    new['verification']={'path':str(verification.relative_to(ROOT)).replace('\\','/'),'sha256':sha(verification)}
    new['adapter_verification']={'path':str(adapter.relative_to(ROOT)).replace('\\','/'),'sha256':sha(adapter)}
    new['goal']={'required_relative_improvement_each':.2,'both_targets_met':verified['target']['both_20pct_targets_met'],
                 'fixed_original_controls':{kind:old[kind]['score'] for kind in wanted},'created':timestamp()}
    attempts=[]
    for run in verified['runs']:
        directory=ROOT/run['directory']
        metrics=json.loads((directory/'metrics.json').read_text(encoding='utf8'))
        number=int(metrics.get('hypothesis',directory.name[:7]).split('-')[1])
        metric=metrics.get('metric')
        kind='regression' if metric=='rmse' else 'multiclass'
        strategy={70:'entity_linked_history',78:'proper_macro_f1_stacking',91:'temporal_transport_pool'}.get(number,f'spec011_hyp_{number:03d}')
        attempts.append({'number':number,'task':kind,'stage':metrics.get('stage'),
            'status':metrics['status'],'metric':metric,'score':metrics.get('score'),
            'candidate':{'strategy':strategy},'source':str((directory/'report.md').relative_to(ROOT)).replace('\\','/')})
    path=ROOT/'automl_known.py';source=path.read_text(encoding='utf8')
    assert 'SPEC011_ATTEMPTS =' not in source
    winners=[{k:v for k,v in new[kind].items() if k in ['metric','score','candidate','source']} for kind in wanted]
    for record in winners:record['candidate_key']=candidate_key(record['candidate'])
    addition='\n\n# SPEC011 verified public evidence; no historical validation fits.\nSPEC011_ATTEMPTS = '+pprint.pformat(attempts,width=115,sort_dicts=False)
    addition+='\n_SPRINT_WINNERS = '+pprint.pformat(winners,width=115,sort_dicts=False)
    addition+='\n\nfor _task in KNOWN_TASKS:\n    _kind = "regression" if _task["metric"]=="rmse" else "multiclass"\n    _task["spec011_attempts"] = [r for r in SPEC011_ATTEMPTS if r["task"]==_kind]\n    _task["previous_directions"] += [f"HYP-{r[\'number\']:03d}: SPEC011 {r[\'stage\']}; {r[\'status\']}" for r in _task["spec011_attempts"]]\n    _task["records"] += [{k:v for k,v in r.items() if k!="metric"} for r in _SPRINT_WINNERS if r["metric"]==_task["metric"]]\n'
    compile(source+addition,str(path),'exec')
    paired_write(ROOT/'baseline.json',path,new,source+addition,SUITE)
    atomic_json(SUITE/'promotion.json',{'created':timestamp(),'old_baseline_sha256':sha(rollback),
        'baseline_sha256':sha(ROOT/'baseline.json'),'known_source_sha256':sha(path),
        'model_fits':0,'winners':winners,'target_met':new['goal']['both_targets_met']})
    print(json.dumps({'promoted':[(kind,new[kind]['score']) for kind in wanted],'both_20pct_targets_met':new['goal']['both_targets_met']}))


if __name__=='__main__':main()
