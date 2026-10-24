"""Only independently verified strict improvements enter the measured baseline."""
import json,pprint
from pathlib import Path
from automl_models import canonical_candidate
ROOT=Path(__file__).resolve().parents[1]
def main():
    suite=ROOT/'results/2026-10-24/series-alternatives-v1';verification=json.loads((suite/'verification.json').read_text(encoding='utf8'))
    assert verification['status']=='passed'
    comparison=json.loads((suite/'comparison.json').read_text(encoding='utf8'));rows=json.loads((ROOT/'results/spec009-attempts.json').read_text(encoding='utf8'))
    attempts=[{'number':r['job']['number'],'task':r['job']['task'],'candidate':r['candidate'],'fingerprint':r['fingerprint'],
       'status':r['status'],'score':r.get('score'),'metric':r.get('metric'),'source':r['directory']+'/report.md'} for r in rows]
    baseline=json.loads((ROOT/'baseline.json').read_text(encoding='utf8'));winners=[]
    for task,win in comparison['best'].items():
        family='extra_trees' if win['candidate']['strategy']=='temporal_profile_memory' else None
        if family is None:raise ValueError('Portable promotion for this new architecture not verified')
        candidate=canonical_candidate({**win['candidate'],**win['learned_state'],'family':family,'hypothesis':win['hypothesis']+': verified past-only profile retrieval, final refit only'})
        record={'metric':win['metric'],'score':win['score'],'candidate':candidate,'source':win['directory']+'/report.md'}
        winners.append(record);baseline[task]={**record,'hypothesis':win['hypothesis'],'previous_best':comparison['baseline'][task]}
    path=ROOT/'automl_known.py';content=path.read_text(encoding='utf8');marker='\n# SPEC-009 alternative outcomes and independently verified portable winners.\n'
    if marker in content:raise RuntimeError('Already embedded; do not rewrite history')
    content+=marker+'SPEC009_ATTEMPTS = '+pprint.pformat(attempts,width=120,sort_dicts=False)+'\n_ALTERNATIVE_WINNERS = '+pprint.pformat(winners,width=120,sort_dicts=False)+'\n'
    content+='''\nfor _task in KNOWN_TASKS:
    _kind = 'regression' if _task['metric']=='rmse' else 'multiclass'
    _task['spec009_attempts'] = [r for r in SPEC009_ATTEMPTS if r['task']==_kind]
    _task['previous_directions'] += [f"HYP-{r['number']:03d}: alternative; {r['status']}; fingerprint={r['fingerprint']}" for r in _task['spec009_attempts']]
    _task['records'] += [{k:v for k,v in r.items() if k!='metric'} for r in _ALTERNATIVE_WINNERS if r['metric']==_task['metric']]
'''
    path.write_text(content,encoding='utf8');(ROOT/'baseline.json').write_text(json.dumps(baseline,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({'status':'promoted','baseline':{k:(r['candidate']['strategy'],r['score']) for k,r in baseline.items()}},ensure_ascii=False))
if __name__=='__main__':main()
