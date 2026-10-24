"""Promote only independently verified improvements; preserve previous evidence."""
import json,pprint
from pathlib import Path
from automl_models import canonical_candidate
from automl_known import KNOWN_TASKS
from automl_history import candidate_key
ROOT=Path(__file__).resolve().parents[1]
def main():
    suite=ROOT/'results/2026-10-23/series-combinations-v1'
    assert json.loads((suite/'verification.json').read_text(encoding='utf-8'))['status']=='passed'
    comparison=json.loads((suite/'comparison.json').read_text(encoding='utf-8'))
    records=json.loads((ROOT/'results/spec008-attempts.json').read_text(encoding='utf-8'))
    attempts=[{'number':r['job']['number'],'task':r['job']['task'],'candidate':r['job']['candidate'],
       'fingerprint':r['fingerprint'],'status':r['status'],'metric':r.get('metric'),'score':r.get('score'),'source':r['directory']+'/report.md'} for r in records]
    winners=[];baseline={}
    for task,metric in [('regression','rmse'),('multiclass','macro_f1')]:
        prior=next(t for t in KNOWN_TASKS if t['metric']==metric)
        if task in comparison['best']:
            winner=comparison['best'][task];c=canonical_candidate({**winner['candidate'],**winner['learned_state'],'hypothesis':winner['hypothesis']+': verified combination; frozen train-only state'})
            record={'metric':metric,'score':winner['score'],'candidate':c,'source':winner['directory']+'/report.md'}
            winners.append(record);baseline[task]={**record,'hypothesis':winner['hypothesis'],'previous_best':comparison['baseline'][task]}
        else:
            key=(min if metric=='rmse' else max)([r for r in prior['records'] if r['candidate'].get('strategy')],key=lambda r:r['score'])
            baseline[task]={**key,'metric':metric,'previous_best':comparison['baseline'][task],'hypothesis':'unchanged incumbent'}
    path=ROOT/'automl_known.py';content=path.read_text(encoding='utf-8')
    marker='\n# SPEC-008 verified combinations; retain every attempted negative graph.\n'
    if marker in content:raise RuntimeError('Already embedded, do not rewrite history')
    content+=marker+'SPEC008_ATTEMPTS = '+pprint.pformat(attempts,width=120,sort_dicts=False)+'\n_COMBINED_WINNERS = '+pprint.pformat(winners,width=120,sort_dicts=False)+'\n'
    content+='''\nfor _task in KNOWN_TASKS:
    _kind = 'regression' if _task['metric']=='rmse' else 'multiclass'
    _task['spec008_attempts'] = [r for r in SPEC008_ATTEMPTS if r['task']==_kind]
    _task['previous_directions'] += [f"HYP-{r['number']:03d}: combination; {r['status']}; fingerprint={r['fingerprint']}" for r in _task['spec008_attempts']]
    _task['records'] += [{k:v for k,v in r.items() if k!='metric'} for r in _COMBINED_WINNERS if r['metric']==_task['metric']]
'''
    path.write_text(content,encoding='utf-8')
    (ROOT/'baseline.json').write_text(json.dumps(baseline,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(baseline,ensure_ascii=False))
if __name__=='__main__':main()
