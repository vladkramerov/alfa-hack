"""Preserve measured histories, add new evidence without training anything."""
import copy
import json
from pathlib import Path
import pprint
from automl_history import candidate_key
from automl_known import KNOWN_TASKS

ROOT=Path(__file__).resolve().parents[1]


def main():
    history=copy.deepcopy(KNOWN_TASKS)
    old=json.loads((ROOT/'results/2026-10-23/series-ten-hypotheses-v1/records.json').read_text(encoding='utf-8'))
    full=json.loads((ROOT/'results/2026-10-23/full-preprocessing-records.json').read_text(encoding='utf-8'))
    for task in history:
        for row in old:
            if row['hashes']['train.csv']!=task['hashes']['train.csv'] or row['status']!='completed': continue
            c=row['candidate']
            # Only directly supported configurations are executable in this agent.
            if (c['hypothesis'],c['task']) not in [('HYP-012','regression'),('HYP-011','multiclass')]: continue
            params={k:v for k,v in c['parameters'].items() if k not in ['objective','n_jobs','random_state']}
            candidate={'family':c['family'],'parameters':params,'preprocessing':'legacy-baseline','scale':False,
                       'target_transform':'log1p' if c['hypothesis']=='HYP-012' else None,'hypothesis':c['hypothesis']+' historical measured score, not full-v1'}
            entry={'candidate':candidate,'score':row['score'],'source':row['directory']+'/evidence.json'}
            if not any(candidate_key(r['candidate'])==candidate_key(candidate) for r in task['records']): task['records'].append(entry)
        for row in full:
            if row['hashes']['train.csv']!=task['hashes']['train.csv'] or row['status']!='completed': continue
            if not any(candidate_key(r['candidate'])==candidate_key(row['candidate']) for r in task['records']):
                task['records'].append({'candidate':row['candidate'],'score':row['score'],'source':row['directory']+'/evidence.json'})
        task['previous_directions'].append('HYP011-020 completed; HYP021 full-v1 scored once per task; never repeat successful evaluations')
    (ROOT/'automl_known.py').write_text('"""Measured evidence; old and full-v1 policies remain separate. No metric was retrained."""\nKNOWN_TASKS = '+pprint.pformat(history,width=120,sort_dicts=False)+'\n',encoding='utf-8')
    print('Embedded new compatible history; no model fits performed.')


if __name__=='__main__': main()
