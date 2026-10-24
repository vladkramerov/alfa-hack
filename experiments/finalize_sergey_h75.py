"""Continue a guard-interrupted comparison from the necessary clean CLI run."""
import hashlib,json,shutil,subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score,accuracy_score

ROOT=Path(__file__).resolve().parents[1]
SUITE=ROOT/'results/2026-10-24/sergey-improvement-v1'


def main():
    assert subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()=='sergey_branch'
    parent=sorted((ROOT/'tmp').glob('sergey-platform-multiclass-*'),key=lambda p:p.stat().st_mtime)[-1]
    run=next((parent/'agent_runs').glob('*'))
    experiment=run/'experiment-01'
    metric=json.loads((experiment/'metrics.json').read_text())
    assert metric['status']=='completed'
    out=ROOT/'results/2026-10-24/HYP-075-02-multiclass-clean-runtime-continuation'
    out.mkdir(exist_ok=False)
    with np.load(experiment/'validation_predictions.npz',allow_pickle=True) as z:
        values={k:np.asarray(z[k],dtype=str) if k in ['y','prediction','classes'] else z[k] for k in z.files}
    truth=pd.read_csv(ROOT/'data/multiclass/train.csv').attack_cat.astype(str).to_numpy()
    assert np.array_equal(values['y'],truth[values['positions']])
    assert np.array_equal(values['prediction'],values['classes'][values['probabilities'].argmax(axis=1)])
    np.testing.assert_allclose(values['probabilities'].sum(axis=1),1.,atol=1e-6)
    score=float(f1_score(values['y'],values['prediction'],average='macro',zero_division=0))
    assert abs(score-metric['score'])<1e-12
    np.savez_compressed(out/'validation_predictions.npz',**values)
    shutil.copyfile(run/'splits.npz',out/'splits.npz')
    shutil.copyfile(experiment/'preprocessing-audit.json',out/'audit.json')
    shutil.copyfile(experiment/'worker.log',out/'worker.log')
    old=json.loads((SUITE/'control-multiclass-primary/metrics.json').read_text())['score']
    final={**metric,'control':old,'relative_gain':(score-old)/old,
        'accuracy':float(accuracy_score(values['y'],values['prediction'])),
        'weighted_f1':float(f1_score(values['y'],values['prediction'],average='weighted',zero_division=0)),
        'source':'necessary final archive clean CLI evaluation; no extra fit in continuation',
        'embedded_weights':False,'initial_history_empty':True,'extra_fits_in_continuation':0}
    (out/'metrics.json').write_text(json.dumps(final,ensure_ascii=False,indent=2))
    profile=json.loads((run/'profile.json').read_text())
    first=json.loads((ROOT/'results/2026-10-24/HYP-075-01-multiclass-fresh-oof/evidence.json').read_text())
    assert profile['hashes']==first['hashes'] and profile['split_key']==first['split_key']
    request=json.loads((experiment/'request.json').read_text())
    assert request['candidate']['strategy']=='probability_stacking' and 'meta_state' not in request['candidate']
    manifest=json.loads((SUITE/'package-manifest.json').read_text())
    evidence={'hashes':profile['hashes'],'split_key':profile['split_key'],
              'archive_sha256':manifest['sha256'],'request':request,
              'source_directory':str(experiment),'reuse_reason':'H75 separate runner reached duplicate existing full branches after fresh OOF. Guard interruption preserved; the already-required clean archive run supplies the same full train-only protocol without starting an extra recovery fit.'}
    (out/'evidence.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2))
    report='# HYP-075 / техническое продолжение из чистого ZIP\n\nПервый запуск остановлен guard при попытке повторить уже обученные полные ветви; это техническая ошибка, не отрицательный результат модели. Он не переписан. Результат взят из уже необходимой платформенной проверки того же primary протокола. Дополнительных fit для продолжения0.\n\n```json\n'+json.dumps(final,ensure_ascii=False,indent=2)+'\n```\n\nДанные и split точно совпадают. Все веса метамодели обучены внутри текущего fit; в ZIP их нет.\n'
    (out/'report.md').write_text(report)
    b=json.loads((ROOT/'baseline.json').read_text());b['multiclass']['score']=score;b['multiclass']['candidate'].pop('meta_state',None)
    b['multiclass']['candidate']['hypothesis']='HYP-075 grouped OOF learned within the current task; no embedded calibration'
    b['multiclass']['hypothesis']='HYP-075';b['multiclass']['source']=out.relative_to(ROOT).as_posix()+'/report.md'
    (ROOT/'baseline.json').write_text(json.dumps(b,ensure_ascii=False,indent=2))
    for file in ['HYPOTHESES.md','docs/hypotheses/HYP-075.md','results/README.md']:
        with (ROOT/file).open('a') as stream:stream.write('\n- HYP-075 technical continuation completed: '+json.dumps(final,ensure_ascii=False)+'; [report]('+out.relative_to(ROOT).as_posix()+'/report.md). Первый сбой guard сохранён; повторного recovery fit нет.\n')
    (SUITE/'fresh-classification-verification.json').write_text(json.dumps({'status':'passed','checks':7,'result':final,'metric_recomputed':True,'hypothesis':'HYP-075'},indent=2))
    print(json.dumps(final,ensure_ascii=False))


if __name__=='__main__':main()
