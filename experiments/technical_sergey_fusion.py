"""Delivery-only full refit; no validation quality measurement or new search."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
DELIVERY=ROOT/'deliveries/sergey_ivan_fusion_v1'
OUT=ROOT/'results/2026-10-24/HYP-107-03-technical-delivery-refit'
sys.path.insert(0,str(DELIVERY))


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,default=str),encoding='utf8')


def main():
    started=time.monotonic()
    # This is the necessary technical full refit for submission prediction,
    # not a repeated hypothesis evaluation; no holdout quality is calculated.
    assert (ROOT/'HYPOTHESES.md').read_text(encoding='utf8')
    assert (ROOT/'tmp/sergey-24e3311f6ba9/HYPOTHESES.md').read_text(encoding='utf8')
    OUT.mkdir()
    save(OUT/'request.json',{'mode':'technical_full_refit_only','quality_metric_computed':False,
        'sources':{p.name:sha(p) for p in DELIVERY.iterdir() if p.is_file()},
        'ledger_sha256':sha(ROOT/'HYPOTHESES.md'),'budget_seconds':900,'learner_threads':4})
    os.environ.update(COMBO_RUN_DIR=str(OUT/'fits'),COMBO_LEDGER_PATH=str(ROOT/'HYPOTHESES.md'),
        COMBO_HYPOTHESIS='HYP-107-technical-full-refit',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
    try:
        import joblib
        import numpy as np
        import automl_models
        from automl_data import load_task,make_submission,atomic_submission,read_submission,validate_submission
        assert Path(automl_models.__file__).parent==DELIVERY
        bundle=load_task(ROOT/'data/regression/task.md')
        model=automl_models.build_pipeline('regression',{'family':'extra_trees','strategy':'sergey_ivan_fusion','parameters':{}},4)
        model.fit(bundle.X,bundle.y)
        prediction=model.predict(bundle.X_test)
        assert prediction.shape==(len(bundle.test),) and np.isfinite(prediction).all()
        path=OUT/'submission.csv'
        atomic_submission(bundle,make_submission(bundle,prediction),path)
        validate_submission(bundle,read_submission(bundle,path))
        joblib.dump(model,OUT/'model.joblib',compress=3)
        save(OUT/'preprocessing-audit.json',model.audit_)
        np.savez_compressed(OUT/'predictions.npz',prediction=prediction)
        result={'status':'completed','mode':'technical_full_refit_only','quality_metric_computed':False,
            'train_rows':len(bundle.train),'test_rows':len(bundle.test),'submission_validated':True,
            'weights_learned_on_current_train':model.weights_.tolist(),'seconds':time.monotonic()-started,
            'model_sha256':sha(OUT/'model.joblib'),'submission_sha256':sha(path),
            'learner_source_directory':str(DELIVERY),'known_tasks_embedded':False}
        save(OUT/'outcome.json',result)
        (OUT/'report.md').write_text('# H107: технический полный refit\n\n'
            'Команда: .venv-ml/Scripts/python.exe -X utf8 experiments/technical_sergey_fusion.py.\n'
            'Обучение переносимого графа на полном train для test submission. Внутренние веса обучаются на данных текущего train.\n'
            'Public holdout метрики не пересчитывались; этот запуск не является новым доказательством качества.\n'
            +json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
        print(json.dumps(result,ensure_ascii=False),flush=True)
    except Exception as error:
        save(OUT/'outcome.json',{'status':'failed','error':repr(error),'seconds':time.monotonic()-started})
        raise


if __name__=='__main__':main()
