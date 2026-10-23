"""Clean ZIP/mock HTTP protocol checks on synthetic data; no public fits.

Usage: python -m experiments.verify_flat_package [--archive PATH]
The full-v1 archive is preferred; an older-named archive may also be supplied.
Each invocation preserves all earlier acceptance directories and reports.
"""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import http.server
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback
import zipfile
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
MODULES={'solution.py','automl_data.py','automl_models.py','automl_history.py',
         'automl_known.py','automl_llm.py','automl_worker.py','automl_preprocessing.py','requirements.txt'}


class Mock(http.server.BaseHTTPRequestHandler):
    mode='tool'; calls=0
    def log_message(self,*args): pass
    def do_POST(self):
        type(self).calls+=1
        request=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        assert self.path=='/v1/chat/completions' and request['tools'][0]['function']['name']=='run_experiment'
        if type(self).mode=='empty': payload={'choices':[]}
        else:
            action={'family':'extra_trees','parameters':{'n_estimators':12,'max_depth':6},'hypothesis':'Synthetic interaction protocol check',
                    'preprocessing':'legacy_standard','scale':False,'max_outlier_fraction':.9,
                    'feature_code':'def engineer(X):\n    X = X.copy()\n    X["signal_square"] = X["signal"] ** 2\n    return X\n'}
            payload={'choices':[{'finish_reason':'tool_calls','message':{'tool_calls':[{'id':'local-1','type':'function','function':{'name':'run_experiment','arguments':json.dumps(action)}}]}}]}
        body=json.dumps(payload).encode(); self.send_response(200); self.send_header('Content-Type','application/json'); self.end_headers(); self.wfile.write(body)


def fixture(folder,classification,metric=None):
    nested=folder/'nested/inputs'; nested.mkdir(parents=True)
    rng=np.random.default_rng(91); signal=rng.normal(size=100)
    target=np.where(signal>0,'positive','negative') if classification else signal**2+.1*rng.normal(size=100)
    # Compute y before introducing feature missingness; no missing target is created.
    signal[3]=np.nan
    kinds=np.array(['a','b']*50,dtype=object); kinds[::11]=None; kinds[5]='__MISSING__'
    pd.DataFrame({'object_key':np.arange(100),'signal':signal,'kind':kinds,'response':target}).to_csv(nested/'train.csv',index=False)
    pd.DataFrame({'object_key':[1000,1001,1002],'signal':[-1.,np.nan,np.inf],'kind':['new','a',None]}).to_csv(nested/'test.csv',index=False)
    pd.DataFrame({'object_key':[1000,1001,1002],'response':['positive']*3 if classification else [0.]*3}).to_csv(nested/'sample_submission.csv',index=False)
    pd.DataFrame({'feature':['object_key','signal','kind','response'],'role':['id','feature','feature','target']}).to_csv(nested/'data_dictionary.csv',index=False)
    declaration=('Classification.\nPrimary metric: Macro F1.' if classification else 'Regression.\nPrimary metric: RMSE.')
    if metric=='mae': declaration='Regression.\nPrimary metric: MAE.\nExample metric: RMSE is diagnostic only.'
    task=folder/'task.md'; task.write_text(declaration+'\nData: nested/inputs/train.csv nested/inputs/test.csv nested/inputs/sample_submission.csv nested/inputs/data_dictionary.csv',encoding='utf-8')
    return task


def check_canonical(candidate):
    assert candidate['preprocessing']=='full-v1',candidate
    assert candidate['scale'] is True,candidate
    assert candidate['max_outlier_fraction']==.05,candidate


def check_workers(run):
    """Inspect actual clean-extraction child artifacts without importing its code."""
    completed=[]
    for path in sorted(run.glob('*/metrics.json')):
        metric=json.loads(path.read_text(encoding='utf-8'))
        if metric['status']!='completed': continue
        request=json.loads((path.parent/'request.json').read_text(encoding='utf-8'))
        check_canonical(request['candidate'])
        audit=json.loads((path.parent/'preprocessing-audit.json').read_text(encoding='utf-8'))
        assert audit['policy']=='full-v1'
        assert audit['preprocessing']['normalization_enabled'] is True
        assert audit['outliers']['max_fraction']==.05
        original=audit['fit_original_rows']; retained=audit['fit_retained_rows']
        assert 0<=original-retained<=int(np.floor(.05*original))
        for counts in audit['outliers']['classes'].values():
            assert counts['removed']<=int(np.floor(.05*counts['before']))
            assert counts['retained']>=min(50,counts['before'])
        assert audit['normalized_training_mean_abs_max']<1e-5
        assert audit['normalized_training_std_error_max']<1e-5
        with np.load(path.parent/'training-filter.npz') as mask:
            keep=mask['keep_mask']; removed=mask['removed_positions']
            assert keep.dtype==np.bool_ and keep.shape==(original,)
            assert int(keep.sum())==retained
            np.testing.assert_array_equal(removed,np.flatnonzero(~keep))
        if request['mode']=='refit':
            assert metric['test_rows']==3
            submission=pd.read_csv(path.parent/'candidate_submission.csv')
            assert len(submission)==3 and submission.object_key.tolist()==[1000,1001,1002]
        else:
            # The fixture is local and trusted; classification y can be an object array.
            with np.load(path.parent/'validation_predictions.npz',allow_pickle=True) as values:
                assert len(values['prediction'])==metric['valid_rows']
                assert len(values['positions'])==metric['valid_rows']
                if metric['metric']=='mae':
                    expected=float(np.mean(np.abs(values['y']-values['prediction'])))
                    assert np.isclose(expected,metric['score'],rtol=1e-12)
        completed.append({'directory':path.parent.name,'mode':request['mode'],
                          'original_rows':original,'retained_rows':retained,'policy':'full-v1'})
    return completed


def default_archive():
    current=ROOT/'dist/solution_full_preprocessing.zip'
    return current if current.exists() else ROOT/'dist/solution_ivan_branch.zip'


def main(archive_path=None):
    archive_path=Path(archive_path).resolve() if archive_path else default_archive()
    date=datetime.now(timezone(timedelta(hours=3))).strftime('%Y-%m-%d')
    parent=ROOT/'results'/date/('platform-verification-'+str(time.time_ns())); parent.mkdir(parents=True,exist_ok=False)
    extracted=ROOT/'tmp'/('flat-'+str(time.time_ns())); extracted.mkdir(parents=True)
    with zipfile.ZipFile(archive_path) as archive:
        optional={'automl_advanced.py','automl_combo_memory.py','automl_feature_sets.py','automl_reg_combo.py','automl_class_combo.py','automl_combo_policy.py','automl_temporal_memory.py'}
        expected_modules=MODULES | (optional & set(archive.namelist()))
        assert len(archive.namelist())==len(expected_modules) and set(archive.namelist())==expected_modules
        assert all('/' not in name and '\\' not in name for name in archive.namelist())
        assert archive.read('requirements.txt')==(ROOT/'requirements.txt').read_bytes()
        archive.extractall(extracted)
    assert 0<archive_path.stat().st_size<100_000_000
    Mock.calls=0
    server=http.server.ThreadingHTTPServer(('127.0.0.1',0),Mock)
    thread=threading.Thread(target=server.serve_forever,daemon=True); thread.start()
    results=[]; failure=None
    try:
        for name,classification,mode,metric in [('regression-tool',False,'tool','rmse'),('classification-tool',True,'tool','macro_f1'),('regression-api-failure',False,'empty','rmse'),('primary-mae-example-rmse',False,'tool','mae')]:
            task=fixture(parent/name,classification,metric); Mock.mode=mode
            env=os.environ.copy(); env.update(OPENAI_BASE_URL=f'http://127.0.0.1:{server.server_port}/v1',OPENAI_API_KEY='local-mock-only',OPENAI_MODEL='local-mock',AUTOML_PORTFOLIO='0',AUTOML_MAX_EXPERIMENTS='1',AUTOML_TIME_LIMIT='120')
            env.pop('PYTHONPATH',None)
            completed=subprocess.run([sys.executable,str(extracted/'solution.py'),str(task)],cwd=extracted,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=120,text=True,encoding='utf-8')
            (parent/(name+'.log')).write_text(completed.stdout,encoding='utf-8')
            assert completed.returncode==0,completed.stdout
            summary_path=next((task.parent/'agent_runs').glob('*/summary.json'))
            summary=json.loads(summary_path.read_text(encoding='utf-8'))
            assert summary['metric']==metric and summary['preprocessing_policy']=='full-v1'
            ledger=json.loads((task.parent/'hypotheses.json').read_text(encoding='utf-8'))
            assert len(ledger['cases'])==1
            if mode=='tool': assert next(iter(ledger['cases'].values()))['records'][0]['status']=='completed'
            output=pd.read_csv(task.parent/'submission.csv'); assert list(output)==['object_key','response'] and len(output)==3
            assert output.object_key.tolist()==[1000,1001,1002]
            assert output.response.notna().all()
            if not classification: assert np.isfinite(output.response).all()
            audits=check_workers(summary_path.parent)
            if mode=='tool':
                assert len(summary['experiments'])==1 and summary['experiments'][0]['status']=='completed'
                assert summary['selected']['feature_code'] and summary['selected_score'] is not None
                check_canonical(summary['selected'])
                assert [a['mode'] for a in audits].count('evaluate')==1
                assert [a['mode'] for a in audits].count('refit')==1
            else:
                assert not summary['experiments'] and summary['llm'][0]['status']=='failed'
                assert summary['selected'] is None
                assert not audits
            results.append({'scenario':name,'status':'passed','rows':len(output),'tool_generated_features':mode=='tool','worker_audits':audits,'summary':summary})
        # Second invocation reuses a measured toy result without another validation fit.
        task=parent/'regression-tool/task.md'; Mock.mode='tool'
        completed=subprocess.run([sys.executable,str(extracted/'solution.py'),str(task)],cwd=extracted,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=120,text=True,encoding='utf-8')
        (parent/'history-reuse.log').write_text(completed.stdout,encoding='utf-8')
        assert completed.returncode==0,completed.stdout
        latest=sorted((task.parent/'agent_runs').glob('*/summary.json'))[-1]
        summary=json.loads(latest.read_text(encoding='utf-8'))
        assert summary['selected_score'] is not None
        check_canonical(summary['selected'])
        assert sum(r['status']=='completed' for r in summary['experiments'])==1
        assert all(r['status']=='skipped' for r in summary['experiments'][1:])
        assert 1<=len(summary['experiments'])-1<=6
        assert not list(latest.parent.glob('experiment-*/metrics.json'))
        audits=check_workers(latest.parent)
        assert len(audits)==1 and audits[0]['mode']=='refit'
        output=pd.read_csv(task.parent/'submission.csv')
        assert len(output)==3 and output.object_key.tolist()==[1000,1001,1002]
        initial=json.loads((latest.parent/'initial_refit.json').read_text(encoding='utf-8'))
        assert initial and initial[0]['result']['status']=='completed'
        results.append({'scenario':'history-reuse-no-validation-repeat','status':'passed','worker_audits':audits,'summary':summary})
    except Exception as exc:
        failure=type(exc).__name__+': '+str(exc)
        (parent/'traceback.txt').write_text(traceback.format_exc(),encoding='utf-8')
    finally: server.shutdown(); server.server_close()
    status='failed' if failure else 'passed'
    record={'status':status,'error':failure,'archive':str(archive_path),'archive_sha256':hashlib.sha256(archive_path.read_bytes()).hexdigest(),
            'archive_root_files':sorted(expected_modules),'mock_http_calls':Mock.calls,'real_platform_api_tested':False,'clean_extraction':str(extracted),'scenarios':results}
    (parent/'verification.json').write_text(json.dumps(record,indent=2,ensure_ascii=False),encoding='utf-8')
    (parent/'report.md').write_text('# Приёмка плоского пакета full-v1\n\nСтатус: '+status+'. Детали и фактически завершённые сценарии — [verification.json](verification.json). Пакет содержит только9 корневых файлов с неизменённым requirements; родительский и дочерние процессы запускаются из чистой распаковки. Сценарии: regression/classification с переименованными target/id, вложенными CSV, missing/unknown категориями и numeric; LLM-попытка отключить full-v1 принудительно канонизируется; worker/refit сохраняют preprocessing-audit и маски с пределом5%; primary MAE при примере RMSE сохраняет правильную метрику; пустой API сохраняет fallback; история допускает до6 skipped запросов, без нового validation fit. Полный refit истории допустим. HTTP-вызовы подсчитаны фактически, без предположения фиксированного числа. Синтетические данные — техническая проверка, не новая публичная ML-гипотеза. Реальный платформенный API не проверен; полный requirements с нуля не переустанавливался.\n',encoding='utf-8')
    print(json.dumps({'status':status,'scenarios':len(results),'mock_calls':Mock.calls,'verification':str(parent/'verification.json')}))
    return 1 if failure else 0


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,default=None)
    raise SystemExit(main(parser.parse_args().archive))
