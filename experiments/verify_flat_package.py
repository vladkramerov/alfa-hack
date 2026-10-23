"""Clean extraction, mock HTTP tool use and error fallback; no public validation fits."""
import contextlib
import http.server
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import zipfile
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]


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
                    'feature_code':'def engineer(X):\n    X = X.copy()\n    X["signal_square"] = X["signal"] ** 2\n    return X\n'}
            payload={'choices':[{'finish_reason':'tool_calls','message':{'tool_calls':[{'id':'local-1','type':'function','function':{'name':'run_experiment','arguments':json.dumps(action)}}]}}]}
        body=json.dumps(payload).encode(); self.send_response(200); self.send_header('Content-Type','application/json'); self.end_headers(); self.wfile.write(body)


def fixture(folder,classification):
    nested=folder/'nested/inputs'; nested.mkdir(parents=True)
    rng=np.random.default_rng(91); signal=rng.normal(size=100)
    target=np.where(signal>0,'positive','negative') if classification else signal**2+.1*rng.normal(size=100)
    pd.DataFrame({'object_key':np.arange(100),'signal':signal,'kind':['a','b']*50,'response':target}).to_csv(nested/'train.csv',index=False)
    pd.DataFrame({'object_key':[1000,1001,1002],'signal':[-1.,0.,1.],'kind':['new','a',None]}).to_csv(nested/'test.csv',index=False)
    pd.DataFrame({'object_key':[1000,1001,1002],'response':['positive']*3 if classification else [0.]*3}).to_csv(nested/'sample_submission.csv',index=False)
    pd.DataFrame({'feature':['object_key','signal','kind','response'],'role':['id','feature','feature','target']}).to_csv(nested/'data_dictionary.csv',index=False)
    task=folder/'task.md'; task.write_text(('Classification. Macro F1.' if classification else 'Regression. RMSE.')+' Data: nested/inputs/train.csv nested/inputs/test.csv nested/inputs/sample_submission.csv nested/inputs/data_dictionary.csv',encoding='utf-8')
    return task


def main():
    parent=ROOT/'results/2026-10-23'/('platform-verification-'+str(time.time_ns())); parent.mkdir(exist_ok=False)
    extracted=ROOT/'tmp'/('flat-'+str(time.time_ns())); extracted.mkdir(parents=True)
    with zipfile.ZipFile(ROOT/'dist/solution_ivan_branch.zip') as archive: archive.extractall(extracted)
    server=http.server.ThreadingHTTPServer(('127.0.0.1',0),Mock)
    thread=threading.Thread(target=server.serve_forever,daemon=True); thread.start()
    results=[]
    try:
        for name,classification,mode in [('regression-tool',False,'tool'),('classification-tool',True,'tool'),('regression-api-failure',False,'empty')]:
            task=fixture(parent/name,classification); Mock.mode=mode
            env=os.environ.copy(); env.update(OPENAI_BASE_URL=f'http://127.0.0.1:{server.server_port}/v1',OPENAI_API_KEY='local-mock-only',OPENAI_MODEL='local-mock',AUTOML_PORTFOLIO='0',AUTOML_MAX_EXPERIMENTS='1',AUTOML_TIME_LIMIT='120')
            completed=subprocess.run([sys.executable,str(extracted/'solution.py'),str(task)],cwd=extracted,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=120,text=True,encoding='utf-8')
            (parent/(name+'.log')).write_text(completed.stdout,encoding='utf-8')
            assert completed.returncode==0,completed.stdout
            summary=json.loads(next((task.parent/'agent_runs').glob('*/summary.json')).read_text(encoding='utf-8'))
            ledger=json.loads((task.parent/'hypotheses.json').read_text(encoding='utf-8'))
            assert len(ledger['cases'])==1
            if mode=='tool': assert next(iter(ledger['cases'].values()))['records'][0]['status']=='completed'
            output=pd.read_csv(task.parent/'submission.csv'); assert list(output)==['object_key','response'] and len(output)==3
            assert output.object_key.tolist()==[1000,1001,1002]
            if mode=='tool':
                assert len(summary['experiments'])==1 and summary['experiments'][0]['status']=='completed'
                assert summary['selected']['feature_code'] and summary['selected_score'] is not None
            else:
                assert not summary['experiments'] and summary['llm'][0]['status']=='failed'
                assert summary['selected'] is None
            results.append({'scenario':name,'status':'passed','rows':len(output),'tool_generated_features':mode=='tool','summary':summary})
        # Second invocation reuses a measured toy result without another validation fit.
        task=parent/'regression-tool/task.md'; Mock.mode='tool'
        completed=subprocess.run([sys.executable,str(extracted/'solution.py'),str(task)],cwd=extracted,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=120,text=True,encoding='utf-8')
        assert completed.returncode==0,completed.stdout
        latest=sorted((task.parent/'agent_runs').glob('*/summary.json'))[-1]
        summary=json.loads(latest.read_text(encoding='utf-8'))
        assert summary['selected_score'] is not None
        assert sum(r['status']=='completed' for r in summary['experiments'])==1
        assert all(r['status']=='skipped' for r in summary['experiments'][1:])
        results.append({'scenario':'history-reuse-no-validation-repeat','status':'passed','summary':summary})
    finally: server.shutdown(); server.server_close()
    record={'status':'passed','mock_http_calls':Mock.calls,'real_platform_api_tested':False,'clean_extraction':str(extracted),'scenarios':results}
    (parent/'verification.json').write_text(json.dumps(record,indent=2,ensure_ascii=False),encoding='utf-8')
    (parent/'report.md').write_text('# Приёмка плоского пакета\n\nВсе сценарии verification.json выполнены из чистой распаковки. Регрессия/классификация с переименованными target/id и вложенными CSV; HTTP tool-call и сгенерированная feature-функция выполнены дочерним процессом; пустой API сохраняет fallback; второй запуск не повторяет validation fit. Синтетические данные — техническая проверка протокола, не новая ML-гипотеза. Реальный платформенный API не проверен: env отсутствуют. Полный requirements не переустанавливался; проверены используемые pinned зависимости.\n',encoding='utf-8')
    print(json.dumps({'status':'passed','scenarios':len(results),'mock_calls':Mock.calls}))


if __name__=='__main__': main()
