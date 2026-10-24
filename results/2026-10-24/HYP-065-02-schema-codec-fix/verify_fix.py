"""Technical codec verification only: synthetic tests and exact public reads, no ML fits."""
import gc
import hashlib
import importlib.util
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT))
import automl_data


def frame_digest(frame):
    metadata=json.dumps([(str(c),str(t)) for c,t in zip(frame.columns,frame.dtypes)],ensure_ascii=False)
    return hashlib.sha256(metadata.encode()+pd.util.hash_pandas_object(frame,index=True).to_numpy().tobytes()).hexdigest()


def main():
    old_path=ROOT/'results/2026-10-24/series-alternatives-v1/source/automl_data.py'
    spec=importlib.util.spec_from_file_location('schema_codec_frozen_reference',old_path)
    old=importlib.util.module_from_spec(spec);sys.modules[spec.name]=old;spec.loader.exec_module(old)
    public=[]
    for kind in ['regression','multiclass']:
        task=ROOT/'data'/kind/'task.md'
        before=old.load_task(task);after=automl_data.load_task(task)
        for attribute in ['train','test','sample','dictionary','X','X_test']:
            pd.testing.assert_frame_equal(getattr(before,attribute),getattr(after,attribute),check_exact=True)
        pd.testing.assert_series_equal(before.y,after.y,check_exact=True)
        for attribute in ['target','id_column','task_type','metric']:
            assert getattr(before,attribute)==getattr(after,attribute)
        assert before.hashes==after.hashes
        fit,valid,description=automl_data.validation_split(after)
        old_fit,old_valid,old_description=old.validation_split(before)
        np.testing.assert_array_equal(fit,old_fit);np.testing.assert_array_equal(valid,old_valid)
        assert description==old_description
        with np.load(ROOT/f'results/2026-10-23/series-{kind}-preprocessing-v1/splits.npz') as frozen:
            np.testing.assert_array_equal(fit,frozen['outer_fit']);np.testing.assert_array_equal(valid,frozen['outer_valid'])
        public.append({'task':kind,'exact_old_new_dataframes':True,'exact_dtypes_and_values':True,
                       'target':after.target,'id_column':after.id_column,'task_type':after.task_type,'metric':after.metric,
                       'train_shape':list(after.train.shape),'test_shape':list(after.test.shape),
                       'X_digest':frame_digest(after.X),'y_digest':frame_digest(after.y.to_frame()),
                       'data_hashes':after.hashes,'split_key':automl_data.split_key(fit,valid),
                       'split_description':description,'fit_rows':len(fit),'valid_rows':len(valid)})
        del before,after;gc.collect()
    commands=[([sys.executable,'-X','utf8','-m','unittest','tests.test_csv_codec','tests.test_schema_semantics','-v'],'codec-tests.log'),
              ([sys.executable,'-X','utf8','-m','unittest','discover','-s','tests','-v'],'all-tests.log')]
    tests=[]
    for command,log in commands:
        result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,encoding='utf-8')
        (HERE/log).write_text(result.stdout+result.stderr,encoding='utf-8')
        tests.append({'command':command,'exit_code':result.returncode,'log':log,'tail':(result.stdout+result.stderr).splitlines()[-4:]})
    sources={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ['automl_data.py','tests/test_csv_codec.py','tests/test_schema_semantics.py']}
    for name in sources:
        destination=HERE/'source'/name;destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes((ROOT/name).read_bytes())
    result={'status':'passed' if all(t['exit_code']==0 for t in tests) else 'failed','public_model_fits':0,
            'new_public_metrics':0,'public_data_checks':public,'tests':tests,'sources':sources,
            'reference_source':str(old_path),'reference_sha256':hashlib.sha256(old_path.read_bytes()).hexdigest(),
            'original_diagnostic_preserved':True,'runtime':{'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__,'platform':platform.platform()}}
    (HERE/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'status':result['status'],'tests':tests,'public_model_fits':0,'public_data_checks':len(public)},ensure_ascii=False))
    if result['status']!='passed':raise SystemExit(1)


if __name__=='__main__':main()
