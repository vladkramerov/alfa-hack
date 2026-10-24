import sys,json,hashlib,time,subprocess,warnings
from pathlib import Path
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R))
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score,classification_report
S=R/'results/2026-10-24/sergey-scientific-v2';ORDER=['raw','smote','frequency'];CONTROL={'primary':.64472256711304,'secondary':.6075743572072007}
def save(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False,default=str))
ledger=(R/'HYPOTHESES.md').read_text();assert 'HYP-081' in ledger
assert subprocess.check_output(['git','branch','--show-current'],cwd=R,text=True).strip()=='sergey_branch'
for number,stage in enumerate(['primary','secondary'],1):
 out=R/f'results/2026-10-24/HYP-081-{number:02d}-neural-stack-{stage}';assert (out/'calibration.npz').exists();assert not (out/'head-metrics.json').exists()
 d=np.load(S/f'oof-cache-{stage}/calibration.npz');n=np.load(out/'calibration.npz');classes=d['classes'];y=d['y'];oy=d['outer_y']
 assert np.array_equal(n['classes'],classes) and np.array_equal(n['y'],y) and np.array_equal(n['fit_positions'],d['fit_positions'])
 parts=[d['oof_'+c] for c in ORDER]+[n['oof_neural']];outer=[d['outer_'+c] for c in ORDER]+[n['outer_neural']]
 values=np.concatenate([np.log(np.clip(p,1e-6,1)) for p in parts],axis=1);ov=np.concatenate([np.log(np.clip(p,1e-6,1)) for p in outer],axis=1)
 started=time.monotonic();save(out/'head-before-fit.json',{'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'ledger_sha256':hashlib.sha256(ledger.encode()).hexdigest(),'candidate':'normalized LR40 from3tree+PLE probabilities','parameters':{'C':.1,'max_iter':3000,'tol':1e-5},'outer_labels_used_for_fit':False})
 counts=np.asarray([(y==c).sum() for c in classes],float);cw=1/np.sqrt(counts);cw*=len(y)/sum(cw*counts);mapping=dict(zip(classes,cw))
 scaler=StandardScaler().fit(values);head=LogisticRegression(C=.1,max_iter=3000,tol=1e-5,class_weight=mapping)
 with warnings.catch_warnings(record=True) as caught:
  warnings.simplefilter('always');head.fit(scaler.transform(values),y)
 assert np.array_equal(head.classes_,classes)
 p=head.predict_proba(scaler.transform(ov));pred=classes[p.argmax(1)];score=float(f1_score(oy,pred,average='macro',zero_division=0))
 coefficient=head.coef_/scaler.scale_[None,:];intercept=head.intercept_-coefficient@scaler.mean_
 save(out/'state.json',{'coef':coefficient.tolist(),'intercept':intercept.tolist(),'classes':classes.tolist(),'component_order':ORDER+['ple_neural'],'clip':1e-6,'audit':{'scaler_mean':scaler.mean_.tolist(),'scaler_scale':scaler.scale_.tolist(),'class_weights':cw.tolist(),'n_iter':head.n_iter_.tolist(),'warnings':[str(v.message) for v in caught]}})
 np.savez_compressed(out/'validation_predictions.npz',y=oy,prediction=pred,probabilities=p,classes=classes,positions=d['valid_positions'])
 save(out/'classification_report.json',classification_report(oy,pred,output_dict=True,zero_division=0))
 result={'status':'completed','metric':'macro_f1','score':score,'control':CONTROL[stage],'relative_gain':score/CONTROL[stage]-1,'stage':stage,'meta_fits':1,'tree_fits':0,'head_seconds':time.monotonic()-started,'selection':'fixed preregistered four-family normalized head; no outer tuning','warnings':[str(v.message) for v in caught]}
 save(out/'head-metrics.json',result)
 with (out/'report.md').open('a') as f:f.write('\n## Head / final scientific assessment\n\n```json\n'+json.dumps(result,indent=2)+'\n```\n')
 with (R/'HYPOTHESES.md').open('a') as f:f.write('\n- SPEC012 **HYP-081/'+stage+'**: '+json.dumps(result)+'; [report]('+str(out.relative_to(R))+'/report.md). Не повторять.\n')
 with (R/'results/README.md').open('a') as f:f.write('| H81 | '+stage+' | '+str(score)+' | [report](../'+str(out.relative_to(R))+'/report.md) |\n')
 print(json.dumps(result),flush=True)
