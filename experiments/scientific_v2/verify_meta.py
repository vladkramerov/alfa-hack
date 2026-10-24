"""Independent SPEC012 H76--H78 verification: predictions/statistics only, no fits."""
from pathlib import Path
import gc, hashlib, json, subprocess, sys
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import f1_score,accuracy_score,confusion_matrix,classification_report
R=Path(__file__).resolve().parents[2]
W=Path(__file__).resolve().parent
S=R/'results/2026-10-24/sergey-scientific-v2'
sys.path.insert(0,str(R))
assert subprocess.check_output(['git','branch','--show-current'],cwd=R,text=True).strip()=='sergey_branch'
checks=[];failures=[];artifacts={};rows=[];all_fingerprints=[]
def check(name,condition,details=None):
 item={'check':name,'passed':bool(condition)}
 if details is not None:item['details']=details
 checks.append(item)
 if not condition:failures.append(item)
def sha(path):
 path=Path(path);value=hashlib.sha256(path.read_bytes()).hexdigest();artifacts[str(path.relative_to(R)) if path.is_relative_to(R) else str(path)]=value;return value
def read(path):return json.loads(Path(path).read_text())
def close(a,b,tol=1e-10):return np.asarray(a).shape==np.asarray(b).shape and np.allclose(a,b,atol=tol,rtol=tol)
def arrays(path):
 with np.load(path,allow_pickle=False) as f:return {k:f[k] for k in f.files}
def scores(y,p,c):return float(f1_score(y,c[np.argmax(p,axis=1)],labels=c,average='macro',zero_division=0))
def probability(name,p,n,k):
 check(name+'.shape',p.shape==(n,k));check(name+'.simplex',np.isfinite(p).all() and np.all(p>=0) and close(p.sum(1),np.ones(n),2e-6))
def features(parts):return np.concatenate([np.log(np.clip(p/p.sum(1,keepdims=True),1e-6,1.)) for p in parts],axis=1)
def train_sha(parts,y,classes):
 v=features(parts);ind=np.array([{c:i for i,c in enumerate(classes)}[a] for a in y],dtype='<i8')
 h=hashlib.sha256(np.ascontiguousarray(v,dtype='<f8').tobytes());h.update(ind.tobytes());return h.hexdigest()
def predict(parts,state,classes):
 check('state.class_order',np.array_equal(state['classes'],classes))
 check('state.protocol',state['component_order']==['raw','smote','frequency'] and state['clip']==1e-6)
 v=features(parts);coef=np.asarray(state['coef']);intercept=np.asarray(state['intercept'])
 check('state.finite',np.isfinite(coef).all() and np.isfinite(intercept).all())
 z=v@coef.T+intercept
 if len(classes)==2:z=np.column_stack([np.zeros(len(v)),z[:,0]])
 z-=z.max(1,keepdims=True);p=np.exp(z);return p/p.sum(1,keepdims=True)
def normalized_audit(name,state,parts,y,c):
 a=state['audit'];v=features(parts);counts=np.array([sum(y==s) for s in c]);weight=1/np.sqrt(counts);weight*=len(y)/(counts@weight)
 check(name+'.mechanism',a['mechanism']=='normalized_converged_meta')
 check(name+'.training_sha256',a['training_sha256']==train_sha(parts,y,c))
 check(name+'.fit_rows',a['fit_rows']==len(y))
 check(name+'.class_counts',a['class_counts']==counts.tolist())
 check(name+'.class_weights',close(a['class_weights'],weight))
 check(name+'.weight_mean_one',close(a['mean_sample_weight'],1.))
 check(name+'.scaler_fit_only',close(a['scaler_mean'],v.mean(0)) and close(a['scaler_scale'],np.where(v.std(0)==0,1.,v.std(0))))
 check(name+'.converged',a['converged'] and not a['warnings'] and max(a['n_iter'])<a['max_iter'])
def crossfit(name,directory,parts,y,c,folds,kind,normal_audit=None):
 a=read(directory/('legacy-crossfit-audit.json' if kind=='legacy' else 'crossfit-audit.json'))
 parts=[p/p.sum(1,keepdims=True) for p in parts] # crossfit_meta normalizes once before partitioning; the fitted head normalizes again
 saved=arrays(directory/('legacy-crossfit.npz' if kind=='legacy' else 'meta-crossfit.npz'))['probabilities']
 reconstructed=np.zeros_like(saved);coverage=np.zeros(len(y),int)
 check(name+'.conditional_disclosure',a['fully_nested_base_evaluation'] is False)
 for i,(fold,record,state) in enumerate(zip(folds,a['folds'],a['states'])):
  fi=np.array(fold['fit']);va=np.array(fold['valid']);coverage[va]+=1
  check(name+f'.fold{i}.positions',record['fit_positions']==fi.tolist() and record['valid_positions']==va.tolist())
  selected=[p[fi] for p in parts]
  check(name+f'.fold{i}.training_sha',state['audit']['training_sha256']==train_sha(selected,y[fi],c))
  check(name+f'.fold{i}.fit_rows',state['audit']['fit_rows']==len(fi))
  if kind=='normalized':normalized_audit(name+f'.fold{i}',state,selected,y[fi],c)
  if kind=='mcc':
   initial=normal_audit['states'][i]['audit']
   check(name+f'.fold{i}.reuse',record['reused_normalized_state'] and record['fit_count']==1 and record['initial_meta_audit']==initial)
   check(name+f'.fold{i}.warm_start',state['audit']['warm_start']=='current_train_H76' and state['audit']['training_sha256']==initial['training_sha256'])
  reconstructed[va]=predict([p[va] for p in parts],state,c)
 check(name+'.coverage',np.all(coverage==1));check(name+'.probabilities_replay',close(saved,reconstructed))
 check(name+'.score',close(a['macro_f1'],scores(y,reconstructed,c)))
 check(name+'.fit_count',a['fit_count']==3)
 return saved,a
IGNORED={'hypothesis','name','display_name','random_state','random_seed','seed','n_jobs','thread_count','verbosity','verbose'}
def semantic(v):
 if isinstance(v,dict):return {str(k):semantic(val) for k,val in v.items() if k not in IGNORED}
 if isinstance(v,(list,tuple)):return [semantic(val) for val in v]
 if isinstance(v,np.generic):return v.item()
 return v
def data_sha(X):
 h=hashlib.sha256()
 if isinstance(X,pd.DataFrame):
  h.update(json.dumps([(str(c),str(t)) for c,t in zip(X.columns,X.dtypes)],ensure_ascii=False).encode());h.update(pd.util.hash_pandas_object(X,index=False).values.tobytes())
 else:
  a=np.asarray(X);h.update(str((a.shape,str(a.dtype))).encode())
  if a.dtype.hasobject or a.dtype.kind in 'US':h.update(pd.util.hash_pandas_object(pd.DataFrame(a),index=False).values.tobytes())
  else:h.update(np.ascontiguousarray(a).tobytes())
 return h.hexdigest()
train=pd.read_csv(R/'data/multiclass/train.csv');target='attack_cat';X=train.drop(columns=[target,'id']);raw_groups=pd.util.hash_pandas_object(X,index=False).to_numpy()
for stage in ('primary','secondary'):
 cache=S/f'oof-cache-{stage}';d=arrays(cache/'calibration.npz');ev=read(cache/'evidence.json');cm=read(cache/'metrics.json');c=d['classes'];y=d['y'];oy=d['outer_y'];fi=d['fit_positions'];va=d['valid_positions'];n=len(y)
 check(stage+'.cache_completed',cm['status']=='completed');check(stage+'.cache_sha',cm['cache_sha256']==sha(cache/'calibration.npz'))
 for name,h in ev['data_hashes'].items():check(stage+'.data_sha.'+name,sha(R/'data/multiclass'/name)==h)
 h=hashlib.sha256(np.asarray(fi,dtype='<i8').tobytes()+b'|'+np.asarray(va,dtype='<i8').tobytes()).hexdigest();check(stage+'.split_sha',h==ev['split_key'])
 oldsplit=arrays(R/f'results/2026-10-24/sergey-improvement-v1/control-multiclass-{stage}/splits.npz')
 check(stage+'.cached_control_positions',np.array_equal(fi,oldsplit['fit']) and np.array_equal(va,oldsplit['valid']))
 check(stage+'.outer_split_disjoint',len(set(fi)&set(va))==0 and len(set(raw_groups[fi])&set(raw_groups[va]))==0)
 check(stage+'.target_alignment',np.array_equal(y,train[target].iloc[fi].to_numpy(str)) and np.array_equal(oy,train[target].iloc[va].to_numpy(str)))
 check(stage+'.class_order',np.array_equal(c,np.unique(y)))
 check(stage+'.reconstruction_disclosed','9 component fits duplicate historical OOF' in ev['reconstruction_disclosure'] and cm['new_required_component_fits']==9 and cm['outer_full_branch_fits']==0)
 check(stage+'.cache_builder_sha',ev['code_sha256']==sha(W/'build_oof_cache.py'))
 parts=[d['oof_'+b] for b in ('raw','smote','frequency')];outer=[d['outer_'+b] for b in ('raw','smote','frequency')]
 for b,p,q in zip(('raw','smote','frequency'),parts,outer):probability(stage+'.oof.'+b,p,n,len(c));probability(stage+'.outer.'+b,q,len(oy),len(c))
 folds=read(cache/'folds.json');coverage=np.zeros(n,int);attempts=[json.loads(s) for s in (cache/'fit-journal/model-fit-attempts.jsonl').read_text().splitlines()];outcomes=[json.loads(s) for s in (cache/'fit-journal/model-fit-outcomes.jsonl').read_text().splitlines()]
 check(stage+'.nine_attempts',len(attempts)==9 and len(outcomes)==9);check(stage+'.unique_fit_fingerprints',len({a['fingerprint'] for a in attempts})==9);all_fingerprints.extend(a['fingerprint'] for a in attempts)
 frame=X.iloc[fi].reset_index(drop=True)
 for i,f in enumerate(folds):
  inf=np.asarray(f['fit']);inv=np.asarray(f['valid']);coverage[inv]+=1
  stored=arrays(cache/f'fold-{i}-split.npz');check(stage+f'.fold{i}.stored_split',np.array_equal(inf,stored['fit']) and np.array_equal(inv,stored['valid']))
  groups=raw_groups[fi];check(stage+f'.fold{i}.raw_group_disjoint',not set(groups[inf])&set(groups[inv]) and not set(inf)&set(inv))
  for j,b in enumerate(('raw','smote','frequency')):
   prefix=stage+f'.fold{i}.{b}';branch=joblib.load(cache/f'fold-{i}-{b}.joblib');saved=arrays(cache/f'fold-{i}-{b}.npz');record=attempts[i*3+j];audit=read(cache/f'fold-{i}-{b}-audit.json')
   check(prefix+'.class_alignment',np.array_equal(branch.classes_,c));check(prefix+'.cache_assembly',np.array_equal(saved['positions'],inv) and close(saved['probabilities'],parts[j][inv]))
   train_frame=frame.iloc[inf].copy();train_y=y[inf].copy()
   if b=='smote' and len(branch.synthetic_):
    synth=branch.synthetic_;train_frame=pd.concat([train_frame,synth.drop(columns='__synthetic_target__')],ignore_index=True);train_y=np.concatenate([train_y,synth['__synthetic_target__'].to_numpy()])
   if branch.feature_ is not None:train_frame=branch.feature_.transform(train_frame)
   values=branch.encoder_.transform(train_frame);encoded=branch.labels_.transform(train_y)
   check(prefix+'.input_hash_replay',record['X_sha256']==data_sha(values) and record['y_sha256']==data_sha(encoded.reshape(-1,1)))
   identity={'X':record['X_sha256'],'y':record['y_sha256'],'configuration':semantic(record['configuration'])}
   fingerprint=hashlib.sha256(json.dumps(identity,sort_keys=True,default=str).encode()).hexdigest()
   check(prefix+'.fingerprint',fingerprint==record['fingerprint']);check(prefix+'.fit_rows',record['rows']==len(values)==audit['fit_model_rows'] and audit['fit_original_rows']==len(inf))
   check(prefix+'.completed',outcomes[i*3+j]['fingerprint']==fingerprint and outcomes[i*3+j]['status']=='completed')
   del branch,values,train_frame;gc.collect()
 check(stage+'.oof_coverage',np.all(coverage==1))
 # Full fitted branch models are reused for predictions, never fitted here.
 full=joblib.load(R/f'results/2026-10-24/sergey-improvement-v1/control-multiclass-{stage}/model.joblib')
 for j,b in enumerate(('raw','smote','frequency')):
  branch=full.branches_[b];p=branch.predict_proba(X.iloc[va]);aligned=np.column_stack([p[:,list(branch.classes_).index(label)] for label in c]);aligned/=aligned.sum(1,keepdims=True)
  check(stage+'.outer.'+b+'.cached_model_replay',close(aligned,outer[j]))
 del full;gc.collect()
 dirs={h:R/f'results/2026-10-24/HYP-{h:03d}-{1 if stage=="primary" else 2:02d}-meta-{stage}' for h in (76,77,78)}
 normalized=read(dirs[76]/'state.json');normalized_audit(stage+'.H76.full',normalized,parts,y,c)
 ncf,naudit=crossfit(stage+'.H76',dirs[76],parts,y,c,folds,'normalized')
 lcf,laudit=crossfit(stage+'.H77',dirs[77],parts,y,c,folds,'legacy')
 mcf,maudit=crossfit(stage+'.H78',dirs[78],parts,y,c,folds,'mcc',naudit)
 expected_by_h={76:predict(outer,normalized,c)}
 legacy_state=read(R/'results/2026-10-24/HYP-075-02-multiclass-clean-runtime-continuation/metrics.json')['learned_state']['meta_state'] if stage=='primary' else read(R/f'results/2026-10-24/sergey-improvement-v1/control-multiclass-{stage}/audit.json')['meta_state']
 anchor='normalized' if scores(y,ncf,c)>scores(y,lcf,c) else 'legacy';state77=read(dirs[77]/'state.json');rule=read(dirs[77]/'rule.json');metrics77=read(dirs[77]/'metrics.json')
 check(stage+'.H77.anchor_train_only',metrics77['anchor_selected_inside_train']==anchor)
 check(stage+'.H77.anchor_state',state77==(normalized if anchor=='normalized' else legacy_state))
 p=predict(outer,state77,c);a=np.asarray(rule['a']);b=np.asarray(rule['b']);utility=p*a-b;utility-=utility.max(1,keepdims=True);expected77=np.exp(utility);expected77/=expected77.sum(1,keepdims=True)
 expected_by_h[77]=expected77
 selectedcf=ncf if anchor=='normalized' else lcf;train_pred=np.argmax(selectedcf*a-b,axis=1)
 check(stage+'.H77.affine_train_score',close(f1_score(y,c[train_pred],labels=c,average='macro',zero_division=0),rule['audit']['macro_f1']))
 check(stage+'.H77.affine_rule',rule['classes']==c.tolist() and np.all(a>0) and np.all(b>=0) and rule['audit']['selection_source']=='training_meta_crossfit_only' and rule['audit']['calibrated_posteriors'] is False)
 state78=read(dirs[78]/'state.json');audit78=state78['audit'];metric78=read(dirs[78]/'metrics.json')
 check(stage+'.H78.full_train_sha',audit78['training_sha256']==train_sha(parts,y,c)==normalized['audit']['training_sha256'])
 check(stage+'.H78.full_fit_rows',audit78['fit_rows']==len(y) and audit78['warm_start']=='current_train_H76' and audit78['steps']==100 and len(audit78['losses'])==100)
 selected=scores(y,mcf,c)>scores(y,ncf,c)
 check(stage+'.H78.cf_selection',metric78['mcc_selected_inside_train']==selected)
 raw=predict(outer,state78,c);expected_by_h[78]=raw if selected else expected_by_h[76]
 standalone=arrays(dirs[78]/'standalone-mcc-predictions.npz');check(stage+'.H78.standalone_replay',close(standalone['probabilities'],raw) and np.array_equal(standalone['prediction'],c[raw.argmax(1)]))
 check(stage+'.H78.standalone_metric',close(metric78['standalone_mcc_outer_score'],scores(oy,raw,c)))
 for hid,directory in dirs.items():
  name=stage+f'.H{hid}';e=read(directory/'evidence.json');m=read(directory/'metrics.json');out=arrays(directory/'validation_predictions.npz');sha(directory/'validation_predictions.npz');sha(directory/'state.json')
  for fn,h in e['source_sha256'].items():check(name+'.source_sha.'+fn,sha(W/('meta/'+fn if fn=='automl_meta_research.py' else fn))==h)
  check(name+'.evidence_cache_sha',e['cache_sha256']==cm['cache_sha256'] and e['cache_evidence']==ev)
  check(name+'.state_training_scope',e['outer_labels_used_for_fit'] is False and e['fit_rows']==n and e['valid_rows']==len(oy))
  check(name+'.prediction_positions',np.array_equal(out['positions'],va));check(name+'.truth_classes',np.array_equal(out['y'],oy) and np.array_equal(out['classes'],c))
  probability(name+'.predictions',out['probabilities'],len(oy),len(c));check(name+'.coefficient_replay',close(out['probabilities'],expected_by_h[hid]));check(name+'.argmax_labels',np.array_equal(out['prediction'],c[np.argmax(out['probabilities'],axis=1)]))
  f1=scores(oy,out['probabilities'],c);acc=float(accuracy_score(oy,out['prediction']))
  check(name+'.f1_accuracy',close(m['score'],f1) and close(m['accuracy'],acc));check(name+'.relative_gain',close(m['relative_gain'],f1/m['control']-1));check(name+'.no_new_base_tree_fits',m['base_tree_fits']==0)
  check(name+'.confusion',read(directory/'confusion.json')=={'classes':c.tolist(),'matrix':confusion_matrix(oy,out['prediction'],labels=c).tolist()})
  report=classification_report(oy,out['prediction'],labels=c,output_dict=True,zero_division=0);check(name+'.per_class',read(directory/'per-class.json')==report)
  rows.append({'hypothesis':hid,'stage':stage,'score':f1,'control':m['control'],'relative_gain':f1/m['control']-1,'accuracy':acc})
check('all_caches.18_distinct_fingerprints',len(all_fingerprints)==18 and len(set(all_fingerprints))==18)
result={'verification_source_sha256':sha(Path(__file__)),'prior_verification':{'path':str(S/'meta-independent-verification-v2.json'),'sha256':sha(S/'meta-independent-verification-v2.json'),'initial_path':str(S/'meta-independent-verification.json'),'initial_sha256':sha(S/'meta-independent-verification.json'),'correction':'Initial verifier omitted row renormalization before log-feature SHA; v2 accounted for full-head normalization but not the extra crossfit-level normalization before partitioning. All scores/fingerprints already matched. v3 reproduces both normalization boundaries; no fits.'},'status':'passed' if not failures else 'failed','new_fits_in_verifier':0,'total_checks':len(checks),'passed':sum(c['passed'] for c in checks),'failures':failures,'metrics':rows,'technical_cache_fits_disclosed':18,'unique_cache_fingerprints':len(set(all_fingerprints)),'limitations':['Meta cross-fit is conditional on base OOF, not fully nested base-model evaluation.','Both public protocols have already been repeatedly examined; neither is a hidden private test.','Historical mutable ledger SHAs are recorded by producers; no immutable historical ledger snapshot was available for independent reconstruction.'],'artifact_sha256':artifacts,'checks':checks}
output=S/'meta-independent-verification-v3.json';assert not output.exists(),'Immutable verification path already exists'
output.write_text(json.dumps(result,ensure_ascii=False,indent=2))
print(json.dumps({k:result[k] for k in ('status','total_checks','passed','failures','metrics','technical_cache_fits_disclosed','unique_cache_fingerprints')},ensure_ascii=False,indent=2))
raise SystemExit(0 if not failures else 1)
