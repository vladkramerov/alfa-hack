"""Strict deterministic source-only flat ZIP, SPEC012."""
import sys,subprocess,json,hashlib,zipfile
from pathlib import Path
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R))
from automl_known import KNOWN_TASKS
NAMES=['solution.py','automl_data.py','automl_models.py','automl_preprocessing.py','automl_history.py','automl_known.py','automl_llm.py','automl_worker.py','automl_advanced.py','automl_combo_memory.py','automl_feature_sets.py','automl_reg_combo.py','automl_class_combo.py','automl_combo_policy.py','automl_temporal_memory.py','automl_panel_memory.py','automl_panel_change.py','requirements.txt']
def main():
 assert subprocess.check_output(['git','branch','--show-current'],cwd=R,text=True).strip()=='sergey_branch'
 assert KNOWN_TASKS==[]
 original=(R.parent/'data_for_participants/requirements.txt').read_bytes();assert (R/'requirements.txt').read_bytes()==original
 archive=R/'dist/solution_sergey_scientific_v2.zip';assert not archive.exists(),'Never overwrite a published package'
 with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
  for name in NAMES:
   assert Path(name).name==name and '/' not in name and '\\' not in name and (name.endswith('.py') or name=='requirements.txt')
   content=(R/name).read_bytes()
   if name.endswith('.py'):compile(content,name,'exec')
   info=zipfile.ZipInfo(name,(2026,10,24,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o100644<<16;z.writestr(info,content,compresslevel=9)
 with zipfile.ZipFile(archive) as z:
  assert z.testzip() is None and z.namelist()==NAMES and len(set(z.namelist()))==len(NAMES)
  assert not any(i.is_dir() or i.flag_bits&1 for i in z.infolist())
  assert all(z.read(name)==(R/name).read_bytes() for name in NAMES)
 assert 0<archive.stat().st_size<=100_000_000
 manifest={'archive':archive.name,'bytes':archive.stat().st_size,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'root_files':NAMES,'source_sha256':{name:hashlib.sha256((R/name).read_bytes()).hexdigest() for name in NAMES},'root_only':True,'directories':0,'provided_requirements_exact':True,'data_models_configs_secrets_included':False,'embedded_pretrained_calibration_or_public_dataset_state':False,'python':'3.12','entrypoint':'python3 solution.py task.md','output':'submission.csv next to task.md','real_llm_api_tested':False,'real_llm_api_skipped_at_user_request':True,'hidden_platform_score_verified':False,'old_archive_preserved_sha256':hashlib.sha256((R/'dist/solution_sergey_panel_v1.zip').read_bytes()).hexdigest()}
 assert manifest['old_archive_preserved_sha256']=='cae8cdbd83e32ece12d4ee0fa1e14997916db861e8732fb5402ce4707f96534c'
 (R/'results/2026-10-24/sergey-scientific-v2/package-manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
 print(json.dumps({k:manifest[k] for k in ['archive','bytes','sha256','root_only','directories']}))
if __name__=='__main__':main()
