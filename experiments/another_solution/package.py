"""Deterministic root-only source ZIP; organizer dependencies unchanged."""
import sys,json,hashlib,zipfile,subprocess
from pathlib import Path
R=Path(__file__).resolve().parents[2]
def main():
 assert subprocess.check_output(['git','branch','--show-current'],cwd=R,text=True).strip()=='another_solution'
 info=json.loads((R/'reports/SOURCE_ARCHIVE.json').read_text());names=['solution.py','requirements.txt']+sorted(n for n in info['source_root_files'] if n.startswith('ag_'))
 assert len(names)==len(set(names))==30
 out=R/'dist/solution_another_solution_v1.zip';out.parent.mkdir(exist_ok=True);assert not out.exists()
 assert hashlib.sha256((R/'requirements.txt').read_bytes()).hexdigest()=='9173b0388532fbd2a709e4f4b3029e3b9ccab685fce43a85261ffa1ba95efbe6'
 hashes={n:hashlib.sha256((R/n).read_bytes()).hexdigest() for n in names}
 with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
  for n in names:
   assert Path(n).name==n and (n.endswith('.py') or n=='requirements.txt')
   data=(R/n).read_bytes()
   if n.endswith('.py'):compile(data,n,'exec')
   zi=zipfile.ZipInfo(n,(2026,10,24,0,0,0));zi.compress_type=zipfile.ZIP_DEFLATED;zi.external_attr=0o100644<<16;z.writestr(zi,data,compresslevel=9)
 with zipfile.ZipFile(out) as z:
  assert z.testzip() is None and z.namelist()==names and not any(i.is_dir() or i.flag_bits&1 for i in z.infolist())
  assert all(hashlib.sha256(z.read(n)).hexdigest()==hashes[n] for n in names)
 assert 0<out.stat().st_size<=100_000_000
 manifest={'archive':out.name,'bytes':out.stat().st_size,'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'root_files':names,'directories':0,'source_sha256':hashes,'original_archive_sha256':info['sha256'],'provided_requirements_exact':True,'entrypoint':'python3 solution.py task.md','output':'submission.csv next to task.md','secret_data_models_included':False,'real_llm_tested':False,'platform_submission_sent':False}
 (R/'reports/PACKAGE_MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2));print(json.dumps({k:manifest[k] for k in ['archive','bytes','sha256','directories']}))
if __name__=='__main__':main()
