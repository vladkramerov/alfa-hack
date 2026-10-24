"""Root-only reproducible delivery, official requirements and source hashes."""
from pathlib import Path
import argparse,hashlib,json,subprocess,zipfile
ROOT=Path(__file__).resolve().parents[2]
REQ='9173b0388532fbd2a709e4f4b3029e3b9ccab685fce43a85261ffa1ba95efbe6'
def sha(data):return hashlib.sha256(data).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('output');a=p.parse_args();out=Path(a.output).resolve()
 assert not out.exists(), 'Preserve earlier packages; choose a new filename'
 names=['solution.py','requirements.txt']+sorted(p.name for p in ROOT.glob('ag_*.py'))
 assert len(names)==30 and sha((ROOT/'requirements.txt').read_bytes())==REQ
 secret=None
 envfile=Path('/Users/maxchibisov/Projects/HACKS/alfa-hack/.env.local')
 if envfile.exists():
  for line in envfile.read_text().splitlines():
   if line.strip().startswith('OPENAI_API_KEY='):secret=line.split('=',1)[1].strip().strip('"\'').encode()
 hashes={}
 out.parent.mkdir(parents=True,exist_ok=True)
 with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
  for name in names:
   data=(ROOT/name).read_bytes();assert not secret or secret not in data
   if name.endswith('.py'):compile(data,name,'exec')
   hashes[name]=sha(data);info=zipfile.ZipInfo(name,(2026,10,24,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
   info.external_attr=0o100644<<16;z.writestr(info,data,compresslevel=9)
 with zipfile.ZipFile(out) as z:assert z.testzip() is None and z.namelist()==names
 assert out.stat().st_size<100_000_000
 m={'zip':str(out),'sha256':sha(out.read_bytes()),'bytes':out.stat().st_size,'entries':len(names),
    'requirements_sha256':REQ,'source_sha256':hashes,'base_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
    'root_only':True,'crc_passed':True,'no_secret_data_models':True}
 out.with_suffix('.manifest.json').write_text(json.dumps(m,indent=2));print(json.dumps({k:v for k,v in m.items() if k!='source_sha256'}))
if __name__=='__main__':main()
