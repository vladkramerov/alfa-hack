"""Explicit local API check. Never imported or bundled by the platform agent."""
import argparse,json,os,sys,time,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from automl_llm import propose

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--base-url',required=True)
    parser.add_argument('--model')
    parser.add_argument('--generate',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Preserve existing receipts; choose a new output')
    path=ROOT/'.env.local'
    if path.exists():
        for line in path.read_text().splitlines():
            if '=' not in line or line.lstrip().startswith('#'):continue
            name,value=line.split('=',1);name=name.strip().removeprefix('export ')
            if name=='OPENAI_API_KEY':os.environ[name]=value.strip().strip('\"\'')
    key=os.environ.get('OPENAI_API_KEY')
    if not key:raise ValueError('Local organizer key is missing')
    result={'base_url':args.base_url,'requested_generations':int(args.generate)}
    started=time.monotonic()
    try:
        request=urllib.request.Request(args.base_url.rstrip('/')+'/models',headers={'Authorization':'Bearer '+key})
        with urllib.request.urlopen(request,timeout=20) as response:
            result['model_ids']=[r['id'] for r in json.load(response).get('data',[])]
        result['discovery_seconds']=time.monotonic()-started
        if args.generate:
            if args.model not in result['model_ids']:raise ValueError('Model must match the authenticated list')
            os.environ.update(OPENAI_BASE_URL=args.base_url,OPENAI_MODEL=args.model)
            action,metadata=propose({'type':'regression','metric':'rmse','rows':1000,
                'columns':['x','category'],'target_min':0,'target_skew':3,
                'task':'Predict a nonnegative target. Propose one simple experiment.'},[],timeout=35)
            result.update(model=args.model,action=action,metadata=metadata)
    except Exception as exc:
        result['error_type']=type(exc).__name__
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result))
if __name__=='__main__':main()
