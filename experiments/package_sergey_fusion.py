"""Create a separate flat source tree from the immutable Sergey delivery."""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'deliveries/sergey_ivan_fusion_v1'
BASE=ROOT/'dist/solution_sergey_panel_v1_24e3311.zip'
OWN=['fusion_weights.py','automl_sergey_fusion.py','sprint_regression.py',
     'sprint_ratio_transport.py','sprint_uncertainty.py']


def change(name,old,new):
    path=DEST/name;text=path.read_text(encoding='utf8')
    assert text.count(old)==1,(name,old)
    path.write_text(text.replace(old,new),encoding='utf8',newline='\n')


def main():
    assert hashlib.sha256(BASE.read_bytes()).hexdigest()=='cae8cdbd83e32ece12d4ee0fa1e14997916db861e8732fb5402ce4707f96534c'
    DEST.mkdir(parents=True,exist_ok=False)
    with zipfile.ZipFile(BASE) as z:
        for info in z.infolist():
            assert '/' not in info.filename and '\\' not in info.filename
            assert info.filename.endswith('.py') or info.filename=='requirements.txt'
            (DEST/info.filename).write_bytes(z.read(info))
    for name in OWN:(DEST/name).write_bytes((ROOT/name).read_bytes())
    change('automl_models.py',"    strategy=result.get('strategy')\n", "    strategy=result.get('strategy')\n"
        "    if strategy=='sergey_ivan_fusion':\n"
        "        from automl_sergey_fusion import canonical_fusion\n"
        "        return canonical_fusion(result)\n")
    change('automl_models.py',"    candidate=canonical_candidate(candidate)\n", "    candidate=canonical_candidate(candidate)\n"
        "    if candidate.get('strategy')=='sergey_ivan_fusion':\n"
        "        if kind!='regression': raise ValueError('Fusion requires regression')\n"
        "        from automl_sergey_fusion import SergeyIvanFusion\n"
        "        return SergeyIvanFusion(candidate,threads)\n")
    change('automl_history.py',"    if candidate.get('strategy')=='panel_hybrid_memory':\n",
        "    if candidate.get('strategy') in {'panel_hybrid_memory','sergey_ivan_fusion'}:\n")
    old="                candidates.append({'family':'extra_trees','parameters':{},'strategy':'panel_hybrid_memory',"
    new="""                fusion_fields={'energy_star_rating','ELEVATION','direction_max_wind_speed','direction_peak_wind_speed'}
                fusion_fields.update(m+'_avg_temp_c' for m in ['january','february','march','april','may','june','july','august','september','october','november','december'])
                if metric=='rmse' and np.all(target>=0) and fusion_fields.issubset(training.columns) and training['Year_Factor'].nunique()>=3:
                    candidates.append({'family':'extra_trees','parameters':{},'strategy':'sergey_ivan_fusion',
                        'hypothesis':'Train-only temporal blend of causal annual dynamics and rating transport; fresh local weights'})
                candidates.append({'family':'extra_trees','parameters':{},'strategy':'panel_hybrid_memory',"""
    change('solution.py',old,new)
    manifest={'base_sha256':hashlib.sha256(BASE.read_bytes()).hexdigest(),
        'base_commit':'24e3311f6ba958a67c740d5b0328cc34cb7e15d7',
        'classification_strategy':'unchanged_Sergey_fresh_OOF',
        'embedded_public_learned_state':False,'new_weight_training':'current_task_latest_train_period',
        'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(DEST.iterdir())}}
    target=ROOT/'results/2026-10-24/series-sergey-ivan-fusion-v1/delivery-source-manifest.json'
    target.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
    print(str(DEST))


if __name__=='__main__':main()
