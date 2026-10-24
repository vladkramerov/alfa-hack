"""Recompute saved branch metrics without training or using an API key."""
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT=Path(__file__).resolve().parents[1]
SNAPSHOT=ROOT/'tmp/sergey-24e3311f6ba9'
OUT=ROOT/'results/2026-10-24/branch-comparison-sergey-24e3311-v3'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    OUT.mkdir(exist_ok=False)
    evidence={}
    def read_run(root,name,metric,kind):
        directory=root/'results/2026-10-24'/name
        path=directory/'validation_predictions.npz'
        with np.load(path,allow_pickle=False) as z:
            positions=z['positions'].copy();y=z['y'].copy();p=z['prediction'].copy()
        train_path=ROOT/'data'/kind/'train.csv'
        truth=pd.read_csv(train_path)['site_eui' if kind=='regression' else 'attack_cat'].to_numpy()[positions]
        if metric=='macro_f1':truth=truth.astype(str);y=y.astype(str);p=p.astype(str)
        if metric=='rmse':np.testing.assert_allclose(y,truth,rtol=0,atol=1e-10)
        else:assert np.array_equal(y,truth)
        split_path=directory/'splits.npz'
        if name=='HYP-091-02-outer-multiblend-sprint':
            split_path=root/'results/2026-10-24/HYP-070-04-outer-high-sprint/splits.npz'
        with np.load(split_path,allow_pickle=False) as z:
            assert np.array_equal(positions,z['valid']) and not np.intersect1d(z['fit'],z['valid']).size
            fit=z['fit'].copy()
        value=float(np.sqrt(np.mean((p-y)**2))) if metric=='rmse' else float(f1_score(y,p,average='macro',zero_division=0))
        stored=json.loads((directory/'metrics.json').read_text(encoding='utf8'))
        assert stored['status']=='completed' and abs(value-stored['score'])<1e-10
        evidence[str(path.relative_to(ROOT))]=sha(path)
        return {'score':value,'positions':positions,'y':y,'fit':fit,'source':str(directory.relative_to(ROOT))}
    tasks={}
    pairs=[('regression','rmse','HYP-091-02-outer-multiblend-sprint','HYP-073-01-regression-panel-primary',39.39219382766771),
           ('multiclass','macro_f1','HYP-078-02-outer-default-sprint','HYP-075-02-multiclass-clean-runtime-continuation',.6495311620074445)]
    for kind,metric,ivan_name,sergey_name,old in pairs:
        ours=read_run(ROOT,ivan_name,metric,kind);sergey=read_run(SNAPSHOT,sergey_name,metric,kind)
        assert np.array_equal(ours['positions'],sergey['positions'])
        if metric=='rmse':np.testing.assert_allclose(ours['y'],sergey['y'],rtol=0,atol=1e-10)
        else:assert np.array_equal(ours['y'],sergey['y'])
        assert np.array_equal(ours['fit'],sergey['fit'])
        gain=lambda a,b:(a-b)/a if metric=='rmse' else (b-a)/a
        tasks[kind]={'metric':metric,'original_windows_control':old,
            'our_best':ours['score'],'sergey_best':sergey['score'],
            'our_gain_vs_original':gain(old,ours['score']),
            'our_nominal_gain_vs_sergey':gain(sergey['score'],ours['score']),
            'nominal_at_least_five_percent_vs_sergey':gain(sergey['score'],ours['score'])>=.05,
            'same_fit_valid_positions':True,'target_comparison_absolute_tolerance':1e-10 if metric=='rmse' else 0,
            'max_target_numeric_delta':float(np.max(abs(ours['y']-sergey['y']))) if metric=='rmse' else 0,
            'ours_source':ours['source'],'sergey_source':sergey['source'],
            'controlled_cross_environment_comparison':False}
    sergey_baseline=json.loads((SNAPSHOT/'baseline.json').read_text(encoding='utf8'))
    for kind,task in tasks.items():
        control=sergey_baseline[kind]['previous_same_environment'];value=task['sergey_best']
        task['sergey_same_environment_control']=control
        task['sergey_gain_vs_his_control']=(control-value)/control if kind=='regression' else (value-control)/control
    archive=ROOT/'dist/solution_sergey_panel_v1_24e3311.zip'
    manifest=json.loads((SNAPSHOT/'results/2026-10-24/sergey-improvement-v1/package-manifest.json').read_text(encoding='utf8'))
    assert sha(archive)==manifest['sha256'] and archive.stat().st_size==manifest['bytes']
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None and z.namelist()==manifest['root_files']
        for name,digest in manifest['source_sha256'].items():assert hashlib.sha256(z.read(name)).hexdigest()==digest
        assert z.read('requirements.txt')==(ROOT/'requirements.txt').read_bytes()
    result={'status':'passed','model_fits':0,'new_hypotheses_tested':0,'tasks':tasks,
        'sergey_commit':'24e3311f6ba958a67c740d5b0328cc34cb7e15d7','shared_original_commit':'4283105bc445d6511680901a714687300b77d29d',
        'archive':str(archive.relative_to(ROOT)),'archive_sha256':sha(archive),'archive_bytes':archive.stat().st_size,
        'requirements_exact':True,'evidence_sha256':evidence,
        'real_api_test':{'status':'not_run','reason':'OPENAI_BASE_URL and exact API model id unavailable; supplied key not transmitted'},
        'platform_rank_independently_verified':False,
        'limitations':['Our saved runs Windows, Sergey runs Mac; identical splits do not remove environment differences.',
            'Our H78 freezes historical H48 meta/calibration; Sergey fresh task-local OOF stack has no embedded learned state.',
            'All public splits repeatedly viewed; no hidden-platform metric conclusion.',
            'HYP numbers collide between branches; names must be qualified by branch.']}
    (OUT/'comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    lines=['# Сравнение Ivan research и Sergey published','',
        'Это сверка готовых прогнозов без обучения. Сергей развивает опубликованную базу4283105; наша новая серия пока находится в рабочем дереве.','',
        '| Метрика | Исходная база Windows | Наш лучший research | Sergey Mac | Наш номинальный выигрыш над Sergey |','|---|---:|---:|---:|---:|']
    for t in tasks.values():lines.append(f"| {t['metric']} | {t['original_windows_control']:.12g} | {t['our_best']:.12g} | {t['sergey_best']:.12g} | {t['our_nominal_gain_vs_sergey']:.3%} |")
    lines+=['','Ни одна метрика не даёт даже номинального5% выигрыша над новой веткой Сергея. От исходной базы наш RMSE улучшился на13.15%, F1 на1.51%; Сергей на своей среде улучшил RMSE на12.31%.',
        'Сравнение Windows/Mac не является контролируемым экспериментом; прирост места в рейтинге предоставлен пользователем и отдельно не проверялся.','',
        '## Направления по веткам','',
        '- Sergey H70: профиль без ELEVATION с прежней памятью; H71: надёжность последнего наблюдения; H72: причинная динамика; H73: fixed50/50 history+dynamics; H74: ExtraTrees по прошлым состояниям; H75: fresh grouped OOF stacking.',
        '- Ivan H70: reject-collision latest-past memory с rating transport; H83: original-MSE isotonic transport; H85: conditional lognormal mean; H91: inner-learned pool этих методов; H78: posterior calibration+expected Macro-F1.',
        '- Остальные Ivan H50/H57/H71–H104 имеют отдельные отчёты. H104 прервана по900sec без метрики; такие исходы не считаются проверенной неудачной моделью.',
        '- Коллизии номеров не означают повтор метода. Близкие направления относятся к одному семейству; новые проверки требуют сравнения механизма, а не номера.','',
        '## API','',
        'PDF называет Qwen3.8-27B, но OPENAI_MODEL — точный id развёртывания, а OPENAI_BASE_URL — адрес сервера организаторов; они не заданы. Пользователь сообщил, что адреса у него нет. Ключ не отправлялся ни на один сервер и не записан в отчёт. Реальная API-проверка не выполнена.','',
        'Архив взят побайтно из sergey_branch, SHA и17rootfiles сверены с его манифестом. Детальные значения и SHA: [comparison.json](comparison.json).','']
    (OUT/'report.md').write_text('\n'.join(lines),encoding='utf8')
    print(json.dumps({'status':'passed','tasks':tasks,'archive':str(archive)},ensure_ascii=False))


if __name__=='__main__':main()
