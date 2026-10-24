# HYP-063-01: feature-only аудит близких временных профилей

- Дата архива:2026-10-24, Москва.
- Связи: [H63 proposal](../../../docs/hypotheses/HYP-049-068-proposals.md), [SPEC011](../../../docs/specs/SPEC-011-three-hour-improvement.md), последующее отдельное развитие [H81](../../../docs/hypotheses/HYP-081-fuzzy-temporal.md).
- Статус выполнения: завершена feature-only диагностика после технической ошибки сериализации первого скрипта.
- Научный вывод H63: **Недостаточно данных** для подтверждения связанных физических объектов/утечки; статистическое сходство само по себе идентичность не доказывает.
- Model fits0, metric evaluations0, новых validation predictions0. Сводный машинный артефакт: [counts.json](counts.json).

## Что проверялось

Поддержка возможных дополнительных связей после exact H70 key при малом изменении floor_area/year_built. Загружены только шесть явно перечисленных признаков, site_eui/ID/rating не читались. Важная точность: CSV содержит все годы, reader сначала считал эти шесть столбцов, затем немедленно отфильтровал Year_Factor<4. Для вычислений и выбора метода использованы только годы1/2/3; годы4/5 не участвовали. Следовательно выражение «годы≥4 не загружены» было бы неточным.

Donors всегда раньше query-года. Retained profiles имеют точные State/class/facility/area/yearbuilt и не имеют повторных наблюдений одного training-года. Приближённые кандидаты сохраняют exact State/class/facility; relative area gap≤.001; известные construction years отличаются≤1, unknown/0year требует exact area. Distance=area_gap/.001+year_gap; unknown-year distance1. Runnerup distinct profile margin≥.5. Matching не использовал y.

## Наблюдаемые counts

| Query год | Past rows | Query rows | Exact support | Любые fuzzy | Fuzzy после margin | Rejected margin | Changed area | Missing/0 year |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 2117 | 9058 | 1633 | 56 | 53 | 3 | 31 | 19 |
| 3 | 11175 | 10879 | 6981 | 516 | 423 | 93 | 373 | 14 |

423/10879 =3.89 процентного пункта feature support. Это не RMSE, не scientific score и не измерение правильности связи. Реальная идентичность требует дополнительных семантических сведений; новый classifier/transport из этого аудита не следует автоматически.

## Команды и фактическая техническая история

Оба запуска были read-only PowerShell here-string → `.venv-ml/Scripts/python.exe -X utf8 -` из корня проекта; отдельный исходный файл на момент запуска не создавался. Raw tool transcript/log отдельным файлом тогда не сохранялся. Первоначальный scalar-скрипт завершил вычисления и упал при `json.dumps`: `TypeError: Object of type int32 is not JSON serializable`; его counts не были получены как численный результат. Научных fit при этом не было.

Последующее vectorized-продолжение выполнилось exit0, observed wall_time2.2499758сек и вывело counts.json statistics. Ниже точная копия **успешного Python source из видимого tool call**, а не выдуманный путь ранее запущенного файла; wrapper был PowerShell `@'…'@ | .venv-ml/Scripts/python.exe -X utf8 -`:

```python
import json, numpy as np,pandas as pd
from sprint_regression import _keys
cols=['Year_Factor','State_Factor','building_class','facility_type','floor_area','year_built']
X=pd.read_csv('data/regression/train.csv',usecols=cols);X=X[X.Year_Factor<4].reset_index(drop=True)
keys=np.array(_keys(X)); years=X.Year_Factor.to_numpy(float);areas=X.floor_area.to_numpy(float);built=X.year_built.to_numpy(float)
blocks=np.array([str(x) for x in zip(X.State_Factor,X.building_class,X.facility_type)])
results=[]
for year in [2,3]:
 fit=np.flatnonzero(years<year);query=np.flatnonzero(years==year);profiles={}
 for i in fit:profiles.setdefault(keys[i],[]).append(i)
 profiles={k:v for k,v in profiles.items() if len(set(years[v]))==len(v)}
 indices={}
 for k,v in profiles.items():
  j=v[np.argmax(years[v])];indices.setdefault(blocks[j],[]).append(j)
 indices={k:np.array(v) for k,v in indices.items()}
 stat=dict(year=year,fit_rows=len(fit),query_rows=len(query),exact=0,extra_any=0,extra_margin=0,rejected_margin=0,extra_area_changed=0,extra_missing_year=0)
 for i in query:
  if keys[i] in profiles:stat['exact']+=1;continue
  donors=indices.get(blocks[i],np.array([],dtype=int))
  if not len(donors) or not np.isfinite(areas[i]) or areas[i]<=0:continue
  gap=np.abs(areas[donors]-areas[i])/np.maximum(areas[donors],areas[i]);known=np.isfinite(built[donors]) & np.isfinite(built[i]) & (built[donors]>0) & (built[i]>0)
  ygap=np.abs(built[donors]-built[i]);accept=np.isfinite(gap)&(areas[donors]>0)&(gap<=.001)&((known&(ygap<=1))|(~known&(gap==0)))
  pos=np.flatnonzero(accept)
  if not len(pos):continue
  stat['extra_any']+=1;scores=np.where(known[pos],gap[pos]/.001+ygap[pos],1.)
  order=np.argsort(scores,kind='stable')
  if len(order)>1 and scores[order[1]]-scores[order[0]]<.5:stat['rejected_margin']+=1;continue
  p=pos[order[0]];stat['extra_margin']+=1;stat['extra_area_changed']+=int(gap[p]>0);stat['extra_missing_year']+=int(not known[p])
 results.append(stat)
print(json.dumps({'fit_years':'<4','label_columns_loaded':False,'statistics':results},indent=2))
```

## Происхождение, ограничения и следующий шаг

Data SHA256 из project provenance: `096dac951cf44fade5417ddd0465c41beee3009663dbea34e8d464a4d3c7f775`; повторная read-only проверка SHA выполнена при архивировании отдельно, не является исторической проверкой во время первоначального скрипта. Исходный git SHA/длительность scalar-скрипта и отдельный rawstdout файл не фиксировались, их не восстанавливаем догадкой. Использованы только pandas/numpy и существующая canonical `_keys` из sprint_regression.py; этот helper не обучает модель. Архив создаётся после диагностики по поручению координатора, а не выдаётся за заранее созданный run-directory.

H63 остаётся диагностикой без доказательства утечки. Для научной проверки transport координатор выделил отдельный H81, его результат не является score этого отчёта. Root добавляет ссылки в общие HYPOTHESES.md/results README; этот worker общие реестры не меняет.
