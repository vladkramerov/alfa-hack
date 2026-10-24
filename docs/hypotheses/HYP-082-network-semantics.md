# HYP-082: семантическое представление сетевого потока для contextual correction

- Дата: 2026-10-24, Москва.
- Спецификация: [SPEC011](../specs/SPEC-011-three-hour-improvement.md), REQ1101–1104, REQ1107–1109.
- Статус: Запланирована. Реальные данные при реализации не обучались.
- Полный HYPOTHESES.md и H27 implementation/proposal прочитаны до реализации.

## Утверждение и новизна

Начальные TCP sequence numbers могут ухудшать переносимость деревьев; TTL и
направленная асимметрия могут стать полезнее после фиксированного физического
преобразования. Это предположение, не установленная причина ошибок. Новый
механизм — удалить stcpb/dtcpb из X contextual learner, добавить rowwise TTL
initial-bucket/hop fraction, signed directional imbalance и RTT consistency.
Teacher остаётся прежним supplied teacher и может нести косвенную информацию
об удалённых полях; удаляются прямые признаки только нового correction learner.

H5 делала bytes/packet и прежние ratios, а не эти bounded signed imbalances;
H16/H17 делали categorical interaction/frequency. H27 маскировала применимость
TCP по proto, H26/H21 добавляли zero/missing indicators и/или Gaussian pipeline.
Здесь нет applicability/zero flags, категориальных комбинаций, частот,
Gaussian/quantile scaling или повторного SMOTE. H74 learner сохранён точно,
меняется только исходное представление X до его fit.

## Frozen преобразование до realfit

`NetworkSemanticFeatures` хранит только train schema, без статистик X или y:

- Полностью убрать из correction X `stcpb`, `dtcpb`, если присутствуют.
- Для `sttl`, `dttl`: при конечном TTL в `[1,255]` взять минимальный из
  `[32,64,128,255]`, не меньший наблюдаемого. Добавить initial bucket и
  `(initial-observed)/initial`; исходный TTL сохранить.
- Это эвристика initial TTL, не восстановленная истинная длина маршрута.
  TTL0, отрицательный, >255, inf/missing дают NaN в двух derived полях;
  raw0 сохраняется0. Никакого дополнительного zero indicator.
- Для пар `spkts/dpkts`, `sbytes/dbytes`, `sload/dload` добавить
  `(source-destination)/(abs(source)+abs(destination)+1)`. Алгебраически
  эквивалентное масштабирование max(abs(source),abs(destination),1) предотвращает
  переполнение суммы. Это signed, bounded, dimensionless representation.
- Добавить `sloss/(spkts+1)`, `dloss/(dpkts+1)` и
  `tcprtt-(synack+ackdat)`. Inf results становятся missing; никаких оценённых
  констант/квантилей/заполнения средним. Нули остаются допустимыми значениями.
- Неизвестные/отсутствующие исходные поля не имитируются: производное поле
  создаётся только при наличии всех входов. Generated prefix `network__`
  зарезервирован, schema mismatch отклоняется. Порядок/индекс строк сохранены.
- Numeric inf в сохраняемых исходных numeric колонках → NaN. Категориальные
  значения передаются неизменными в существующий native categorical protocol.

## Frozen модель и teacher

`sprint_network.NetworkSemanticTeacherCorrector(threads=2)` вызывает неизменённый
`ContextualTeacherCorrector`: CatBoost400/depth5/lr.05/l2_leaf_reg10,
MultiClass/random_seed42, boost_from_averageFalse, no writing files,
inverse-sqrt class counts mean-normalized weights. Никакого подбора параметров.
Training baseline — explicit log teacher; inference складывает raw tree logits
и query teacher logits по контракту H74, затем softmax. Teacher columns строго
соответствуют sorted training classes; provenance проверяет caller.

Новых baseline/teacherfit нет; один correction learner, до900сек суммарно на
гипотезу и до4CPU, замороженный default2CPU. Интерфейс: `.fit(X,y,p)`,
`.predict_proba(X,p)`, `.predict(X,p)`, `.configuration()`, `.audit_`.

## Данные и критерий успеха — до запуска

Classification trainSHA256
`171a8915070b7575b729a32f70a45d1ad7b6f43c65b4ce12a0a68af30d5cb4e0`.
Inner — существующий raw OOF fold0 fit47075/valid23454. Готовый raw teacher
MacroF1 `.5873380697311004`, сопоставимый H74 `.6143385510752261`.
Успех representation utility — strict improvement над H74 при том же teacher
и split; лучший текущий H78 inner `.6288481661155734` учитывается при выборе
финалиста. Meta-in-sample training teacher score не считается независимым.

Original grouped outer70529/17141 допускается только после выбора финалиста до
score. Готовый H48 `.6495311620074445`, текущий H78 `.6593084142037399`;
цель20% `>=.7794373944089334`. Старый outer открыт и остаётся разведочным,
меньший выигрыш не объявлять выполнением20%. Source/config/data/splitSHA и
started/outcome/predictions/report фиксирует координатор; отрицательный исход
сохраняется без последующей подгонки семантики или параметров.

## Техническая проверка

TDD:4tests сначала упали с отсутствующим модулем, затем4/4green. Проверены
ручные TTL buckets/hops/zero/missing и ratios/RTT, отсутствие raw sequence
признаков, неизменность source/category/index, schema collision, batch
invariance, pickle, real synthetic learner и explicit teacher contract.

Команда: `.venv-ml/Scripts/python.exe -X utf8 -m unittest tests.test_sprint_network -v`.
Synthetic learner fit не является scientific проверкой на competition data.

## Запуски

| Дата | Отчёт | Статус выполнения | Вывод |
| --- | --- | --- | --- |
| — | Координатор создаёт results/ до fit | Не запускалась | Не оценивалась |

## Следующий шаг

Регистрация координатором в общем журнале и один frozen innerfit. Модуль не
изменяет ledger/harness, исходные scientific sources или действующую поставку.

SPEC011 result 2026-10-24T02:18:48.716619+03:00 HYP-082/inner/default: completed macro_f1=0.6150544797423486 vscontrol=0.5873380697311004; [report](../../results/2026-10-24/HYP-082-01-inner-default-sprint/report.md). Повтор этой конфигурации запрещён.
