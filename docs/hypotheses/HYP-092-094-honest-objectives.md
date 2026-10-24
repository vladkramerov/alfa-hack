# HYP-092–094: неизменённые residual objectives с честным training OOF teacher

- Дата: 2026-10-24, Москва.
- Спецификация: [SPEC011](../specs/SPEC-011-three-hour-improvement.md), REQ1101–1104, REQ1107–1109.
- Статус каждого ID: Запланирована; научные fit этой серии ещё не запускались.

## Утверждение и новизна

H72/H77/H80 обучали correction на in-sample raw teacher probabilities своих
training rows, а query teacher был held-out. Новая комбинация оставляет objective,
representation и learner прежними, но заменяет training teacher на уже
построенные H88 nested grouped-OOF вероятности: каждый training target исключён
из teacher, сформировавшего его строку. Предположение — устраняется mismatch
teacher confidence между training/query и улучшается перенос correction.

Это новый совместный training protocol, не новый seed/имя старого fit. H88
использует такую честность с contextual CE CatBoost; H92/H93/H94 исследуют
отдельно точные softF1 trees, Lovasz rank trees и softF1 residual MLP. Ни один
новый base teacher не обучается: H88 caches только read-only reuse.

## Frozen методы и заранее объявленная небольшая сетка

- **H92** `HonestMacroTeacherBoost`: прежний H80 exact `.5meanCE+.5(1-softMacroF1)`
  LightGBM. До первого H92 fit фиксируются два варианта: **300 и900rounds**.
  Прочее как H80: lr.05/leaves63/minchild30/lambda5/threads2; точный fullbatch
  gradient*N, положительный surrogate Hessian `max(.5*p*(1-p),.001)`, не exact.
- **H93** `HonestLovaszTeacherBoost`: прежний H77 `.5CE+.5LovaszSoftmax`,
  **200rounds единственный вариант**, прочие параметры H77 неизменны.
- **H94** `HonestResidualF1Classifier`: прежний H72 objective/MLP hidden128,
  lr.003/l2=.0001/fullbatch/zero last layer/threads2. До первого H94 fit
  фиксируются **40 и150epochs**, остальные параметры прежние.

Сетка объявлена координатором до первого realfit этих новых honest algorithms;
варианты выбираются только по inner, суммарно≤900сек **на каждый ID**, включая
возможный единственный outer. Никаких outer-сеток или изменения алгоритма
между вариантами. Constructor kwargs передаются прежним superclass;
configuration отражает фактические iterations/epochs. У MLP configuration
сохраняет форму dict property, у деревьев — method.

Preprocessing у всех прежний train-only median/log1p nonnegative/
standardization/clip8/OHE. Teacher init logits и inference query teacher
складываются по прежнему контракту; градиенты/objectives не переписываются.
H48 meta state, synthetic X, ID и external data не добавляются.

## Read-only provider и fail-closed provenance

`experiments.sprint_honest_reuse.load_reused_honest_teacher(bundle,stage,fit,valid,
root=...,output=...)` никогда не fit и не пишет output/cache.

Inner source: `results/2026-10-24/HYP-088-01-inner-default-sprint/honest-raw-oof/`.
Для каждого из3fold проверяются immutable started/configuration, completed
outcome, prediction artifact SHA, exact local/global splits, SGKF3_seed42 raw-X
groups disjoint, sorted class order, X/y digests и bundle dataSHA. Полное
покрытие каждой training строки ровно одним prediction проверяет assemble_oof.
Query — прежний held-out raw fold0 cache, exact исходные positions/classes.

Historical source hashes сверяются с **старым H88 frozen.json/source snapshot**,
включая целостность всех frozen source files; started6source hashes должны
совпадать с ним. Изменения текущего production automl_models.py не являются
причиной нового teacherfit и не сравниваются с old source. Неполный,
незавершённый, изменённый или несовместимый cache отклоняется без fallback fit.
Завершение самого H88 correction не требуется, если3teacher caches completed.

Outer provider вызывает существующий H88 loader только с stageouter и
fit_hook=None: read-only original raw OOF training + H11 held-out query.
Контроль H48/H78 root сохраняет отдельно и не смешивает с raw teacher score.

## Данные и критерии до запуска

Classification trainSHA256
`171a8915070b7575b729a32f70a45d1ad7b6f43c65b4ce12a0a68af30d5cb4e0`.
Inner исходный raw fold0: fit47075/valid23454, дополнительно SGKF3 внутриfit.
Готовый raw teacher MacroF1 `.5873380697311004`; H78inner `.6288481661155734`
учитывается при выборе финалиста. Utility — strict inner improvement; честный
training teacher сам по себе не является доказательством улучшения метрики.

Outer исходный grouped70529/17141, только после inner выбора ID/варианта до
score. Готовый H48 `.6495311620074445`, H78 `.6593084142037399`, цель20%
`>=.7794373944089334`. Старый outer уже открыт, оценки разведочные. Новый
вариантный fit учитывается отдельно, отрицательные результаты сохраняются.

## Техническая проверка до fit

TDD: первоначальные tests упали при отсутствии модулей, затем7/7green.
Проверены ручная OOF assembly,0fits/nooutput, неизменённые objectives/parameters,
config300/900 и40/150, frozen oldsource strict/current unrelatedsource allowed,
missing/incomplete/byte-mutated caches, class order, Xdigest/globalpositions,
rawgroup collision. Synthetic cache fixtures не выполняют learnerfit.

Разрешённая nofit проверка реальных H88 caches дала train47075x10/query23454x10,
model_fits0, trainSHA
`a8827d2e8a4643b5b24a6876f1a59bb56b5f8df1c772404874bc258c3820a161`,
точно совпадающий с сохранённым H88 teacher-audit.json.

Команда: `.venv-ml/Scripts/python.exe -X utf8 -m unittest tests.test_sprint_honest_objectives -v`.

## Запуски

| ID | Отчёт | Статус | Вывод |
| --- | --- | --- | --- |
| H92 | Координатор создаёт отдельный results/ до fit каждого варианта | Не запускалась | Не оценивалась |
| H93 | Координатор создаёт results/ до fit | Не запускалась | Не оценивалась |
| H94 | Координатор создаёт отдельный results/ до fit каждого варианта | Не запускалась | Не оценивалась |

## Следующий шаг

Координатор регистрирует новые совместные протоколы и всю сетку, фиксирует
source/config/data/splits/teacherSHA, запускает ровно заранее объявленные inner
варианты, сохраняет каждый outcome. Provider/wrappers общий ledger не меняют.

SPEC011 result 2026-10-24T02:36:47.071733+03:00 HYP-092/inner/default: completed macro_f1=0.6075155330908212 vscontrol=0.5873380697311004; [report](../../results/2026-10-24/HYP-092-01-inner-default-sprint/report.md). Повтор этой конфигурации запрещён.

SPEC011 result 2026-10-24T02:40:16.964501+03:00 HYP-092/inner/extended: completed macro_f1=0.599340154183666 vscontrol=0.5873380697311004; [report](../../results/2026-10-24/HYP-092-02-inner-extended-sprint/report.md). Повтор этой конфигурации запрещён.

SPEC011 result 2026-10-24T02:37:21.769093+03:00 HYP-093/inner/default: completed macro_f1=0.6002777710371685 vscontrol=0.5873380697311004; [report](../../results/2026-10-24/HYP-093-02-inner-default-sprint/report.md). Повтор этой конфигурации запрещён.

SPEC011 result 2026-10-24T02:36:54.439468+03:00 HYP-094/inner/default: completed macro_f1=0.6192492910174916 vscontrol=0.5873380697311004; [report](../../results/2026-10-24/HYP-094-02-inner-default-sprint/report.md). Повтор этой конфигурации запрещён.

SPEC011 result 2026-10-24T02:38:09.649730+03:00 HYP-094/inner/long: completed macro_f1=0.6223414898157842 vscontrol=0.5873380697311004; [report](../../results/2026-10-24/HYP-094-03-inner-long-sprint/report.md). Повтор этой конфигурации запрещён.
