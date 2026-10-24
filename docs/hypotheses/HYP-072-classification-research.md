# HYP-072–074: classification research before SPEC011

## Допуск SPEC011 до первого fit

H72 реализована как **teacher-residual MLP128ReLU→10**, zeroW2 сохраняетteacher дообучения; .5CE+.5softmacroF1,40fullbatchepochs Adam.003/.9/.999, L2.0001,threads2. Teacherinner — совместимый RAWcachedfold0; outer — H48fullOOFmetaсготовымиouterprobabilities. Это proxyinner screening, не независимаяinnerоценка всей H48pipeline; outertrainingteacher meta-in-sample явноотмечен. Полныеградиенты провереныfinite differences доданных.

H73 Hadamard16/15bits, LightGBMbinary300/lr.05/leaves31/min20/lambda1, class inverse-frequency weights,threads2; каждыйheadотдельноclaimbeforefit. H74 CatBoost residualbaseline(logteacher) наrawX+teacherlogp,400/depth5/lr.05/lambda10, inverse-sqrtclassweights,threads2. Предобработкаfitonly,norepeatedbaselinefit. Подробныепараметрыsourceconfiguration.json сохраняютсядоfit.

Sourcefreeze/ledgerbeforefit/900secлимит обязательны. Наouter допускается заранее выбранныйфиналистзадачи, не всеinnerкандидаты; ужеизвестныйouter не объявлятьновымprivate доказательством.

Дата: 2026-10-24, Москва. Статус: **предложения, реальные model fits не выполнялись**. Исследование до SPEC011; регистрация и разрешение fit остаются у основного исполнителя. Общий HYPOTHESES.md этим документом не изменяется.

Прочитаны Codex.md, полный HYPOTHESES.md, карточки H11–20/H22–41/H42–48/H49–68, словарь multiclass и сохранённый разбор ошибок H48. Новизна сверена с OVR H18, normal/attack hierarchy H19, duplicate weights H20, transforms H25–27, feature selection H28, posthoc multipliers H37, SMOTE H38, fixed hardness weights H39, pair expert H40, LR stacking H48, distillation H49, Balanced Softmax H51, mixed-distance NN H52 и masking H60. Focal H50 пока не проверена, но это существующий ID, а не новая гипотеза.

## Наблюдение, меняющее приоритет

H48 outer Macro-F1=.6495311620074445; целевой относительный рост20% означает F1>=.7794373944089334. В сохранённом classification-report.json: Analysis .36517, Backdoor .21801, DoS .47502, Exploits .69843. Описательные outer ошибки не используются для настройки новых коэффициентов.

Отдельная read-only диагностика только на прежней **outer-fit**, без fit моделей и без чтения outer меток для расчёта: 70,529 строк, 43,387 уникальных наборов 42 признаков; 1,167 групп одинаковых X имеют разные метки и содержат 11,984 строк. ID и target исключены из группировки. Число уникальных raw-X через drop_duplicates и число pandas-хешей совпало: 43,387; коллизий различных raw-X в этом наборе не обнаружено.

Для класса k обозначим n_g размер exact-X группы, c_gk число меток k, N_k суммарную поддержку. Любой детерминированный классификатор без ID/порядка строк предсказывает одну метку всей группе. Независимая оптимистичная граница каждого F1:

`U_k = max_S 2 sum(g in S)c_gk / (N_k + sum(g in S)n_g)`.

Оптимум достигается одним из префиксов групп, отсортированных по c_gk/n_g. Для upper bound разрешаем разным классам одновременно выбирать одни группы; это ослабляет ограничение взаимной исключительности. Поэтому mean(U_k) — верхняя граница train empirical Macro-F1 для данного X, но **не** точный достижимый максимум и **не** upper bound validation/test F1.

| Класс | U_k |
| --- | --- |
| Analysis | .4549763033175355 |
| Backdoor | .4698672114402451 |
| DoS | .6000759589821496 |
| Exploits | .8329161451814768 |
| Fuzzers | .9492494639027877 |
| Generic | .9960518198642813 |
| Normal | .9977061666258379 |
| Reconnaissance | .9108402822322001 |
| Shellcode | .9988876529477196 |
| Worms | 1.0 |
| Средняя оптимистичная граница | **.8210571004494234** |

Вывод: обещать F1=.8 у Analysis/Backdoor по имеющимся X нельзя. 20% улучшение общего F1 не исключено этой слабой train-границей, но требует приблизиться к ней сразу по многим классам. Ошибочные точные дубликаты не доказывают испорченную разметку: разные атаки могут иметь одинаковые наблюдаемые flow-признаки. Сильное фокусирование на каждой ошибке может усиливать неразрешимые конфликты.

Воспроизводимая команда диагностики (PowerShell here-string -> .venv-ml/Scripts/python.exe -, без model fit):

```python
import numpy as np, pandas as pd
frame = pd.read_csv('data/multiclass/train.csv')
split = np.load('results/2026-10-23/HYP-048-01-multiclass-combinations/splits.npz')
frame = frame.iloc[split['fit']].reset_index(drop=True)
raw = frame.drop(columns=['id', 'attack_cat'])
group = pd.util.hash_pandas_object(raw, index=False)
assert group.nunique() == len(raw.drop_duplicates())
counts = pd.crosstab(group, frame.attack_cat)
sizes = counts.sum(axis=1).to_numpy()
upper = {}
for label in counts:
    positive = counts[label].to_numpy()
    order = np.argsort(-positive / sizes)
    upper[label] = float(np.max(2 * np.cumsum(positive[order]) /
        (positive.sum() + np.cumsum(sizes[order]))))
print(upper, np.mean(list(upper.values())))
```

Первый технический вызов диагностики ошибочно ожидал split key train, завершился KeyError без изменения данных и model fits. Повтор чтения использовал фактический key fit. Научных модельных запусков не было.

## HYP-072 — Связанный soft Macro-F1 training objective

**Утверждение.** Прямая оптимизация гладкой confusion-matrix метрики может перераспределить ошибочные DoS/Exploits/редкие классы лучше независимой CE, class weights и posthoc multipliers.

`P=softmax(Z)`, `T_k=sum_i Y_ik P_ik`, `D_k=sum_i Y_ik + sum_i P_ik + eps`, `Fsoft=mean_k 2T_k/D_k`. Замороженный objective: `L=.5*mean_CE + .5*(1-Fsoft)`, eps=1e-8. Применяется full training batch, чтобы метрика не зависела от случайной встречаемости Worms в mini-batch.

Точная производная `A_ik=dL_F/dP_ik = -(2/K)*(Y_ik*D_k-T_k)/D_k**2`; `dL_F/dZ_ij=P_ij*(A_ij-sum_k P_ik A_ik)`. CE добавляется в согласованном масштабе. До любого real fit обязательна центральная finite-difference проверка всех компонент на маленькой синтетике, включая противоречивые точные X и отсутствующий класс.

**Новизна:** градиент зависит от общей текущей confusion matrix; не фиксированный вес H39, не focal H50 и не готовое решающее правило H37. Поддержка редкого класса не увеличивается синтетикой.

**Реализация/риск:** soft F1 неаддитивна по строкам, не выпукла; обычную CE Hessian нельзя выдавать за её точную Hessian. XGBoost прямо перечисляет аддитивность и свойства Hessian как требования custom objective. Предпочтительный строгий вариант — небольшой NumPy MLP с точным full-batch градиентом и Adam; возможен явно обозначенный эксперимент functional-gradient boosting с постоянной положительной curvature surrogate, без обещаний Newton/convergence. Перед регистрацией SPEC011 выбрать ОДИН вариант и зафиксировать точные ширины/итерации. Torch в окружении отсутствует. Изменение optimizer после результатов — другая конфигурация, не техническое продолжение.

**Предлагаемый scope:** NumPy MLP 128 ReLU ->10 softmax, log1p для неотрицательных numeric, fit-only StandardScaler/OHE, clip standardized numeric ±8, L2=1e-4, Adam lr=.003,40 full-batch epochs, seed42. Для времени BLAS threads=2; max900s, checkpoints по эпохам; не выбирать эпоху по outer. Сначала training-only inner, затем одна outer оценка только при соответствии заранее утверждённому продвижению. Фиксированный бюджет может оказаться недостаточным; это честный технический/отрицательный исход.

**Критерий:** на исходном outer превзойти .6495311620074445; целевой результат >=.7794373944089334. Дополнительные precision/recall/confusion описательные. Новый split нельзя сравнивать с прежним числом.

**Первичный источник:** [sigmoidF1](https://arxiv.org/abs/2108.10566) обосновывает гладкий surrogate confusion-matrix метрики в multilabel; наш single-label softmax objective — явная адаптация, статья не доказывает наш выигрыш. [XGBoost custom objectives](https://xgboost.readthedocs.io/en/release_3.0.0/tutorials/advanced_custom_obj.html) объясняет ограничения Hessian и аддитивности; установленную2.1.4 проверять локально.

## HYP-073 — Dense error-correcting output codes

**Утверждение.** Несколько задач распознавания сбалансированных подмножеств классов дадут избыточное представление метки; совместное декодирование может исправлять ошибки отдельных бинарных разделителей.

**Метод:** K классов сортируются лексикографически; для K=10 взять первые10 строк Hadamard16 без постоянного столбца, получая15 бит. Каждый бит делит классы на две группы и обучается на ВСЕХ training-строках. Бинарные LightGBM: n_estimators=300,lr=.05,num_leaves31,min_child_samples20,reg_lambda1,threads2; никаких OVR или pair-only подмножеств. В каждом бите вес строки `N/(K*n_class)` с нормировкой среднего1, чтобы суммарный вес каждого исходного класса одинаков. Decoding — средняя Bernoulli cross-entropy между15 предсказанными вероятностями и каждой кодовой строкой, argmin; probabilities clip1e-6. Кодовая матрица фиксируется до fit и проверяется на уникальность/расстояния. Для другого K использовать ближайшую степень2 и заранее ограничить число бит31.

**Новизна:** H18 имела один отдельный положительный класс на head, H40 обучала только пару; здесь каждый head решает разные объединения >=2 исходных классов и final label декодируется по всему коду. Избыточность должна уменьшать отдельные ошибки, но коррелированные ошибки heads и конфликтные X этим не устраняются.

**Бюджет:** максимум15 inner+15 outer бинарных fit, общий900s. Inner один fixed grouped holdout, не новый baseline. Сохранить каждый head fingerprint; если бюджет не помещается, не уменьшать bits после outer и не объявлять результат проверенным. На полных данных все15 heads составляют один классификатор, а не15 независимых гипотез.

**Критерий:** те же F1 пороги; дополнительно сравнить per-class confusion. Источник: [Dietterich & Bakiri, 1995](https://arxiv.org/abs/cs/9501101); статья показывает метод и исторические результаты, не гарантирует улучшение UNSW.

## HYP-074 — Условная нелинейная поправка logits

**Утверждение.** Одинаковые base probabilities могут требовать разных поправок при разных proto/service/state/traffic contexts. H48 LR log-probabilities не может выучить зависимость коэффициента от X.

**Метод:** использовать сохранённые совместимые grouped OOF исходных raw/smote/frequency branches, не refit старых branches. Взять средние log probabilities как initial logits и обучить ограниченный residual multi-class GBDT на OOF rows с признаками X+30 logp+entropy/top-margin. При final inference residual добавляется к logits трёх уже обученных branches. Train-only feature maps. Число деревьев150,depth3,lr.03,L2=5,коэффициент residual1 фиксируются до запуска. Метаданные cache должны доказать SHA, outer-fit positions, отсутствие outer valid в branches и OOF prediction для каждой training строки.

**Новизна:** это X-условные нелинейные поправки, не ещё один LR stack либо новые константные веса. Отличие должно сохраняться в коде; если получится только `LR(logp)`, это запрещённый повтор H48. H49 использовала teacher distillation, здесь base branches остаются при inference.

**Ограничение:** честный inner score всей процедуры требует nested OOF: мета-валидационная строка не должна попадать в training какой-либо части графа. Обычная CV по уже созданным OOF может давать косвенную утечку через взаимные training labels, поэтому её нельзя выдавать за независимый selection score. При недостаточном nested cache допустим один заранее frozen outer fit/evaluation нового residual; старые outer labels не выбирают параметры. Если root SPEC011 запрещает такой протокол, H74 остаётся proposed.

**Бюджет:** новый residual1–3 fits; без скрытого повторного base fit. Время до900s. Критерии прежние. Теоретический мотив — interaction features и conditional residual modeling, без ссылки на якобы доказанный эффект именно такой схемы.

## Приоритет и вывод до запуска

Рекомендуются H72, затем H73. H74 только после cache/provenance проверки. H50 focal является отдельным существующим предложением: из-за конфликтных точных X нет достаточного основания предпочесть её только по rareclass recall. Все три новые гипотезы **не оценивались**. Диагностический bound не доказывает невозможность цели и не разрешает использование ID/внешних истинных меток. Исходная SHA train:171a8915070b7575b729a32f70a45d1ad7b6f43c65b4ce12a0a68af30d5cb4e0; outer SGKF5 fold0 seed42,70529/17141.

## Запуски

Реальные модели:0. До регистрации SPEC011 и отдельных results/ карточек fit не выполнять; каждый будущий исход внести в общий журнал основным исполнителем сразу после проверки.

## Уточнение до real fit: замороженная реализация H72/H73

2026-10-24, после регистрации [SPEC011](../specs/SPEC-011-three-hour-improvement.md) / [PLAN011](../plans/PLAN-011-three-hour-improvement.md). Этот раздел заменяет первоначальный standalone scope H72 выше до единого научного запуска.

H72 реализована как **teacher-residual NumPy MLP**: initial logits=`log(H48_teacher_p)`; residual128ReLU→K; последний linear слой и bias нулевые, поэтому при0epochs prediction совпадает с teacher. Вход MLP — fit-only median/log1p для неотрицательных training numeric, standardization/clip±8/OHE плюс K `log(clip(teacher_p,1e-6,1))`. Initial teacher logits clip1e-12; query teacher обязателен, если training teacher задан. Классы в teacher должны совпадать с `np.unique(y)`; caller проверяет provenance и alignment. Если teacher не дан, используется заранее фиксированный uniform fallback, без скрытого fit teacher.

Objective=.5meanCE+.5(1−softmacroF1)+.5L2(sumW1²+sumW2²), L2=1e-4; Adam(.9,.999,epsilon1e-8),lr=.003,40fullbatch epochs,seed42,BLASthreads2. СтартовыйW1 HeNormal; b1=0; W2/b2=0. Число эпох по validation не подбирается. Training loss history и in-sample predictions H48 meta не называются independent OOF score. Это отличается от CE student distillation H49 сохранением teacher при inference, нулевой residual и связанным F1-loss.

H73: Hadamard16 первые10строк,15непостоянных столбцов; каждый head отдельный LGB300/lr.05/leaves31/minchild20/lambda1/seed42/threads2. Каждая training строка имеет весN/(K*nclass), mean1; decoder argmin meanBernoulliCE (prob clip1e-6). Моно-классный бит, если возникнет для другогоK, сохраняется как constant без fit. До31бит, K<=32; свыше — явная ошибка. Для registration каждого actualhead caller предоставляет `fit_hook(model,X,y,configuration,**fit_kwargs)` и передаёт sample_weight. Если COMBO_RUN_DIR установлен, отсутствие hook вызывает ошибку, а не незарегистрированное обучение.

Реализация: `sprint_classification.py`. Синтетические проверки `tests/test_sprint_classification.py`: exact objective finite differences включая absentclass; сохранение teacher приzeroresidual; reduction train loss и разделение простых данных; fit-only encoder/unseen category; code pairdistances8 и точное декодирование; реальные малые LGBheads ccallback; сериализация с исключением runnercallback. Это unit fits на синтетике, не оценка гипотез на publicdata. Точное время/команды проверки переданы координатору; окончательный sourceSHA фиксирует runner после доработок.

SPEC011 result 2026-10-24T01:53:49.517515+03:00 HYP-072/inner/default: completed macro_f1=0.6090378831239697 vscontrol=0.5873380697311004; [report](../../results/2026-10-24/HYP-072-01-inner-default-sprint/report.md). Повтор этой конфигурации запрещён.

SPEC011 result 2026-10-24T01:55:41.158516+03:00 HYP-073/inner/default: completed macro_f1=0.5803974034255268 vscontrol=0.5873380697311004; [report](../../results/2026-10-24/HYP-073-01-inner-default-sprint/report.md). Повтор этой конфигурации запрещён.

SPEC011 result 2026-10-24T01:58:44.104213+03:00 HYP-074/inner/default: completed macro_f1=0.6143385510752261 vscontrol=0.5873380697311004; [report](../../results/2026-10-24/HYP-074-01-inner-default-sprint/report.md). Повтор этой конфигурации запрещён.
