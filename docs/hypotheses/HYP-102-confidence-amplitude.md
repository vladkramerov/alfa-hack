# HYP-102: amplitude calibration коррекции H91 относительно H69

- Дата:2026-10-24, Москва.
- Спецификация: [SPEC011](../specs/SPEC-011-three-hour-improvement.md), REQ1101–1104, REQ1107–1109.
- Статус: заранее фиксированный protocol; реальные метрики/parameter fits worker не запускает. Root регистрирует перед inner.
- Основание: прямое поручение координатора проверить scalar confidence/amplitude calibration только по cached predictions.

## Утверждение и новизна

H91 сочетает H70/H83/H85 simplex-весами; это может сохранять избыточное сокращение коррекции к H69. Один scalar amplitude относительно исходного H69 допускает extrapolation и проверяет величину общей поправки, не обучая learner заново. H102 не меняет компоненты/веса H91 и не повторяет H79/H91 simplex optimization: λ>1 выводит prediction за отрезок H69–H91. Порог/состав после результата не меняются.

## Один протокол до запуска

Cached base b=H69 reference из соответствующего H70high validation_predictions.npz, p=H91 prediction. Exact positions/y и H70 valid split должны совпадать. δ=p−b; no-intercept least squares amplitude `λ=clip(dot(δ,y−b)/dot(δ,δ),0,2)`. При δ≡0 выбираетсяλ1 (всеλэквивалентны). Итог `max(0,b+λδ)`; free intercept отсутствует. Аналитический optimum относится к unclipped squared loss, post-clip outcome проверяется отдельно; при отсутствии фактического выигрыша над H91 перенос запрещён.

Inner Year<4→4: H91 `HYP-091-01-inner-multiblend-sprint`, H70reference `HYP-070-03-inner-high-sprint`; готовый H69RMSE42.46005247026025, H91RMSE30.636363541581975. Outer Year<5→5: соответствующие H91-02/H70-04. Контроль primary metric — оригинальный H69; incremental comparator **H91**, strict improvement требуется для продвижения. Innereligible=`score<H91score−1e-10`; технический epsilon предотвращает roundoff-победу. Goalouter≤31.51375506213417; current H91outer34.21382789041039 не обещает достижимости.

`experiments/sprint_amplitude.py` сначала читает полный HYPOTHESES.md и требует явной HYP-102 регистрации. Создаёт уникальную run folder до вычислений: `HYP-102-01-inner-amplitude-sprint` / `HYP-102-02-outer-amplitude-sprint`. Data/source/split/prediction SHA и evidence.started сохраняются до optimizer; Memory.claim на actualcached b/p/y предшествует λ calculation. Это один parameter fit, learner fits0/old model refits0. Ошибки/negative outcomes сохраняются отдельно, history не перезаписывается.

Outer допускается только при completed/eligible inner и загружает **замороженный λ** из его metrics.json; внутренний fit_amplitude на outer запрещён. SHA inner metrics сохраняется до outer outcome. Внешний y используется только для оценки frozen prediction. Уже просмотренный outer — дальнейший adaptive exploratory search, не независимое подтверждение.

Сохраняются evidence/configuration/data/splits/source snapshot, claim/outcome, predictions (base,pool,delta,λ) и report. Script при фактическом запуске root сразу дописывает исход в общие HYPOTHESES.md/results README; worker общие файлы не меняет. Root не запускает повтор известного claim. Input data SHA verified by streaming files; raw CSV labels не нужны для обучения нового learner.

## Проверка реализации

Синтетические tests: analyticλ1.5, lower/upper clamp, zeroδ, no intercept, postclip nonnegativity, exactalignment/finite errors, Memory claim before fitting, outer reuse without optimizer. Реальные cached metrics/новые predictions worker не вычисляет. Научный исход до отдельного запуска: **Не оценивалась**.

Фактически `.venv-ml/Scripts/python.exe -X utf8 -m unittest discover -s tests -p test_sprint_amplitude.py -v`: **4/4 green**,0.276сек; полный `.venv-ml/Scripts/python.exe -X utf8 -m unittest discover -s tests -q`: **521 tests, OK**,13.657сек, три известных PyparsingDeprecationWarning от matplotlib. Все оптимизации в unit checks — синтетические; реальный main worker не вызывал.

Source SHA256 при передаче: `84d96df3003b7fe35684e02489be02db5984661b8cb2a5f804623f04eaec2ee6`. Для фактического запуска координатор использует `.venv-ml/Scripts/python.exe -X utf8 -m experiments.sprint_amplitude --stage inner`; outer запускается отдельно с `--stage outer` только после completed/eligible inner.

2026-10-24: inner RMSE30.449416831901488, amplitude1.1289189550164425; outer RMSE34.50690731292415 хужеH91 34.21382789041039. Incremental гипотеза не подтверждена; в baseline не включать и не повторять. [inner](../../results/2026-10-24/HYP-102-01-inner-amplitude-sprint/report.md), [outer](../../results/2026-10-24/HYP-102-02-outer-amplitude-sprint/report.md).
