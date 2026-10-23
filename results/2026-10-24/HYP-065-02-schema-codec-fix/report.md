# HYP-065 / техническое исправление codec

Дата: 2026-10-24, Москва. Статус до изменения: **Начато по отдельному разрешению после десяти terminal outcomes SPEC009**. [Исходная диагностика](../HYP-065-01-schema-semantics/report.md) сохранена без изменения. Это устранение установленной ошибки, не повтор ML-гипотезы и не новая метрика.

Разрешённые изменения: `automl_data.py`, тесты codec/schema; ведущий агент отдельно обновляет два CSV-read в `solution.py`. API: `read_submission(bundle, path)` сохраняет лексические ID/классы, numeric outputs остаются числовыми; atomic round-trip использует тот же API. Обычные integer ID и regression target не меняют тип; ID `01`/`1` не сливаются до discovery. Пустой target отвергается, строковые literal `NA`/`+1`/`001` сохраняются при объявленной строковой семантике.

До реализации наблюдены два RED schema-теста; новый набор `tests.test_csv_codec` RED из-за отсутствующего `read_submission`. План проверки: восемь исходных synthetic metamorphic cases, дополнительные codec tests, старые unit tests и точное сравнение public parsed data/schema/hash/split без model fits. Число public model fits = 0.

## Завершение технического исправления

Статус: **Проверено, исправление подтверждено в проверенных условиях**. Восемь исходных сценариев теперь GREEN при использовании общего typed reader на выходной границе; восемь дополнительных cross-dtype тестов также GREEN. Исходный диагностический script/run 01 не изменён: unit-контроллер подставляет новый production API только на границе чтения результата, вместо старого произвольного pandas inference.

`load_task` сохраняет исходные строковые target/ID при объявленной строковой семантике или необратимой CSV coercion; ID восстанавливается до определения уникальности. Пустые target отвергаются, literal `NA` при строковом объявлении сохраняется. Probability/regression output остаётся numeric, CSV boolean text вместо numeric prediction отвергается. Обычные numeric ID и regression target не переводятся в строки. `atomic_submission` использует экспортируемый `read_submission(bundle, path)`; два caller-read в `solution.py` изменяются ведущим агентом отдельно.

- [Полная проверка](verification.json): **205/205 unit tests**, отдельные codec/schema **16/16**.
- [Codec tests log](codec-tests.log), [полный старый и новый suite](all-tests.log).
- [Воспроизводимая no-fit сверка](verify_fix.py): `.venv-ml/Scripts/python.exe -X utf8 results/2026-10-24/HYP-065-02-schema-codec-fix/verify_fix.py`.
- Обе публичные задачи прочитаны старым frozen SPEC009 и новым loader; `train/test/sample/dictionary/X/X_test/y`, типы и порядок **побитово/по точным значениям совпали**. SHA CSV неизменны; новые `validation_split` совпали со старыми и сохранёнными outer positions. Ключи split и digests в verification.json.
- Public model fits **0**, новых public metrics **0**; all-unit suite использует только уже предусмотренные синтетические технические fits.
- Снимки изменённых разрешённых исходников и SHA сохранены в `source/` и verification.json. Первоначальный отрицательный H65 report/evidence не изменён; этот run — отдельное исправление, не попытка скрыть отрицательный исход.

Ограничение: интеграция обоих `solution.py` CSV-read принадлежит ведущему агенту; её итоговая flat ZIP проверка выполняется отдельно. Улучшение private score этим исправлением не доказано.
