# SPEC010: проверка clean CLI

Статус: **passed**. Синтетических сценариев пройдено: 4/4.

Источник требований: [SPEC010](../../../docs/specs/SPEC-010-clean-rewrite.md), REQ1001–1003, REQ1010–1012.
Независимая копия корневых solution.py (из rewrite_solution.py), research_*.py и byte-exact requirements.txt находится в source/. Сценарии: RMSE, macro-F1, binary log_loss (вектор), multiclass log_loss (нелексический порядок столбцов классов). Каждый CLI выполняет один evaluate и один полный refit на новых синтетических 90 train / 12 test; вложенные CSV, переименованные target/ID, лексические 001/+1/NA, missing и unknown category.

Проверены схема, ID/порядок/число строк, конечность, диапазон вероятностей/суммы строк, классы, два completed claims, completed final.json; inner metrics и diagnostics независимо пересчитаны из NPZ через sklearn/numpy. Все команды, fingerprints, input/source SHA256, worker/CLI logs и результаты сохранены рядом. API env очищено, ledger читается из корневого HYPOTHESES.md, temporary files остаются в этой папке.

Ограничения: это техническая синтетическая проверка, не проверка качества на real/hidden данных; установка requirements в новом окружении и вызов внешнего API не выполнялись. ZIP не создавался. Подробности: verification.json; source_manifest.json. Ошибка (если есть): нет.
