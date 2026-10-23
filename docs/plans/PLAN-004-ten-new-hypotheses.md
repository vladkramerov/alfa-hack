# PLAN-004: одиннадцать оценок десяти новых гипотез

**Цель:** выполнить [SPEC-004](../specs/SPEC-004-ten-new-hypotheses.md) и вернуть измеренный вывод без повторов старых проверок.
**Архитектура:** отдельные experiments/novel_features.py, novel_models.py, novel_worker.py, run_novel_hypotheses.py; основной платформенный агент пока не меняется. Постоянный registry + централизованный HYPOTHESES.md; каждый run получает исходный snapshot и отдельные артефакты.
**Стек:** существующий Python3.12/numpy1.26.4/pandas2.2.3/sklearn1.5.2/lightgbm4.6.0 + разрешённый xgboost2.1.4. Обучение максимум4threads последовательно.
**Метод:** независимый субагент реализует и тестирует learned feature transformers, второй проверяет протокол; root интегрирует модели, runner и результаты.

- [x] Прочитать единый журнал и предложения; аудит новизны, исходных hashes/splits, доменов y и категорий Normal.
- [x] Создать отдельные карточки HYP-011…020 с точными настройками и сравнением, обновить статус в журнале до fit.
- [x] TDD: tests/test_novel_models.py — обратное преобразование target, groupweights, композиция иерархии и фиксированный план11runs.
- [x] Субагент: experiments/novel_features.py и tests/test_novel_features.py — grouped OOF encoding, cross и частоты; проверка утечек и unknowns.
- [x] Root: experiments/novel_models.py и novel_worker.py; явное выравнивание классов, train-only preprocessing, prediction/probability artifacts.
- [x] Root: runner с read-before-every-run, write-started-before-fit, namespace-independent fingerprint и запретом повторов; снимок кода до запуска.
- [x] Запустить unit tests; проверить11 новых fixed manifests, зависимости и protocol review; запустить только ещё не выполнявшиеся runs.
- [x] Независимо пересчитать метрики без fit; проверить исходы всех10 гипотез, построить общий отчёт и обновить журнал сразу после каждого run.
- [x] Проверить экспериментальный код и отсутствие данных/моделей/секретов в commit; push ivan_branch разрешён последним поручением пользователя.

Проверки модели на синтетике не считаются проверкой публичной гипотезы. Сбой или timeout сохраняется; перед любым новым fit той же гипотезы требуется отдельное обоснование и отсутствие завершённой проверки, а не смена seed/имени.
