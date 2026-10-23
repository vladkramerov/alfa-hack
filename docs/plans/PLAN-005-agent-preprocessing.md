# PLAN-005: предобработка full-v1

Спецификация: [SPEC-005](../specs/SPEC-005-agent-preprocessing.md), прямое поручение пользователя. Архитектура: отдельный корневой automl_preprocessing.py, wrapper модели и версионированные fingerprint/история; старые экспериментальные модули/snapshots не переписывать.

- [ ] TDD субагент: FullPreprocessor и TrainingOutlierFilter, train-only statistics, гарантированные caps и сохранение строк transform.
- [ ] Root: интеграция в automl_models/worker/history/solution, generic XGBoost и log-target ExtraTrees, mandatory full-v1 при разрешении кандидата и корректное отделение исторических метрик.
- [ ] Исправить выбор primary metric, расширить ограниченный research budget и train-only planning profile; тестами проверить ошибочные secondary mentions и отсутствие повторов.
- [ ] Обновить статусы HYP-011…020, создать HYP-021 до новых fit; зафиксировать два составных кандидата/данные/splits/code snapshots.
- [ ] Пройти unit/protocol review; два новых HYP-021 evaluation без прежних baseline fit, записать audit/маски/метрики/ошибки отдельно и сразу обновить память.
- [ ] Независимая сверка метрик и доли удаления, нормализации и непрерывных преобразований; честный вывод о выигрыше/проигрыше.
- [ ] Новый root-only ZIP solution_full_preprocessing.zip и неизменённый requirements; clean extraction/mock API/подпроцессы/полный refit с финальными submission.
- [ ] Отчёт, 10+1 гипотеза в журнале, commit/push ivan_branch, проверка удалённого SHA и архива.
