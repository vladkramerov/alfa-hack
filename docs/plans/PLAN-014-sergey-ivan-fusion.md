# PLAN014

Шаги1–5 завершены; [отчёт](../../results/2026-10-24/series-sergey-ivan-fusion-v1/report.md). Независимый read-only reviewer не выявил critical/important дефектов; ограничения cached metrics сохранены. Отдельный вариант создан без перезаписи root WIP и старой поставки Сергея.

1. Прочитать журналы двух веток, создать SPEC014 и H107 до parameter fit (REQ001–004).
2. Написать synthetic tests и модуль fusion_weights.py; simplex solver без публичных констант состояния.
3. experiments/sergey_fusion.py: immutable source/input SHA freeze, cached inner alignment, один fit весов; веса сохранить до открытия outer. Отдельно записать оба исхода и ledger (REQ002–004).
4. При научном успехе перенести стратегию в отдельную поставку на основе Sergey с task-local обучением весов и проверить без повторов старого поиска (REQ005). Иначе оставить исследовательские изменения и явно закрыть отрицательный опыт.
5. Независимо пересчитать метрики/сверить артефакты; обновить отчёты и Codex.md. Не обещать недоказанных hidden gains.
