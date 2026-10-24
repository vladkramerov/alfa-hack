# Следующие действия

Текущая strongest-team серия завершена; results/strongest-team/README.md и единый EXPERIMENT_LOG.md содержат все исходы. Действующая поставка улучшила regression policy на1.67%, classification сохранила; лучшие team research не превзойдены.

1. Один контролируемый Boosters-submit только по отдельному поручению: ZIPf3a24044… с точной связью SHA→runID→per-tasklogs. Версия13.7676 неизвестна; обещать top10 по локальным метрикам нельзя.
2. Ограниченное парное whole-policy сравнение onlineQwen vs frozenoffline: API/model теперь работают, но полное влияниеLLM на publicMLquality не измерено. Не повторять законченнуюofflineсерию.
3. Свежая проверка лучших teamfusion/TabM H109/110 с task-local обучением смеси/genericroles и одинаковыми split/preprocessing/OS; cachedweights и просмотренныйouter не считать независимым подтверждением. Сначала получить опубликованные итоги, не дублировать pendingнейронныеfits.

API https://changellenge.up.railway.app/v1, qwen/qwen3.8-27b изauthenticated/models. Ключ только localprocessenv/Boostersenv; .env.localнеGit/неZIP. Linux/официальныйcontainer ещё не проверен.
