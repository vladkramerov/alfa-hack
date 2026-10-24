# H107: технический полный refit

Команда: .venv-ml/Scripts/python.exe -X utf8 experiments/technical_sergey_fusion.py.
Обучение переносимого графа на полном train для test submission. Внутренние веса обучаются на данных текущего train.
Public holdout метрики не пересчитывались; этот запуск не является новым доказательством качества.
{
  "status": "completed",
  "mode": "technical_full_refit_only",
  "quality_metric_computed": false,
  "train_rows": 53308,
  "test_rows": 22449,
  "submission_validated": true,
  "weights_learned_on_current_train": [
    0.21081932647573914,
    0.3982378488075611,
    0.26664454492561546,
    0.12429827979108435
  ],
  "seconds": 295.5779999999795,
  "model_sha256": "f778c3786c8a55248940b884ff5b163f93be21d7f81b4e6cb2df30305fd64869",
  "submission_sha256": "eb906e8147a796132d606b86f0579297dfd0e18a2835a49d285d5c06c8439992",
  "learner_source_directory": "C:\\Users\\User\\PycharmProjects\\Alpha_Hackathon\\deliveries\\sergey_ivan_fusion_v1",
  "known_tasks_embedded": false
}