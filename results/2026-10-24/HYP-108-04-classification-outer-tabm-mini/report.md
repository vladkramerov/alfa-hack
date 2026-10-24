# HYP-108: classification outer

Начало: 2026-10-24T14:11:57.474277+03:00

Статус: started до fit. [SPEC015](../../../docs/specs/SPEC-015-tabm-trial.md). Standalone NumPy TabM-mini; cached controls, 0 old refits.

Окончание: 2026-10-24T14:15:00.122709+03:00

Команда: `.venv-ml/Scripts/python.exe -X utf8 -m experiments.tabm_trial --kind classification --stage outer`

```json
{
  "status": "completed",
  "hypothesis": "HYP-108",
  "kind": "classification",
  "stage": "outer",
  "metric": "macro_f1",
  "score": 0.6602795926985132,
  "pure_score": 0.5099944452622676,
  "control_score": 0.6593084142037399,
  "selected_weight": 0.1,
  "arm_scores": [
    [
      0.1,
      0.6602795926985132
    ]
  ],
  "eligible_for_outer": true,
  "fit_rows": 70529,
  "valid_rows": 17141,
  "new_learner_fits": 1,
  "old_model_refits": 0,
  "seconds": 180.85899999999674,
  "fingerprint": "c2c69474d79a3fd304178c8bf26539e6051ee93e589791b76061c9a7edebabf9",
  "audit": {
    "implementation": "standalone-numpy-tabm-mini",
    "official_package": false,
    "rows": 70529,
    "encoded_features": 125,
    "members": 8,
    "epochs_completed": 60,
    "selected_epoch": 59,
    "training_losses": [
      0.7422390745708674,
      0.5561849697174636,
      0.5304520803396112,
      0.5161425830083908,
      0.5074294301402718,
      0.5000010449633397,
      0.49621609138775485,
      0.49125586643059305,
      0.48754733542665213,
      0.4852625784394076,
      0.48270168974648064,
      0.48055441157047,
      0.4784855959908269,
      0.477565781578628,
      0.47401294539196487,
      0.47333761518482154,
      0.4729605207057327,
      0.47109701322571823,
      0.4689485419849375,
      0.4681031110137838,
      0.46757794009722925,
      0.4668687362735735,
      0.4655752486013427,
      0.4644921393756954,
      0.4631932082597782,
      0.4631727395232751,
      0.4621548072118667,
      0.4612245047979294,
      0.46047419481722396,
      0.4605581723728793,
      0.4590026715178233,
      0.4592579054035352,
      0.45803679327653174,
      0.45716894951887793,
      0.4569777446109966,
      0.4562960839627843,
      0.45570077034183737,
      0.45557447067671614,
      0.45492450735206924,
      0.45537433600739763,
      0.4540188383680187,
      0.45318297214142567,
      0.4529699552772276,
      0.45304142440406053,
      0.4515819838845351,
      0.4521029990113448,
      0.4513118938692262,
      0.45177396005517234,
      0.45177719004227324,
      0.45029925076374233,
      0.44976218439571325,
      0.4502387958444987,
      0.4494143981595314,
      0.44897232486593247,
      0.44822925801792945,
      0.44939886053120387,
      0.448819891023765,
      0.44763131631884584,
      0.4472659337746919,
      0.447978763039098
    ],
    "checkpoint_criterion": "training_loss_only",
    "time_limited": false,
    "seconds": 179.18700000000536,
    "target_transform": "label_encode",
    "parameter_count": 18424,
    "fit_only_preprocessing": true
  }
}
```

Вывод: Подтверждена в проверенных условиях

Среда: Python3.12/Windows, NumPy1.26.4/sklearn1.5.2/pandas2.2.3. SHA данных/кода: request.json; split: splits.npz; predictions: validation_predictions.npz; лог: worker.log; model: model.joblib. Нет platform/hidden проверки.

Датированное техническое примечание 24.10.2026: исходный validation_predictions.npz содержит object-y и сохранён неизменным; новая validation_predictions_portable.npz доступна без pickle с теми же значениями. Проверка: ../series-tabm-mini-v1/archive-codec-fix.json. Научные метрики/модель не менялись, дополнительных fits0.
