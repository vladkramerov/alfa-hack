# Сборка и отправка решения

Обновлено: 2026-10-23. Отправляйте только архив, собранный из того commit, который вы записали для результата.

## Получить ветку

Из клона репозитория используйте ветку `codex/ultimate-autoresearch-20261023`:

```sh
git clone --branch codex/ultimate-autoresearch-20261023 --single-branch <REPOSITORY_URL> alfa-autoresearch
cd alfa-autoresearch
git rev-parse HEAD
```

Передавать URL нужно от своего Git remote/хостинга. Не фиксируйте commit в инструкции: он изменяется при новых коммитах.

## Собрать ZIP

В корне проекта, с Python 3:

```sh
python3 experiments/package_solution.py
```

Команда создаёт `dist/alfa_autoresearch_submission.zip` и `dist/alfa_autoresearch_submission.manifest.json`. В ZIP ожидаются только flat-файлы `solution.py`, `automl_engine.py`, `automl_data.py`, `automl_models.py`, `automl_worker.py`, `automl_llm.py`, `automl_runtime.py`, `requirements.txt`. Пакет ограничен 100 MB; официальный `requirements.txt` должен совпадать с исходным по SHA.

Проверьте состав, CRC и SHA архива:

```sh
zipinfo -1 dist/alfa_autoresearch_submission.zip
unzip -t dist/alfa_autoresearch_submission.zip
shasum -a 256 dist/alfa_autoresearch_submission.zip
cat dist/alfa_autoresearch_submission.manifest.json
```

Сохраните рядом commit, SHA архива, submission ID/дату и опубликованный платформой score. Не записывайте заранее ожидаемый score: локальная метрика и номер в очереди его не определяют.

| Commit | SHA256 ZIP | Submission ID / время (MSK) | Итоговый score |
|---|---|---|---|
| `вставить фактический SHA commit` | `вставить SHA из manifest` | `вставить после загрузки` | `вставить после расчёта платформы` |

## Загрузка и интерпретация статуса

По предоставленному правилу платформы разрешено до **3 загрузок в сутки**, счётчик обновляется в **00:00 по Москве**. Считайте это ограничением организатора; перед загрузкой проверьте актуальный интерфейс соревнования.

Статус «в очереди» подтверждает лишь постановку на обработку, но не оценку. Дождитесь конечного результата; свяжите его с commit и SHA именно загруженного ZIP. При исчерпанном лимите не подменяйте результат локальной оценкой и не создавайте повторную загрузку в обход лимита.

Ошибки запуска/загрузки, которые обычно делают submission непригодной: отсутствующий `solution.py` в корне ZIP, лишний уровень вложенности, изменённый официальный `requirements.txt`, недостающий модуль, превышение лимита размера, импорт отсутствующей зависимости, неверный entrypoint, невалидный `submission.csv`, превышение дедлайна или памяти. Локальная проверка архива и clean-room важны перед отправкой; final clean-room для текущего пересобранного архива ещё требуется.
