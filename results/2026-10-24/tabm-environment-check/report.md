# Проверка разрешённых зависимостей

Дата: 24.10.2026, Москва. Техническая проверка окружения, не ML-гипотеза; 0 model fits.

`requirements.txt` разрешает torch==2.5.1, numpy==1.26.4, pandas==2.2.3, scikit-learn==1.5.2 и threadpoolctl==3.6.0. Отдельного пакета tabm нет; его установка не выполнялась. requirements.txt не изменялся.

Python .venv-ml: 3.12.14. NumPy1.26.4/pandas2.2.3/sklearn1.5.2 установлены; torch отсутствует. Вторая .venv также не содержит torch; проверенные стандартные локальные расположения не дали доступной установки.

По прямому уточнению пользователя установка разрешённого torch повторена:

```powershell
.venv-ml/Scripts/python.exe -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu --disable-pip-version-check --no-cache-dir --retries 0 --timeout 15
```

Результат: exit1, [install-retry.log](install-retry.log); дополнительный verbose-log: [install-verbose.log](install-verbose.log). pip сообщает `No matching distribution found`, но отдельная проверка соединения выявляет сетевой отказ до чтения индекса, поэтому отсутствие wheel или несовместимость версии этим не доказаны.

Диагностика:

```powershell
.venv-ml/Scripts/python.exe -X utf8 -c "import urllib.request; urllib.request.urlopen('https://download.pytorch.org/whl/cpu/torch/',timeout=10)"
```

Точный результат:

```text
PermissionError: [WinError 10013] Сделана попытка доступа к сокету методом, запрещенным правами доступа
urllib.error.URLError: <urlopen error [WinError 10013] Сделана попытка доступа к сокету методом, запрещенным правами доступа>
```

Полный стек: [network-error.log](network-error.log). Сетевые настройки и ограничения не изменялись. Исследовательская NumPy-реализация использует только уже установленные разрешённые зависимости; официальный пакет TabM и PyTorch-репликация не проверены.

## Обновление после установки пользователем, 24.10.2026

По просьбе «я вроде установил пайторч проверь» выполнен повторный import-check двух окружений. В `.venv` (Python3.14) теперь установлен и успешно импортируется `torch==2.14.1+cpu`, путь `.venv/Lib/site-packages/torch/__init__.py`. В `.venv-ml` (Python3.12, окружение экспериментов) `torch` по-прежнему отсутствует. Предоставленный requirements фиксирует `torch==2.5.1`, поэтому установка пользователя не считается проверкой целевого платформенного окружения. Установленные пользователем пакеты не изменялись. H108 продолжает заранее зарегистрированную NumPy-конфигурацию, без повторов прежних fit под другим backend.

## Успешная установка пользователем в целевую среду, 24.10.2026

После сообщения «я сделал» проверен `.venv-ml/Scripts/python.exe`: Python3.12.14, torch2.5.1+cpu, import успешен, операция `torch.tensor([1.,2.])+1` вернула `[2.0,3.0]`, exit0. Версия совпадает с requirements.txt; CUDA=False для установленной CPU-сборки. [Машинная запись](torch-installed-check.json). Прежние сообщения об отсутствии torch описывают прошлое состояние. Новых model fits0, H108 не повторялась.
