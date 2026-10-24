"""Понимание задачи: id, таргет, тип задачи, метрика, формат сабмита.

Сначала детерминированные правила по файлам и тексту (они надёжнее), LLM только уточняет то, что
правила не смогли определить однозначно: метрику, положительный класс, колонку группы.

Порядок решений:
1. Роли колонок из data_dictionary.csv (target, id, group) и тип каждой колонки (category, text...).
2. Таргет: колонка train, которой нет в test; несколько бинарных таргетов в шаблоне сабмита — multilabel.
3. Метрика из текста (строка «основная метрика» -> строки со словом «метрика» -> весь текст).
4. Тип задачи — от метрики: метрика регрессии (RMSE, Poisson, WAPE...) делает задачу регрессией даже при
   целочисленном таргете с малым числом значений; NDCG/MAP — ранжирование; F1, AUC — классификация.
   Без метрики — по данным и тексту.
5. Бинарная: положительный класс — из текста («вероятность дефолта (`loan_status = bad`)»,
   `y_true == "bad"`, `pos_label=1`); иначе 1/yes/true; иначе более редкий класс.
6. Порядковая (QWK): классы упорядочены, модель учится как регрессия на кодах, метки — по порогам."""
from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd

import ag_ml_metrics as M
from ag_schemas import BINARY, MULTICLASS, MULTILABEL, RANKING, REGRESSION, TaskSpec

log = logging.getLogger("agent.task")

_ID_NAMES = ("id", "row_id", "index", "idx", "uid", "sample_id", "record_id")
_POSITIVE_WORDS = ("1", "1.0", "true", "yes", "y", "t", "да", "positive", "pos", "success", "successful", "fraud",
                   "default", "bad", "churn", "event", "approved", "accepted", "spam", "attack", "abnormal", "malignant")
_COST_WORDS = r"(?:стоимост\w*|штраф\w*|убыт\w*|cost)\W+(?:\w+\W+){0,6}?(?:ошиб|ложн|false|error|mistake|отказ)|матриц\w* (?:стоимост|штраф|потерь)|cost matrix"
_RANKING_WORDS = r"ранжир|ranking|rank\b|выдач|сортир|рекоменд|recommend|внутри\s+(?:кажд|одн|групп|пользоват|запрос)"


def _dtype_kind(v: str) -> str | None:
    v = str(v).lower()
    if re.search(r"text|текст|string_long|free", v):
        return "text"
    if re.search(r"date|time|дата|время", v):
        return "datetime"
    if re.search(r"categ|категор|cat\b|nominal|string|str\b|object|id\b", v):
        return "category"
    if re.search(r"num|int|float|binary|bool|real|count|числ", v):
        return "numeric"
    return None


def _dictionary(dd: pd.DataFrame | None) -> dict:
    """Роли и типы колонок из data_dictionary.csv."""
    out = {"targets": [], "ids": [], "groups": [], "desc": {}, "types": {}}
    if dd is None or dd.empty:
        return out
    cols = {c.lower(): c for c in dd.columns}
    fcol = cols.get("feature") or cols.get("column") or cols.get("name") or dd.columns[0]
    rcol = cols.get("role")
    tcol = cols.get("dtype") or cols.get("type")
    dcol = next((cols[c] for c in cols if c.startswith("description")), None)
    for _, row in dd.iterrows():
        name = str(row[fcol])
        role = str(row[rcol]).lower() if rcol else ""
        if re.search(r"target|таргет|цел", role):
            out["targets"].append(name)
        elif re.search(r"\bid\b|identifier|идентиф|ключ", role):
            out["ids"].append(name)
        elif re.search(r"group|групп|query|запрос|session|сесси", role):
            out["groups"].append(name)
        if tcol and pd.notna(row[tcol]):
            k = _dtype_kind(row[tcol])
            if k:
                out["types"][name] = k
        if dcol and pd.notna(row[dcol]):
            unit = row[cols["unit"]] if "unit" in cols and pd.notna(row[cols["unit"]]) else ""
            out["desc"][name] = f"{row[dcol]}" + (f" [{unit}]" if unit else "")
    return out


def _is_binary_col(s: pd.Series) -> bool:
    v = s.dropna().unique()
    return len(v) <= 2 and set(pd.Series(v).astype(str)) <= {"0", "1", "0.0", "1.0", "True", "False", "true", "false"}


def _match_class(value: str, classes: list):
    value = value.strip().strip("`'\"«»*.,;:)")
    for c in classes:
        if str(c) == value or str(c).lower() == value.lower():
            return c
        try:
            if float(c) == float(value):
                return c
        except (TypeError, ValueError):
            pass
    return None


def positive_class(text: str, target: str, classes: list, desc: str = "") -> tuple[object, str]:
    """Положительный класс бинарной задачи и откуда он взят."""
    t = re.escape(target)
    pats = [
        rf"`?\b{t}\b`?\s*={{1,2}}\s*[`'\"«]?([^`'\"\s,)»]+)",
        r"y_true\s*==\s*[`'\"]?([^`'\"\s,)]+)",
        r"pos_label\s*=\s*[`'\"]?([^`'\"\s,)]+)",
        r"(?:вероятност\w*|probabilit\w*)[^.\n]{0,100}?(?:класс\w*|class|метк\w*|label|значени\w*)\s*[`'\"«]?([^`'\"\s,).»]+)",
        r"(?:для|of|for)\s+(?:положительного\s+)?(?:класс\w*|class)\s*[`'\"«]?([^`'\"\s,).»]+)",
        r"положительн\w*\s+класс\w*\s*(?:—|-|:|это)?\s*[`'\"«]?([^`'\"\s,).»]+)",
        r"positive\s+class\s*(?:is|=|:)?\s*[`'\"]?([^`'\"\s,).]+)",
    ]
    hits = []
    for p in pats:
        for m in re.finditer(p, text or "", re.I):
            c = _match_class(m.group(1), classes)
            if c is not None:
                hits.append((m.start(), c))
    if hits:
        return sorted(hits, key=lambda h: h[0])[0][1], "из условия"
    low = {str(c).lower(): c for c in classes}
    for w in _POSITIVE_WORDS:
        if w in low and not (w in ("1", "1.0") and len(low) == 2 and not set(low) <= {"0", "1", "0.0", "1.0"}):
            return low[w], f"по названию класса ({low[w]})"
    m = re.search(r"(?:^|\s)[`'\"]?([^\s`'\"]+)[`'\"]?\s*[—-]\s*(?:да|событие|положит)", desc or "", re.I)
    if m and _match_class(m.group(1), classes) is not None:
        return _match_class(m.group(1), classes), "из описания таргета"
    return None, ""


def _group_from_text(text: str, cols: list[str]) -> str | None:
    """Колонка группы ранжирования из текста: «внутри пользователя (`userId`)», «по запросам `qid`»."""
    best = None
    for sent in re.split(r"(?<=[.!?\n])\s+", text or ""):
        if not re.search(_RANKING_WORDS + r"|групп|group|пользоват|user|запрос|query", sent, re.I):
            continue
        for m in re.finditer(r"`([^`]+)`", sent):
            c = m.group(1).strip()
            if c in cols:
                score = 2 if re.search(r"внутри|within|per\b|для\s+кажд|по\s+групп|групп", sent, re.I) else 1
                if best is None or score > best[0]:
                    best = (score, c)
    return best[1] if best else None


def read_task(train: pd.DataFrame, test: pd.DataFrame, dd: pd.DataFrame | None,
              sample: pd.DataFrame | None, description: str, llm=None, metric_override: str | None = None) -> TaskSpec:
    notes: list[str] = []
    D = _dictionary(dd)
    feat_desc = D["desc"]
    text = description or ""

    # --- id ---
    common = [c for c in test.columns if c in train.columns]
    id_col = None
    for c in D["ids"] + (list(sample.columns[:1]) if sample is not None else []) + list(_ID_NAMES):
        if c in test.columns and (sample is None or c in sample.columns or c in D["ids"]):
            id_col = c
            break
    if id_col is None:
        for c in common:
            if c.lower() in _ID_NAMES or (test[c].is_unique and train[c].is_unique and pd.api.types.is_integer_dtype(test[c])
                                          and c.lower().endswith("id")):
                id_col = c
                break

    # --- таргет(ы): колонка train, которой нет в test, — самый надёжный сигнал ---
    only_train = [c for c in train.columns if c not in test.columns]
    sub_cols = [c for c in (sample.columns if sample is not None else []) if c != id_col]
    dd_t = [c for c in D["targets"] if c in train.columns]
    in_sub = [c for c in sub_cols if c in only_train]
    targets: list[str] = []
    if len(in_sub) >= 2 and all(_is_binary_col(train[c]) for c in in_sub):
        targets = in_sub                                   # multilabel: по колонке на метку в шаблоне
    elif len(dd_t) >= 2 and all(_is_binary_col(train[c]) for c in dd_t) and \
            (not sub_cols or set(sub_cols) >= set(dd_t)):
        targets = [c for c in (sub_cols or dd_t) if c in dd_t]
    target = None
    for cand in (dd_t, in_sub, only_train):
        if cand:
            target = targets[0] if targets else cand[0]
            break
    if target is None:
        raise ValueError("не удалось определить таргет: в train нет колонки, отсутствующей в test")
    extra = [c for c in only_train if c not in (targets or [target])]
    if extra:
        notes.append(f"в train есть лишние колонки, которых нет в test: {extra}; они не используются")

    # --- метрика из текста (до типа задачи: тип часто следует из метрики) ---
    y = train[target]
    n_unique = y.nunique(dropna=True)
    numeric_y = pd.api.types.is_numeric_dtype(y) and not pd.api.types.is_bool_dtype(y)
    # метрика пользователя (METRIC) тип задачи не меняет: она применяется ниже, если подходит к типу по данным
    raw_metric = M.parse_metric(text, None)
    group_col = None

    # --- тип задачи ---
    proba_output = False
    submission_columns = list(sample.columns) if sample is not None else [c for c in (id_col, target) if c]
    raw = M.METRICS.get(raw_metric) if raw_metric else None
    if targets:
        task_type = MULTILABEL
        notes.append(f"multilabel: {len(targets)} бинарных таргетов {targets}")
    elif raw is not None and RANKING in raw.tasks:
        task_type = RANKING
    elif raw is not None and raw.tasks == (REGRESSION,) and numeric_y:
        task_type = REGRESSION
        notes.append(f"тип задачи по метрике {raw_metric}: регрессия")
    elif raw is not None and REGRESSION not in raw.tasks and RANKING not in raw.tasks:
        task_type = BINARY if n_unique == 2 else MULTICLASS
        if numeric_y and n_unique > 2:
            notes.append(f"тип задачи по метрике {raw_metric}: классификация ({n_unique} классов)")
    elif not numeric_y:
        task_type = BINARY if n_unique == 2 else MULTICLASS
    elif n_unique == 2:
        task_type = BINARY
    elif n_unique <= 30 and np.allclose(y.dropna() % 1, 0) and n_unique < len(y) * 0.01:
        # целые с малым числом значений: классификация, если текст не говорит о регрессии или счёте
        task_type = REGRESSION if re.search(r"регресс|regression|rmse|mae|числ\w+\s+событ|количеств|count|сколько",
                                            text, re.I) else MULTICLASS
        notes.append(f"таргет целочисленный с {n_unique} значениями, выбран тип {task_type}")
    else:
        task_type = REGRESSION
    if raw is None and task_type != MULTILABEL:
        if re.search(r"регресс|regression", text, re.I) and numeric_y and n_unique > 2:
            task_type = REGRESSION
        elif re.search(r"многокласс|multi-?class", text, re.I) and n_unique > 2:
            task_type = MULTICLASS
        elif re.search(r"бинарн|binary", text, re.I) and n_unique == 2:
            task_type = BINARY
    if task_type in (REGRESSION, MULTICLASS) and raw is None and numeric_y and re.search(_RANKING_WORDS, text, re.I):
        # «чем больше число, тем выше в выдаче» без названной метрики — тоже ранжирование, если есть группа
        g = (D["groups"] and D["groups"][0]) or _group_from_text(text, common)
        if g:
            task_type = RANKING
            notes.append("в условии ранжирование внутри групп, метрика не названа: NDCG@10")

    # --- группа для ранжирования ---
    if task_type == RANKING:
        group_col = next((g for g in D["groups"] if g in common), None) or _group_from_text(text, common)
        if group_col is None and llm is not None and llm.available():
            group_col = _llm_pick(llm, text, "колонку группы, внутри которой сортируются объекты (запрос, пользователь)",
                                  [c for c in common if c != id_col])
        if group_col:
            notes.append(f"ранжирование внутри групп `{group_col}`")
        else:
            notes.append("ранжирование: колонка группы не найдена, весь тест — одна группа")

    # --- классы, положительный класс, вероятности ---
    classes: list = []
    positive = None
    if task_type in (BINARY, MULTICLASS):
        classes = sorted(y.dropna().unique().tolist(), key=lambda v: (str(type(v)), v))
        if task_type == BINARY:
            positive, why = positive_class(text, target, classes, feat_desc.get(target, ""))
            if positive is None and llm is not None and llm.available():
                positive = _llm_pick(llm, text, f"положительный класс таргета `{target}` (вероятность которого "
                                                 f"нужно предсказать или F1 которого считается)", classes)
                why = "выбрала LLM" if positive is not None else ""
            if positive is None:
                vc = y.value_counts()
                positive, why = vc.index[-1], "более редкий класс"
            classes = [c for c in classes if c != positive] + [positive]
            notes.append(f"положительный класс: {positive} ({why})")
        cls_cols = [c for c in sub_cols if c != target]
        if len(cls_cols) >= 2 and len(cls_cols) == len(classes):
            proba_output = True
        elif sample is not None and target in sample.columns and pd.api.types.is_float_dtype(sample[target]):
            vals = sample[target].dropna()
            if len(vals) and vals.between(0, 1).all() and not np.allclose(vals % 1, 0):
                proba_output = True
    elif task_type == MULTILABEL:
        classes = [0, 1]

    # --- метрика под тип задачи ---
    # Порядок: явный выбор пользователя (METRIC / своя метрика) -> метрика, названная в тексте ->
    # LLM выбирает по цели из текста и по данным -> правило по данным.
    metric = None
    if metric_override:
        m = M.METRICS.get(M.ensure(metric_override) or "")
        if m and (task_type in m.tasks or m.tasks == (REGRESSION, BINARY, MULTICLASS)):
            metric = metric_override
            notes.append(f"метрика задана пользователем: {metric}")
        else:
            notes.append(f"метрика {metric_override} не подходит для {task_type}, выбираю сам")
    if metric is None:
        metric = M.adapt(raw_metric, task_type) if raw_metric else None
        if metric is None:
            metric = M.parse_metric(text, task_type)
        if metric:
            notes.append(f"метрика названа в условии: {metric}")
        if metric and M.METRICS[metric].kind == "label" and task_type in (BINARY, MULTICLASS):
            cost = parse_cost_matrix(text, classes)
            if cost is not None and re.search(r"стоимост|cost|штраф|убыт|потер", text, re.I):
                metric = M.register_cost(cost)
                notes.append(f"метрика — стоимость ошибок по матрице из условия: {cost}")
    if metric is None and task_type in (BINARY, MULTICLASS):
        cost = parse_cost_matrix(text, classes)
        if cost is not None:
            metric = M.register_cost(cost)
            notes.append(f"метрика — стоимость ошибок по матрице из условия: {cost}")
    if task_type in (BINARY, MULTICLASS) and (metric is None or M.METRICS[metric].kind == "label") \
            and "cost_matrix" not in (M.METRICS.get(metric).args if metric else {}) \
            and re.search(_COST_WORDS, text, re.I) and llm is not None and llm.available() \
            and (metric is None or _cost_in_metric_lines(text)):
        # стоимость ошибок описана словами, а не таблицей: матрицу извлекает LLM, ответ проверяется
        cost = _llm_cost_matrix(llm, text, classes)
        if cost is not None:
            metric = M.register_cost(cost)
            notes.append(f"метрика — стоимость ошибок, матрицу из текста извлекла LLM: {cost}")
    if metric is None and llm is not None and llm.available():
        metric, why = _llm_metric(llm, text, task_type, _data_summary(y, task_type))
        if metric:
            notes.append(f"метрику выбрала LLM: {metric} ({why})")
    if metric is None:
        metric, why = _default_metric(y, task_type, proba_output)
        notes.append(f"метрика выбрана по данным: {metric} ({why})")
    mk = M.METRICS[metric].kind
    if task_type == BINARY and mk == "proba" and not proba_output:
        proba_output = True
        notes.append("метрика по вероятностям: в сабмит пишем вероятность положительного класса")
    if task_type == MULTILABEL:
        proba_output = mk == "proba"
    if proba_output and mk == "label" and task_type != MULTILABEL:
        notes.append("сабмит требует вероятности, но метрика по меткам; пишем вероятности")
    ordinal = task_type == MULTICLASS and metric == "qwk" and len(classes) >= 3
    if ordinal:
        notes.append("порядковые классы (QWK): модель учится как регрессия на кодах классов, метки — по порогам")

    spec = TaskSpec(task_type=task_type, target=target, id_col=id_col, metric=metric,
                    submission_columns=submission_columns, proba_output=proba_output, classes=classes,
                    description=description, feature_descriptions=feat_desc, notes=notes, targets=targets,
                    group_col=group_col, ordinal=ordinal, positive_class=positive, column_types=D["types"])
    log.info("task: type=%s target=%s id=%s metric=%s proba=%s", task_type, target if not targets else targets,
             id_col, metric, proba_output)
    for n in notes:
        log.info("task note: %s", n)
    return spec


def parse_cost_matrix(text: str, classes: list) -> list[list[float]] | None:
    """Матрица стоимости из markdown-таблицы условия: строки — истинный класс, колонки — ответ (или
    наоборот, если так подписано). Ячейки — числа. Возвращает matrix[истинный][ответ] по кодам classes."""
    if not text or len(classes) < 2:
        return None
    rows = [ln for ln in text.splitlines() if ln.strip().startswith("|")]
    blocks, cur = [], []
    for ln in text.splitlines():
        if ln.strip().startswith("|"):
            cur.append(ln)
        elif cur:
            blocks.append(cur)
            cur = []
    if cur:
        blocks.append(cur)
    del rows
    for blk in blocks:
        cells = [[c.strip().strip("`*") for c in ln.strip().strip("|").split("|")] for ln in blk
                 if not re.fullmatch(r"[\s|:\-]+", ln)]
        if len(cells) < 3:
            continue
        head, body = cells[0], cells[1:]
        col_cls = [_match_class(_last_token(h), classes) for h in head[1:]]
        row_cls = [_match_class(_last_token(r[0]), classes) for r in body]
        if sum(c is not None for c in col_cls) < len(classes) or sum(c is not None for c in row_cls) < len(classes):
            continue
        idx = {c: i for i, c in enumerate(classes)}
        C = np.full((len(classes), len(classes)), np.nan)
        for r, rc in zip(body, row_cls):
            if rc is None:
                continue
            for j, cc in enumerate(col_cls):
                if cc is None or j + 1 >= len(r):
                    continue
                v = re.sub(r"[^\d.\-]", "", r[j + 1].replace(",", "."))
                try:
                    C[idx[rc], idx[cc]] = float(v)
                except ValueError:
                    pass
        if np.isnan(C).any():
            continue
        corner = (head[0] + " " + " ".join(r[0] for r in body[:1])).lower()
        # «ответ \\ истина» или «предсказано» в углу — строки таблицы это ответ, транспонируем
        if re.search(r"^(?:ответ|предсказ|predict)", head[0].strip().lower()) or "предсказ\\" in corner:
            C = C.T
        return C.tolist()
    return None


def _last_token(s: str) -> str:
    s = re.sub(r"[`*«»\"']", " ", s)
    toks = [t for t in re.split(r"[\s=:()]+", s) if t]
    return toks[-1] if toks else s


def _data_summary(y: pd.Series, task_type: str) -> dict:
    if task_type in (REGRESSION, RANKING):
        yv = y.astype(float)
        return {"n": int(len(y)), "min": round(float(yv.min()), 4), "max": round(float(yv.max()), 4),
                "skew": round(float(yv.skew()), 2), "share_zero": round(float((yv == 0).mean()), 3),
                "integer": bool(np.allclose(yv.dropna() % 1, 0))}
    vc = y.value_counts(normalize=True)
    return {"n": int(len(y)), "class_shares": {str(k): round(float(v), 3) for k, v in vc.head(20).items()}}


def _default_metric(y: pd.Series, task_type: str, proba_output: bool) -> tuple[str, str]:
    """Правило без LLM: метрика, которая честно отражает качество на таких данных."""
    if task_type == RANKING:
        return M.ensure("ndcg@10"), "ранжирование внутри групп"
    if task_type == MULTILABEL:
        return "ml_auc", "несколько бинарных меток: средний ROC-AUC"
    if task_type == REGRESSION:
        yv = y.astype(float)
        if (yv >= 0).all() and np.allclose(yv % 1, 0) and yv.mean() < 10 and (yv == 0).mean() > 0.3:
            return "poisson", "неотрицательный целочисленный таргет с большим числом нулей: счётная величина"
        if (yv >= 0).all() and yv.skew() > 2 and (yv > 0).mean() > 0.95:
            return "rmsle", "таргет неотрицательный и сильно скошен: важна относительная ошибка"
        return "rmse", "стандартная метрика регрессии"
    vc = y.value_counts(normalize=True)
    if task_type == BINARY:
        if proba_output or vc.min() < 0.2:
            return "roc_auc", "классы несбалансированы или нужны вероятности: оцениваем ранжирование"
        return "f1", "классы сбалансированы, ответ метками"
    if proba_output:
        return "logloss", "нужны вероятности классов"
    if vc.max() / max(vc.min(), 1e-9) < 1.5:
        return "accuracy", "классы сбалансированы"
    return "macro_f1", "классы несбалансированы: редкие классы должны весить столько же"


def _llm_metric(llm, description: str, task_type: str, data: dict) -> tuple[str | None, str]:
    allowed = {k: M.METRIC_DOCS.get(k, "") for k, m in M.METRICS.items() if task_type in m.tasks}
    ans = llm.chat_json(
        system="Ты опытный DS. Выбираешь метрику качества для ML-задачи. Отвечай только JSON.",
        user=(f"Тип задачи: {task_type}. Данные (таргет): {data}.\n"
              f"Допустимые метрики: {allowed}.\n"
              f"Текст задачи от пользователя:\n{description[:4000] or '(пусто)'}\n\n"
              "Если в тексте метрика названа явно, верни её. Иначе выбери ту, что лучше всего отражает цель "
              "пользователя из текста (что для него ошибка и что важнее) с учётом данных (дисбаланс, скошенность).\n"
              'Верни {"metric": "<одна из допустимых>", "reason": "<коротко почему>"}'),
        max_tokens=300,
    )
    m = (ans or {}).get("metric")
    return (m, str((ans or {}).get("reason", ""))[:200]) if m in allowed else (None, "")


def _cost_in_metric_lines(text: str) -> bool:
    """Стоимость упомянута там, где описана метрика (а не просто в описании бизнеса)."""
    lines = [ln for ln in text.splitlines() if re.search(r"метрик|metric|оценива|score|качеств", ln, re.I)]
    return any(re.search(r"стоимост|штраф|убыт|потер|cost", ln, re.I) for ln in lines)


def _llm_cost_matrix(llm, description: str, classes: list) -> list[list[float]] | None:
    """Матрица стоимости ошибок, описанная в тексте словами. Строки — истинный класс, колонки — ответ."""
    opts = [str(c) for c in classes]
    ans = llm.chat_json(
        system="Ты внимательно читаешь условие ML-задачи и извлекаешь из него числа. Отвечай только JSON.",
        user=(f"Текст задачи:\n{description[:5000]}\n\nКлассы таргета (в этом порядке): {opts}.\n"
              "Если в тексте задана стоимость (штраф) ошибок классификации, верни матрицу cost[i][j] — стоимость "
              "ответа классом j, когда истинный класс i (диагональ — стоимость верного ответа, обычно 0). "
              'Если стоимость не задана, верни null. Формат: {"cost_matrix": [[...], ...] или null, "quote": "<фраза из текста>"}'),
        max_tokens=400, temperature=0.0,
    )
    C = (ans or {}).get("cost_matrix")
    try:
        C = np.asarray(C, dtype=float)
    except (TypeError, ValueError):
        return None
    k = len(classes)
    if C.shape != (k, k) or not np.isfinite(C).all() or (C < 0).any():
        return None
    off = C[~np.eye(k, dtype=bool)]
    if off.max() <= 0 or np.diag(C).max() > off.min():
        return None
    return C.tolist()


def _llm_pick(llm, description: str, what: str, options: list):
    """LLM выбирает одно значение из списка (класс, колонку) по тексту условия; ответ проверяется."""
    opts = [str(o) for o in options][:200]
    ans = llm.chat_json(
        system="Ты опытный DS и внимательно читаешь условие ML-задачи. Отвечай только JSON.",
        user=(f"Текст задачи:\n{description[:4000]}\n\nВыбери {what}. Варианты: {opts}.\n"
              'Верни {"answer": "<один из вариантов или null>", "reason": "<коротко>"}'),
        max_tokens=200,
    )
    a = (ans or {}).get("answer")
    for o in options:
        if a is not None and str(o) == str(a):
            return o
    return None
