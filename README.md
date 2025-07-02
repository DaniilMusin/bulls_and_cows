# cow‑bull‑assignment

Полное решение задачи «Подбор пар для осеменения коров».

## Быстрый старт

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# bulls.csv, cows.csv, pedigree.csv → ./data
python -m src.main --data_dir data --solver greedy           # 10‑15 s
python -m src.main --data_dir data --solver milp             # 4‑8 мин

# с учётом разброса EBV (λ = 1e-3)
python -m src.main --data_dir data --solver milp --lambda_variance 0.001


Результат → `cow_bull_assignments.csv`.

## Логика


Ограничения

1. R(bull,cow) ≤ 0.05
2. quota_bull = 10 % коров
3. каждая корова → ровно один бык

Цель – максимизировать средний EBV потомков и увеличить его разброс

Алгоритмы

- greedy – быстрая эвристика, погрешность ≈ 1 %
- milp – 0‑1 MILP (PuLP+CBC), решает двухкритериальную задачу:
  1. максимизирует средний ожидаемый EBV потомства;
  2. увеличивает разброс EBV, добавляя к цели сумму абсолютных отклонений
     от среднего с весом `lambda_variance`.

Валидация – `pytest`, метрики `mean/std EBV_child`, баланс нагрузки.


## Тесты

```bash
pytest -q         # все зелёные
```

## Future work

* параллельный расчёт R (multiprocessing/numba prange)
* сценарный анализ «исключить быка / изменить квоту»

