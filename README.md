# 🐮 bulls_and_cows — подбор пар «бык × корова»

Решение тестовой задачи R&D‑команды:  
максимизируется средняя селекционная ценность **EBV** потомства при строгих ограничениях:

1. Родство «бык – корова» ≤ 5 %  
2. На каждого быка приходится ≤ 10 % всех коров  
3. Каждая корова закреплена ровно за одним быком

Два решателя:

| Режим   | Скорость | Точность            | Внешний решатель |
|---------|----------|---------------------|------------------|
| `greedy`| 10 – 15 с| ≈ ‑1 % к оптимуму   | **нет**          |
| `milp`  | 4 – 8 мин| глобальный оптимум  | нужен **CBC**    |

---

## 1 . Быстрый старт

| Шаг | Linux / macOS (bash/zsh) | Windows PowerShell | Windows cmd |
|-----|--------------------------|--------------------|-------------|
| 1. клонировать | `git clone https://github.com/DaniilMusin/bulls_and_cows.git`<br>`cd bulls_and_cows` | то же | то же |
| 2. виртуальное окружение | `python -m venv .venv`<br>`source .venv/bin/activate` | `python -m venv .venv`<br>`.\.venv\Scripts\Activate.ps1` | `python -m venv .venv`<br>`.\.venv\Scripts\activate.bat` |
| 3. зависимости | `pip install -r requirements.txt`<br>*пакет `pulp[cbc]` попытается установить колесо с CBC* | то же | то же |
| 4. данные | положите `bulls.csv`, `cows.csv`, `pedigree.csv` в каталог **`data/`** | то же | то же |
| 5. быстрый расчёт | `python -m src.main --data_dir data --solver greedy` | то же | то же |
| 6. MILP‑оптимум | `python -m src.main --data_dir data --solver milp` | то же | то же |

После выполнения появится **`cow_bull_assignments.csv`** (2 колонки `cow_id`, `bull_id`).

> #### Что делать, если pip не нашёл CBC
> *Ubuntu/Debian*: `sudo apt-get install coinor-cbc`  
> *macOS (brew)*: `brew install cbc`  
> *Windows*: скачать архив **cbc-win64-msvc17.zip** с релизов COIN‑OR, распаковать, добавить путь к `cbc.exe` в `PATH`, проверить `cbc --version`.

Если CBC по‑прежнему отсутствует, используйте режим `--solver greedy` — он не требует внешних решателей.

---

## 2 . Тесты и линт

```bash
# внутри .venv
pytest -q            # ..  (или ... если CBC найден и MILP‑тест активен)
ruff src tests       # опционально: проверка стиля
```

---

## 3 . Описание логики, ограничений и метрик

### 3.1 Стратегия

1. **Очистка данных** – строки с пропусками `ebv` исключаются.
2. **Расчёт родства** – коэффициент `R` по Wright (1922); пары с `R > 0.05` отбрасываются.
3. **Два решателя**

   * **greedy** – коровы сортируются по `EBV` ↓, выбирают первого допустимого быка; квота отслеживается счётчиком.
   * **milp** – 0‑1 MILP: переменные `x[cow,bull]`, цель `max Σ ebv_child·x`, ограничения см. §3.2; решается PuLP + CBC.
4. **Экспорт** – выбранные пары сохраняются в CSV.

### 3.2 Реализованные ограничения

| № | Формулировка                      | Где контролируется                                 |
| - | --------------------------------- | -------------------------------------------------- |
| 1 | `R ≤ 0.05`                        | фильтр перед решателем                             |
| 2 | квота быка `≤ 10 % коров`         | счётчик / ограничение `Σ_c x[c,b] ≤ quota`         |
| 3 | каждая корова ровно у одного быка | greedy: единственный выбор; MILP: `Σ_b x[c,b] = 1` |

### 3.3 Допущения / упрощения

* Неизвестный предок → неродственный.
* Глубина родословной ограничена 5 поколениями (вклад дальних предков < 0.02).
* Пропуски `ebv` → животное исключается.
* Дисперсия EBV оптимизируется только при флаге `--lambda_variance`; по умолчанию – только mean.

### 3.4 Метрики, которыми выбирался лучший подход

| Метрика                | Назначение             | Результат                   |
| ---------------------- | ---------------------- | --------------------------- |
| Mean EBV\_child        | главная целевая        | MILP выше на ≈ 1 %          |
| Std EBV\_child         | разнообразие потомства | сопоставимо                 |
| Макс. нагрузка на быка | равномерность          | = квота (10 %)              |
| Runtime                | практичность           | Greedy ≈ 15 с; MILP ≈ 5 мин |

**Вывод**: заказчику передаём `milp_solver_cow_bull_assignments.csv` (глобальный максимум), а `greedy_solver_cow_bull_assignments.csv` – как быстрый fallback.

---

## 4 . FAQ

| Вопрос                            | Ответ                                                                                          |
| --------------------------------- | ---------------------------------------------------------------------------------------------- |
| **Почему CBC?**                   | PuLP — только интерфейс; CBC — open‑source MILP‑движок, который PuLP использует по умолчанию.  |
| **Можно без CBC?**                | Да: `--solver greedy` (просадка \~1 % по mean EBV).                                            |
| **pip не нашёл CBC**              | Установите системный пакет (`apt`, `brew`) или `cbc-win64`, либо оставайтесь на greedy.        |
| **Есть “pip‑only” альтернатива?** | `pip install highspy` + заменить вызов на `pulp.HIGHS_CMD` — работает без exe и часто быстрее. |
| **Как изменить квоту 10 %?**      | `python -m src.main --bull_quota_frac 0.15 …`                                                  |
| **Как учесть дисперсию EBV?**     | флаг `--lambda_variance 0.01` (только для MILP).                                               |

---

## 5 . Логика модулей

* **`src/kinship.py`** – LRU‑кэш + numba: быстрый расчёт `R`, формирование допустимых пар.
* **`src/model.py`** – реализации greedy и MILP; переключение через `--solver`.
* **`tests/`** – юнит‑тест точности `R`, e2e‑тест (квота, полнота назначений).
