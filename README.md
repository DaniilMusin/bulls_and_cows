# 🐮 bulls_and_cows — подбор пар «бык × корова»

Решение тестовой задачи R&D‑команды:  
максимизируется средняя селекционная ценность **EBV** потомства при строгих ограничениях:

1. родство «бык‑корова» ≤ 5 %  
2. на каждого быка приходится ≤ 10 % всех коров  
3. каждая корова закреплена ровно за одним быком

Два решателя:

| Режим | Скорость | Точность | Внешний решатель |
|-------|----------|----------|------------------|
| `greedy` | 10‑15 с | ≈ ‑1 % к оптимуму | **нет** |
| `milp`   | 4‑8 мин | глобальный оптимум | нужен **CBC** |

---

## 1 . Быстрый старт

| Шаг | Linux / macOS (bash/zsh) | Windows PowerShell | Windows cmd |
|-----|--------------------------|--------------------|-------------|
| 1. клонировать | `git clone https://github.com/DaniilMusin/bulls_and_cows.git`<br>`cd bulls_and_cows` | то же | то же |
| 2. вирт. окружение | `python -m venv .venv`<br>`source .venv/bin/activate` | `python -m venv .venv`<br>`.\.venv\Scripts\Activate.ps1` | `python -m venv .venv`<br>`.\.venv\Scripts\activate.bat` |
| 3. зависимости | `pip install -r requirements.txt`<br>*пакет `pulp[cbc]` попытается установить колесо с готовым CBC* | то же | то же |
| 4. данные | положите `bulls.csv`, `cows.csv`, `pedigree.csv` в папку **`data/`** | то же | то же |
| 5. быстрый расчёт | `python -m src.main --data_dir data --solver greedy` | то же | то же |
| 6. MILP‑оптимум | `python -m src.main --data_dir data --solver milp` | то же | то же |

После выполнения появится **`cow_bull_assignments.csv`**: 2 колонки `cow_id`, `bull_id`.

---

### 🔄 Если pip не нашёл wheel с CBC

`pip install -r requirements.txt` может вывести  
`ERROR: Could not find a version that satisfies… coin-or-cbc`.

**Исправления**:

| ОС | Как установить CBC вручную |
|----|----------------------------|
| **Ubuntu / Debian** | `sudo apt-get install coinor-cbc` |
| **macOS (Homebrew)** | `brew install cbc` |
| **Windows** | 1. Скачайте архив **cbc-win64-msvc17.zip** с релизов: <https://github.com/coin-or/Cbc/releases><br>2. Распакуйте, например, в `C:\cbc`.<br>3. Добавьте путь к `cbc.exe` в текущую сессию:<br/>`$Env:Path += ';C:\cbc'`<br>4. Проверьте: `cbc --version` |

Если CBC по‑прежнему не найден, запустите решение в режиме `greedy` — он не требует внешних решателей.

---

## 2 . Тесты и линт

```bash
# внутри .venv
pytest -q            # ..  (или ... если CBC найден и тест MILP активен)
ruff src tests       # опционально: стиль кода
