# Лабораторная работа №1: Семантическая сегментация (CV)

Студент: Халимов И И

Группа: М8О-408Б-22

Курс **«Киберфизические системы»**. Вся работа собрана в единый Jupyter-ноутбук **`lab1_water_segmentation.ipynb`** и покрывает все 4 пункта ТЗ.

- **Задача:** бинарная семантическая сегментация водных поверхностей на мультиисточниковых спутниковых снимках.
- **Датасет:** [Multi-source Satellite Imagery for Segmentation](https://www.kaggle.com/datasets/hammadjavaid/multi-source-satellite-imagery-for-segmentation) — объединяет снимки из нескольких источников (Sentinel-1/2, Landsat и др.) с готовыми масками воды.
- **В работе используется случайная половина датасета** (укорачивание прописано прямо в ноутбуке после загрузки, чтобы не затрагивать остальной код).
- **Обоснование (КФС):** мониторинг паводков, БПЛА-картирование водоёмов, экологический мониторинг, управление водными ресурсами; мультиисточниковые данные дополнительно повышают реалистичность задачи — в реальной КФС модели приходится работать с потоками от разных сенсоров.
- **Метрики:** IoU (главная), Dice, Pixel Accuracy, Precision, Recall.

Все обоснования, гипотезы, таблицы сравнений и выводы — внутри ноутбука.

---

## Содержимое репозитория

```
ciber_fiz/
├── README.md                          # этот файл — инструкция
├── requirements.txt                   # зависимости Python
├── lab1_water_segmentation.ipynb      # ВСЯ работа (код + эксперименты + выводы)
└── data/                              # (создаётся вручную, см. §2)
    └── <распакованный датасет>/
        ├── Images/   (или images/)
        └── Masks/    (или masks/)
```

Результаты обучения автоматически сохраняются в `results/<имя_запуска>/`.

---

## 1. Установка окружения

### 1.1 Требования

- **Python** 3.10 или 3.11;
- **GPU с CUDA** желательно (T4 / RTX 20xx–40xx и т.п.). На CPU тоже работает, но медленно;
- `git`.

### 1.2 Клонирование и окружение

```powershell
git clone https://github.com/IsmailKhalimov/cyber_fiz_lab1.git
cd ciber_fiz

# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Linux / macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

### 1.3 Установка PyTorch

Сначала установите PyTorch под вашу систему (см. [pytorch.org](https://pytorch.org/get-started/locally/)). Примеры:

```bash
# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# или CPU-only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 1.4 Остальные зависимости

```bash
pip install -r requirements.txt
```

---

## 2. Данные (Kaggle CLI или ручное скачивание)

### Вариант A — через Kaggle CLI (используется в ноутбуке)

1. Получите API-токен: [kaggle.com](https://www.kaggle.com) → Account → *Create New API Token* → `kaggle.json`.
2. Положите его в `~/.kaggle/kaggle.json` (Linux/macOS/Colab) или `%USERPROFILE%\.kaggle\kaggle.json` (Windows).
3. В ноутбуке уже есть ячейка:
   ```
   !kaggle datasets download -d hammadjavaid/multi-source-satellite-imagery-for-segmentation -p data --unzip
   ```
   Она скачает и распакует датасет в папку `data/`.

### Вариант B — ручное скачивание

1. Зайдите на [kaggle.com/datasets/hammadjavaid/multi-source-satellite-imagery-for-segmentation](https://www.kaggle.com/datasets/hammadjavaid/multi-source-satellite-imagery-for-segmentation) → **Download** → получите `archive.zip`.
2. Создайте рядом с ноутбуком папку `data/`:

3. Распакуйте архив в `data/`:

### Укорачивание датасета

Сразу после загрузки списка пар в ноутбуке делается **`all_pairs = all_pairs[:len(all_pairs)//2]`** — используется случайная половина с фиксированным сидом. Это позволяет уложить обучение восьми моделей в разумное время и при этом оставить весь дальнейший код без изменений.

---

## 3. Запуск ноутбука

### 3.1 В Jupyter / JupyterLab

```bash
pip install jupyter
jupyter lab
```

Откройте `lab1_water_segmentation.ipynb` и выполните ячейки сверху вниз (`Run All` / `Ctrl+Shift+End` → `Shift+Enter`).

Порядок выполнения:

| Раздел ноутбука | Пункт ТЗ | Что делается |
|---|---|---|
| 0. Установка и данные | — | Импорты, конфигурация, проверка пути до данных |
| 1. EDA и разбиение | 1 | Разведочный анализ, train/val/test (60/20/20) |
| 2. Инфраструктура | — | Dataset, аугментации, метрики, лоссы, циклы обучения |
| 3. Пункт 2 — Бейзлайн | 2 | U-Net/ResNet34, FPN/ResNet34, SegFormer/MiT-B0 |
| 4. Пункт 3 — Улучшенный бейзлайн | 3 | Гипотезы H1–H5, улучшенные версии всех моделей, сравнение |
| 5. Пункт 4 — Собственная U-Net | 4 | Своя имплементация + её обучение с base/improved пресетами |
| 6. Итоговая сводка | — | Общая таблица, визуализация предсказаний лучшей модели |
| 7. Выводы | — | Выводы по всем пунктам и КФС-рекомендации |

### 3.2 В VS Code

- Откройте папку проекта в VS Code;
- установите расширение **Jupyter**;
- выберите ядро вашего venv (`.venv`);
- жмите **Run All**.

### 3.3 В Google Colab (если нет локального GPU)

1. Залейте ноутбук и архив с данными в Colab.
2. Разархивируйте:
   ```python
   !mkdir -p data && unzip -q /content/archive.zip -d data
   ```
3. Убедитесь, что путь `data/images` существует.
4. Выполните ячейки по порядку.

### 3.4 Сколько времени идёт обучение?

Всего в ноутбуке 8 прогонов обучения (3 baseline SMP + 3 improved SMP + 2 custom).
На **T4/RTX 3060** один прогон занимает 3–8 минут → всего ~30–60 минут.
На CPU — в разы дольше (несколько часов).

Чтобы сделать быстрый прогон для проверки, уменьшите в ячейке «Конфигурация»:

```python
EPOCHS_BASELINE = 3
EPOCHS_IMPROVED = 3
```

---

## 4. Как читать результаты

- Все метрики каждой модели сохраняются в `results/<run_name>/metrics.json` и чекпоинты в `results/<run_name>/best.pt`.
- В разделе **«6. Итоговая сводка»** ноутбука собирается общая таблица `df_all` и строится визуализация предсказаний лучшей модели.
- Сравнения по пунктам 2→3 и 2→4→3 — в таблицах `df_cmp_23`, `cmp_custom_vs_smp`, `cmp_improved`.

---

## 5. Воспроизводимость

- Все сиды фиксируются (`SEED=42`), `cudnn.deterministic=True`;
- Версии библиотек закреплены в `requirements.txt`;
- Ноутбук полностью самодостаточен — один файл, одна команда «Run All».

---

## 6. Лицензии

- Код: MIT.
- Датасет: см. условия на [странице Kaggle](https://www.kaggle.com/datasets/hammadjavaid/multi-source-satellite-imagery-for-segmentation).
- `segmentation_models.pytorch`: MIT.
