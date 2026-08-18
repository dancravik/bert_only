# BERT text classification — coursework project

Задача: https://www.kaggle.com/code/nayansakhiya/text-classification-using-bert/input
Данные: [`datatattle/covid-19-nlp-text-classification`](https://www.kaggle.com/datasets/datatattle/covid-19-nlp-text-classification)
("Coronavirus tweets NLP") — твиты про COVID-19, 5-классовая тональность
(`Extremely Negative / Negative / Neutral / Positive / Extremely Positive`),
41157 train / 3798 test.

## Структура репозитория

Каждый эксперимент — отдельная папка со своим `config.yaml` и `train.py`.
Общий код (загрузка данных, метрики, тренировочный цикл, Comet, error
analysis) вынесен в `common/`, чтобы не дублировать его 6 раз, но сами
эксперименты между собой не смешаны.

```
common/                      # общие утилиты, импортируются во всех exp*
├── data.py                  # чтение csv, TweetDataset, collate с динамическим паддингом
├── metrics.py                # accuracy/precision/recall/f1 (macro/weighted/per-class),
│                             #   ROC-AUC OVR/OVO, log-loss, Cohen's kappa, MCC, top-2 acc
├── engine.py                  # train_epoch / evaluate (+ per-sample loss)
├── error_analysis.py          # топ hardest/easiest примеров по loss, лог в Comet
├── comet_utils.py             # инициализация comet_ml.Experiment из config.yaml
└── utils.py                   # seed_everything, freeze_backbone

exp1_frozen_head/            # БЕЙЗЛАЙН, вариант A: backbone заморожен,
│                             #   учится только новая голова
exp2_full_finetune/          # БЕЙЗЛАЙН, вариант B: ничего не заморожено
exp3_lora/                   # LoRA (PEFT) — третий способ дообучения
exp4_alt_architectures/      # 4 модификации BERT (RoBERTa/ALBERT/DistilBERT/ELECTRA),
│                             #   один train.py, разные config.yaml
```

Каждая `exp*/config.yaml` содержит:
- `comet:` — API key, workspace, project_name (см. ниже про безопасность ключа)
- `data:` — пути к csv, размер валидации, max_length токенизации
- `model:` / `lora:` — какую модель грузить, заморозка/LoRA-параметры
- `search:` — сетка `learning_rates` × `batch_sizes` для короткого поиска
  гиперпараметров (`search_epochs` эпох на каждую комбинацию)
- `final:` — победившие `learning_rate`/`batch_size` + полное число эпох для
  финального прогона с полными метриками и error analysis
- `error_analysis.top_k` — сколько худших/лучших по loss примеров сохранять

## Как запускать на Kaggle

1. В Kaggle-ноутбуке включить Internet (Settings → Internet → On) и
   подключить датасет через "+ Add Data" → `covid-19-nlp-text-classification`.
2. Склонировать репозиторий и поставить зависимости:
   ```bash
   !git clone https://github.com/<your-username>/bert-text-classification.git
   %cd bert-text-classification
   !pip install -q -r requirements.txt
   ```
3. Вписать свой Comet API key в нужный `config.yaml` (или переопределить
   после клонирования — см. "Про API-ключ" ниже).
4. Для каждого эксперимента — сначала поиск гиперпараметров, потом финальный прогон:
   ```bash
   !python exp1_frozen_head/train.py --config exp1_frozen_head/config.yaml --mode search
   # смотрим результаты в Comet / outputs/exp1_frozen_head/search_results.csv,
   # вписываем лучшие lr/batch_size в final: внутри config.yaml
   !python exp1_frozen_head/train.py --config exp1_frozen_head/config.yaml --mode final
   ```
   Аналогично для `exp2_full_finetune`, `exp3_lora`, и для каждого файла из
   `exp4_alt_architectures/configs/*.yaml`:
   ```bash
   !python exp4_alt_architectures/train.py --config exp4_alt_architectures/configs/roberta.yaml --mode search
   !python exp4_alt_architectures/train.py --config exp4_alt_architectures/configs/roberta.yaml --mode final
   # ... albert.yaml, distilbert.yaml, electra.yaml так же
   ```
5. Бейзлайн = лучший из exp1/exp2 по `test_f1_macro` (out-of-the-box multi-class
   accuracy тут обманчива — классы сильно несбалансированы за счёт двух
   "Extremely *" категорий, поэтому именно macro-F1, а не accuracy, решает,
   какой вариант брать за бейзлайн). exp4 сравнивается с этим бейзлайном по
   тем же метрикам, залогированным в тот же Comet project.

## Про API-ключ Comet

Ключ лежит прямо в `config.yaml`, как и просили — это удобно для клонирования
в Kaggle без дополнительной возни с секретами. Единственный практический
риск: если репозиторий публичный, ключ утечёт вместе с историей коммитов.
Если это важно — самый простой вариант: держать реальные `config.yaml` в
`.gitignore` (или `os.environ` override поверх yaml), а в репозитории
коммитить только `config.example.yaml` с плейсхолдером. Сейчас в коде этого
нет, чтобы не усложнять — просто имей это в виду перед `git push`.

## Метрики (используются везде одинаково)

`common/metrics.py` считает по каждому прогону: accuracy, balanced accuracy,
precision/recall/F1 (macro, weighted, micro, и отдельно по каждому из 5
классов), ROC-AUC (OVR и OVO, macro), log-loss, top-2 accuracy, Cohen's
kappa, Matthews correlation coefficient. Плюс confusion matrix и
classification_report логируются в Comet как confusion matrix widget и text asset.

## Error analysis

После финального прогона `common/error_analysis.py` сортирует тестовую
выборку по per-sample cross-entropy loss и сохраняет/логирует в Comet топ-K
(`error_analysis.top_k`, по умолчанию 25) самых сложных и самых простых
примеров с true/pred лейблами и уверенностью модели. Стоит явно посмотреть
на них руками — обычно самые "дорогие" ошибки это (а) короткие/неоднозначные
твиты без явной эмоциональной окраски, (б) сарказм, (в) путаница между
соседними классами (`Negative` vs `Extremely Negative`, `Positive` vs
`Extremely Positive`) — а не между полярными классами. Это стоит явно
проговорить в выводах к работе.

## exp4: какие 4 модификации BERT и почему именно они

Выбраны по признаку "максимум цитирований + разные типы модификации", чтобы
сравнение было содержательным, а не 4 варианта одного и того же трюка:

| Модель | Статья | Что меняет относительно BERT |
|---|---|---|
| **RoBERTa** | Liu et al., 2019 (~20k+ цитирований) | Тот же архитектура, но убран NSP, в 10x больше данных, динамическое маскирование, byte-level BPE — модификация **режима претрейна**, не архитектуры |
| **ALBERT** | Lan et al., 2019 (~10k+ цитирований) | Factorized embedding parameterization + cross-layer parameter sharing (все слои используют одни и те же веса) + SOP вместо NSP — модификация **архитектуры/параметризации** |
| **DistilBERT** | Sanh et al., 2019 (~9k+ цитирований) | 6-слойный student, обученный дистилляцией с полного BERT-teacher (loss = дистилляция + MLM + cosine embedding) — модификация через **knowledge distillation** |
| **ELECTRA** | Clark et al., 2020, ICLR (~5k+ цитирований) | Замена MLM на replaced-token detection: генератор портит токены, дискриминатор учится находить какие токены заменены — сигнал приходит с каждой позиции, а не с ~15% как в MLM — модификация **objective претрейна** |

Подробное обоснование и ссылки на первоисточники — прямо в комментариях
шапки каждого `exp4_alt_architectures/configs/*.yaml`.
