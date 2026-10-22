# Distributed Network Intrusion Detection — PySpark + Keras

**Үлестірілген жүйелер пәніне арналған оқу прототипі.**

KDD Cup 1999 желілік трафик мәліметтерін Spark DataFrame арқылы 4 partition-ға бөліп параллель сүзгілейді және топтайды. Keras DNN моделін SciKeras + `GridSearchCV` / `RandomizedSearchCV` арқылы салыстырады. Spark Structured Streaming `rate` ағыны арқылы held-out желілік жазбалардың болжамын microbatch-пен шығарады.

## Google Colab

```python
!pip -q install "pyspark>=3.5" "scikeras>=0.13" "scikit-learn>=1.5"
!python main.py
```

Colab ішінде файлды `main.py` ретінде жүктеңіз; бірінші жүктеуде интернет қажет. `search_comparison.csv`, `scores.png`, `runtime.png`, `confusion_matrix.png`, `stream_batches.csv` генерацияланады.

## Архитектура

KDD99 → Spark (`local[2]`, repartition/aggregation/filter) → pandas/Sklearn preprocessing → Keras DNN + hyperparameter search → metrics → Spark Structured Streaming microbatches → Keras inference on driver.

## Шектеулер

- `local[2]` — бір Colab машинасындағы екі параллель thread, көпсерверлі кластер емес. Spark distributed API және partition/shuffle механизмдері нақты қолданылады.
- Streaming көзі `rate`: нақты network capture немесе Kafka емес. Event ID-лері шынайы KDD99 тест мысалдарын қайта ойнатады.
- DNN оқыту және inference **драйверде**; GPU/cluster-based distributed training емес.
- KDD99 — ескірген тарихи benchmark, production intrusion detection жүйесі емес.
- Ешбір accuracy, F1 немесе latency нәтижесі алдын ала ойдан берілмейді: бағдарламаны орындау қажет.

## Дереккөздер

- KDD Cup 1999: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_kddcup99.html
- Spark Structured Streaming: https://spark.apache.org/docs/latest/streaming/index.html
- SciKeras: https://adriangb.com/scikeras/stable/
- Архитектуралық идеямен салыстыруға болатын сыртқы жоба: https://github.com/shruti-sivakumar/Realtime-Network-IDS

Бұл жоба — аталған ашық дерек пен құралдарды пайдаланатын бөлек оқу прототипі.
