"""Network intrusion detection with Keras/SciKeras and hyperparameter search.
Data: KDD Cup 1999 via scikit-learn (real dataset; downloaded on first run).
Streaming component is a local replay simulation, not production distributed streaming.
"""
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_kddcup99
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, ConfusionMatrixDisplay
from scikeras.wrappers import KerasClassifier
import keras
from pyspark.sql import SparkSession, functions as F
from keras import layers

SEED = 42
np.random.seed(SEED)
keras.utils.set_random_seed(SEED)

# Load real network flow samples: 10% KDD99 subset; binary classification
X_raw, y_raw = fetch_kddcup99(subset=None, percent10=True, return_X_y=True)
X_raw = pd.DataFrame(X_raw)
y = np.array([int(v != b'normal.') for v in y_raw], dtype=int)
# Spark local[2]: екі параллель worker thread; standalone multi-node cluster емес
spark = (SparkSession.builder.appName("Distributed-Network-IDS")
         .master("local[2]")
         .config("spark.sql.shuffle.partitions", "4")
         .getOrCreate())
spark.sparkContext.setLogLevel("ERROR")

# Шынайы KDD99-дың бір бөлігін Spark DataFrame-ге көшіру
_, X_small, _, y_small = train_test_split(
    X_raw, y, test_size=0.12, stratify=y, random_state=SEED
)
raw = X_small.copy().reset_index(drop=True)
raw.columns = [f"feature_{i}" for i in range(raw.shape[1])]
for col in raw.columns:
    raw[col] = raw[col].map(lambda v: v.decode("utf-8") if isinstance(v, bytes) else str(v))
raw["label"] = y_small.astype(int)
spark_df = spark.createDataFrame(raw).repartition(4).cache()
print("Spark partitions:", spark_df.rdd.getNumPartitions())
spark_df.groupBy("label").agg(F.count("*").alias("records")).show()
# Параллель сүзгілеу мен shuffle жұмысын Spark орындайды.
# Машинада ML модельді оқыту үшін өңделген үлгі қайта pandas-қа алынады.
selected = spark_df.filter(F.col("feature_0").cast("double") >= 0)
clean_pd = selected.toPandas()
X_small = clean_pd[[f"feature_{i}" for i in range(41)]]
X_small.columns = list(range(41))
y_small = clean_pd["label"].to_numpy(dtype=int)

X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_small, y_small, test_size=0.2, stratify=y_small, random_state=SEED)

# KDD99's categorical columns: protocol_type (1), service (2), flag (3)
cat_cols = [1, 2, 3]
num_cols = [c for c in X_train_raw.columns if c not in cat_cols]
for col in num_cols:
    X_train_raw[col] = pd.to_numeric(X_train_raw[col])
    X_test_raw[col] = pd.to_numeric(X_test_raw[col])
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols),
])
X_train = preprocessor.fit_transform(X_train_raw).astype('float32')
X_test = preprocessor.transform(X_test_raw).astype('float32')
print('Training:', X_train.shape, 'Testing:', X_test.shape)

def build_model(meta, hidden_layers=1, neurons=32, learning_rate=0.001):
    net = keras.Sequential([layers.Input(shape=(meta['n_features_in_'],))])
    for _ in range(hidden_layers):
        net.add(layers.Dense(neurons, activation='relu'))
    net.add(layers.Dense(1, activation='sigmoid'))
    net.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                loss='binary_crossentropy', metrics=['accuracy'])
    return net

base = KerasClassifier(model=build_model, verbose=0, random_state=SEED, epochs=5, batch_size=128)
params = {'model__hidden_layers':[1,2], 'model__neurons':[32,64],
          'model__learning_rate':[0.001,0.01]}

searches = {
 'GridSearchCV': GridSearchCV(base, params, cv=2, scoring='f1', n_jobs=1, refit=True),
 'RandomizedSearchCV': RandomizedSearchCV(base, params, n_iter=5, cv=2,
                                         scoring='f1', random_state=SEED, n_jobs=1, refit=True)
}
rows = []
for name, search in searches.items():
    start = time.perf_counter()
    search.fit(X_train, y_train)
    duration = time.perf_counter() - start
    pred = search.predict(X_test)
    rows.append({'method':name, 'accuracy':accuracy_score(y_test,pred),
                 'f1':f1_score(y_test,pred), 'seconds':duration,
                 'best_params':str(search.best_params_)})
    print(f'\n{name}:', search.best_params_, f'F1={f1_score(y_test,pred):.4f}')

results = pd.DataFrame(rows)
print('\nRESULTS\n', results.to_string(index=False))
results.to_csv('search_comparison.csv', index=False)
ax = results.plot(x='method', y=['accuracy','f1'], kind='bar', rot=0, ylim=(0,1.05))
ax.set_ylabel('Test score'); plt.tight_layout(); plt.savefig('scores.png', dpi=150); plt.show()
ax = results.plot(x='method', y='seconds', kind='bar', rot=0, legend=False)
ax.set_ylabel('Training time (seconds)'); plt.tight_layout(); plt.savefig('runtime.png', dpi=150); plt.show()

best_name = results.sort_values('f1', ascending=False).iloc[0]['method']
model = searches[best_name].best_estimator_
print('\nSelected:', best_name)
print(classification_report(y_test, model.predict(X_test), target_names=['Normal','Attack']))
ConfusionMatrixDisplay.from_predictions(y_test, model.predict(X_test), display_labels=['Normal','Attack'])
plt.tight_layout(); plt.savefig('confusion_matrix.png',dpi=150); plt.show()

# Spark Structured Streaming: нақты streaming қозғалтқышы, rate source — синтетикалық clock.
# Қауіпсіз демо: stream event ID-лерін бұрынғы нақты test жазбаларына сәйкестендіреміз.
# NN inference driver жағында орындалады (distributed inference емес).
print("\\nSPARK STRUCTURED STREAMING DEMO:")
MAX_EVENTS = min(128, len(X_test))
stream_events = (spark.readStream.format("rate")
                 .option("rowsPerSecond", 20).load()
                 .select((F.col("value") % F.lit(MAX_EVENTS)).cast("int").alias("event_id")))

stream_log = []
def on_batch(batch_df, batch_id):
    ids = [int(r["event_id"]) for r in batch_df.select("event_id").collect()]
    if not ids:
        return
    t0 = time.perf_counter()
    preds = model.predict(X_test[ids]).astype(int)
    latency = (time.perf_counter()-t0)*1000
    stream_log.append({"batch":int(batch_id), "events":len(ids),
                       "alerts":int(preds.sum()), "latency_ms":latency})
    print(f"Microbatch {batch_id}: events={len(ids)}, alerts={preds.sum()}, latency={latency:.1f} ms")

query = (stream_events.writeStream.foreachBatch(on_batch)
         .trigger(processingTime="3 seconds").start())
# Демонстрация ұзақтығы: шамамен 12 секунд, query.stop() ағынды тоқтатады.
try:
    query.awaitTermination(12)
finally:
    query.stop()
pd.DataFrame(stream_log).to_csv("stream_batches.csv", index=False)
spark.stop()
