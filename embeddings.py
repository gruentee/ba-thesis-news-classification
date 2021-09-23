#!/usr/bin/env python
# coding: utf-8

import bz2
from pathlib import Path

# import keras_tuner
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
# from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.text import Tokenizer

np.set_printoptions(precision=4)

MAX_FEATURES = 8000
NUM_SAMPLES = 12000
NUM_TEST_SAMPLES = (NUM_SAMPLES // 100) * 30
# nr. of input vectors for which loss is calculated
BATCH_SIZE = 40
# nr. of batches to prefetch
PREFETCH_COUNT = 2
PARALLEL_READS = 4

DATA_FOLDER = Path().cwd().parent.parent.resolve() / "data" / "amazon"
TRAIN_PATH = DATA_FOLDER.joinpath("train", "train.ft.*.csv.gz")
TEST_PATH = DATA_FOLDER.joinpath("test", "test.ft.*.csv.gz")


def get_labels_and_texts(file, limit=100000):
    for line_number, line in enumerate(bz2.BZ2File(file), start=1):
        line = line.decode("utf-8")
        yield (int(line[9]) - 1), line[10:].strip()
        if 0 < line_number > limit:
            yield


def load_from_h5():
    train_h5 = pd.HDFStore(DATA_FOLDER / "train.h5")
    chunksize = 10000
    # .to_hdf("train.h5", "amazon_train", mode="w", complib="bzip2")
    pd.DataFrame(iter(get_labels_and_texts(DATA_FOLDER / "train.ft.txt.bz2")),
                 columns=["label", "text"]).to_hdf("train.h5.bz2", "amazon_train", mode="w", complib="bzip2")
    df = pd.read_fwf(DATA_FOLDER / "train.ft.txt.bz2", compression="bz2",
                     index_col=0, chunksize=chunksize)  # iterator=True)
    with df as reader:
        for chunk in reader:
            train_h5.append("amazon_train", chunk)
        # as reader:
        # train_h5.append(reader.get_chunk(chunksize))


# !curl -O https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
# !tar -xf aclImdb_v1.tar.gz


##
# Korpus (Amazon-Reviews) vorbereiten
##
# ------------------------------------------------------------
# Datein einladen
# ------------------------------------------------------------
dataset_params = dict(
    batch_size=BATCH_SIZE,
    compression_type="GZIP",
    num_parallel_reads=PARALLEL_READS,
    shuffle_seed=1337,
    select_columns=[0, 1],
    column_defaults=[np.int64(), str()],
    #     column_names=["label", "text"],
    #     label_name=["label"],
    use_quote_delim=True,
    header=True,
    #     validation_split=0.2,
)

# all_ds = tf.data.experimental.make_csv_dataset(
#     file_pattern=TRAIN_PATH.as_posix(),
#     **dataset_params
# )

raw_train_ds = tf.data.experimental.make_csv_dataset(
    file_pattern=TRAIN_PATH.as_posix(),
    **dataset_params
)
raw_val_ds = tf.data.experimental.make_csv_dataset(
    TRAIN_PATH.as_posix(),
    **dataset_params,
)
raw_test_ds = tf.data.experimental.make_csv_dataset(
    TEST_PATH.as_posix(),
    compression_type="GZIP",
    batch_size=BATCH_SIZE,
    select_columns=[0, 1],
    column_defaults=[np.int64(), str()],
    column_names=["label", "text"],
    #     label_name=["label"],
)

print(
    "Number of batches in raw_train_ds: %d" % tf.data.experimental.cardinality(raw_train_ds)
)
print(
    "Number of batches in raw_val_ds: %d" % tf.data.experimental.cardinality(raw_val_ds)
)
print(
    "Number of batches in raw_test_ds: %d"
    % tf.data.experimental.cardinality(raw_test_ds)
)

##
# Bis hierhin sind noch keine Daten eingelesen worden: dank *Eager Loading* werden die Daten erst dann, in `batch_size`-
# großen Portionen abgeholt, wenn ein Aufruf von `.take()` und eine Iteration über `.enumerate()` oder einen Schleifen-
# aufruf erfolgt.
##
# dataset = raw_train_ds.prefetch(tf.data.AUTOTUNE).take(1)
# pprint.pprint(list(dataset.enumerate()), indent=2)


tokenize_layer = Tokenizer(
    num_words=None,
    filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
    lower=True,
    split=" ",
    char_level=False,
    oov_token=None,
    document_count=0,
    #     **kwargs
)
text_ds = raw_train_ds.map(lambda feature: feature["text"])
tokenize_layer.fit_on_texts(iter(text_ds))


# In[ ]:


# Let's make a text-only dataset (no labels):
# text_ds = raw_train_ds.map(lambda x, y: x)
# Let's call `adapt`:
# tokenizer.adapt(raw_train_ds)


# In[ ]:


# tf.strings.
# TODO: text_to_sequences
def tokenize_text(text, label):
    text = tf.expand_dims(text, -1)
    return tokenize_layer(text), label


train_set = raw_train_ds.map(tokenize_text, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
test_set = raw_test_ds.map(tokenize_text, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
val_set = raw_val_ds.map(tokenize_text, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)

# train_set = raw_train_ds.map(lambda x, y: (tokenize_text(x), y), num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
# test_set = raw_test_ds.map(lambda x, y: (tokenize_text(x), y), num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
# val_set = raw_val_ds.map(lambda x, y: (tokenizer(x), y), num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
# Prefetch with a buffer size of N batches

train_set = train_set.cache().prefetch(tf.data.AUTOTUNE)
test_set = test_set.cache().prefetch(tf.data.AUTOTUNE)
val_set = val_set.cache().prefetch(tf.data.AUTOTUNE)

# In[ ]:


# TODO: pad_sequences

MAX_LENGTH = max(len(text) for text in train_texts)

# In[ ]:


model = models.Sequential()
model.add(layers.Input(shape=(MAX_LENGTH)))
# TODO: Embedding layer
# TODO: Flatten layer
model.add(layers.Dense(100, activation="sigmoid"))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(50, activation="sigmoid"))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation="sigmoid"))

# In[ ]:


model.summary()

# In[ ]:


model.compile(loss="binary_crossentropy",
              optimizer="sgd",
              metrics=["accuracy"],
              run_eagerly=True, )

model.fit(train_set,
          epochs=10,
          verbose=1,
          validation_data=val_set)

# In[ ]:


ffn.evaluate(test_set)

# In[ ]:


preds = model.predict(test_set)
print("Accuracy score: {:0.4}".format(accuracy_score(test_labels, 1 * (preds > 0.5))))
print("F1 score: {:0.4}".format(f1_score(test_labels, 1 * (preds > 0.5))))
print("ROC AUC score: {:0.4}".format(roc_auc_score(test_labels, preds)))
