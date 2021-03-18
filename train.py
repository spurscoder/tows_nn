""" 这里放训练的过程代码

包括了数据加载，模型构建，模型保存，loss日志等
"""


import bert4keras
import keras
import json_lines
import numpy as np
import tensorflow as tf
from bert4keras.backend import keras, K
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open, ViterbiDecoder
from bert4keras.layers import ConditionalRandomField
from keras.layers import Dense
from keras.models import Model
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from keras.layers import Bidirectional, LSTM, Dense, Dropout
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
from packaging import version

set_gelu = "tanh"
maxlen = 512
epochs = 60
batch_size = 16
bert_layers = 12
learing_rate = 1e-5  # bert_layers越小，学习率应该要越大

rnn_units = 128
drop_rate = 0.5
labels=[]
id2label={}
label2id={}
num_labels=0

model=0
tokenizer = 0
train_generator=0
valid_generator=0
test_generator=0

#日志文件目录
logdir =''
tensorboard_callback = 0
# bert配置
config_path = "/home/user2/albert_base/albert_config.json"
checkpoint_path = "/home/user2/albert_base/model.ckpt-best"
dict_path = "/home/user2/albert_base/vocab.txt"

def para():
    global label2id
    global num_labels
    global labels
    global id2label
    a = set()
    with open("/home/user2/webservicefile/ws5.jl.txt", "r") as f:
        for item in json_lines.reader(f):
            a.add(item["tag"])
    labels = list(a)
    id2label = dict(enumerate(labels))
    label2id = {j: i for i, j in id2label.items()}
    num_labels = len(labels)


def load_file(filename):
    D = []
    with open(filename, "r") as f:
        for item in json_lines.reader(f):
            apiintro1, apiintro2, label = item["intro1"], item["intro2"], item["tag"]
            #                 d=[]
            #                 d.append(item['name']+'/t/t'+item['intro'])
            #                 d.append(tag)
            D.append((apiintro1, apiintro2, label2id[label]))
    return D


class data_generator(DataGenerator):
    """数据生成器
    """

    def __iter__(self, random=False):
        global tokenizer
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text1, text2, label) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(text1, text2, maxlen=maxlen)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


def evaluate(data):
    total, right = 0.0, 0.0
    for x_true, y_true in data:
        y_pred = model.predict(x_true).argsort(axis=1, kind="heapsort")[:, ::-1][:, :1]
        y_true = y_true[:, 0]
        total += len(y_true)
        for inx, tag in enumerate(y_true):
            if tag in y_pred[inx, :]:
                right += 1
    return right / total


class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """

    def __init__(self):
        self.best_val_acc = 0.0

    def on_epoch_end(self, epoch, logs=None):
        val_acc = evaluate(valid_generator)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            model.save_weights("relationdetect_best_model.weights")
        test_acc = evaluate(test_generator)
        print(
            "val_acc: %.5f, best_val_acc: %.5f, test_acc: %.5f\n"
            % (val_acc, self.best_val_acc, test_acc)
        )


def main():
    para()
    wsdata = load_file("/home/user2/webservicefile/ws5.jl.txt")

    train_set, test_data = train_test_split(wsdata, test_size=0.05, random_state=30)
    train_data, valid_data = train_test_split(train_set, test_size=0.05, random_state=30)

    print(f"{'train':<6}: {len(train_data)}\n{'val':<6}: {len(valid_data)}\n{'test':<6}: {len(test_data)}")

    global tokenizer
    global model
    global train_generator
    global valid_generator
    global test_generator
    global tensorboard_callback
    global logdir
    # 建立分词器
    tokenizer = Tokenizer(dict_path, do_lower_case=True)

    model = build_transformer_model(config_path, checkpoint_path, model="albert")
    output_layer = "Transformer-FeedForward-Norm"
    output = model.get_layer(output_layer).get_output_at(bert_layers - 1)
    print("albert last layer shape: {output.shape}")

    output = LSTM(units=rnn_units, return_sequences=True)(output)
    output = LSTM(units=rnn_units, return_sequences=False, go_backwards=True)(output)
    print("lstm layer shape: {output.shape}")
    output = Dropout(drop_rate)(output)
    output = Dense(num_labels)(output)
    output = Dense(num_labels, activation="softmax")(output)
    print("dense shape: {output.shape}")
    model = Model(model.input, output)
    model.summary()

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=Adam(learing_rate),
        metrics=["accuracy"],
    )

    train_generator = data_generator(train_data, batch_size)
    valid_generator = data_generator(valid_data, batch_size)
    test_generator = data_generator(test_data, batch_size)

    evaluator = Evaluator()

    logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
    model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=60,
        callbacks=[evaluator, tensorboard_callback],
    )

    model.load_weights("3_best_model.weights")
    print("final test acc: %05f\n" % (evaluate(test_generator)))


if __name__ == "__main__":
    main()
