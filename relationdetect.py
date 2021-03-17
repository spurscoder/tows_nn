#!/usr/bin/env python
# coding: utf-8

# In[26]:


import bert4keras
import keras
import json_lines

bert4keras.__version__, keras.__version__


# In[27]:


"""
预测：step2
训练：step2
"""
import numpy as np
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


# In[28]:


"""
预测：step3
训练：step3
"""

set_gelu = "tanh"

maxlen = 512
epochs = 60
batch_size = 16
bert_layers = 12
learing_rate = 1e-5  # bert_layers越小，学习率应该要越大

# bert配置
config_path = "/home/user2/albert_base/albert_config.json"
checkpoint_path = "/home/user2/albert_base/model.ckpt-best"
dict_path = "/home/user2/albert_base/vocab.txt"


# In[29]:


a = set()
with open("/home/user2/webservicefile/ws5.jl.txt", "r") as f:
    for item in json_lines.reader(f):
        for tag in item["tag"]:
            a.add(tag)
labels = list(a)

# 类别映射

id2label = dict(enumerate(labels))
label2id = {j: i for i, j in id2label.items()}
num_labels = len(labels)


# In[30]:


def load_file(filename):
    D = []
    with open(filename, "r") as f:
        for item in json_lines.reader(f):
            for tag in item["tag"]:
                apiintro1, apiintro2, label = item["intro1"], item["intro2"], tag
                #                 d=[]
                #                 d.append(item['name']+'/t/t'+item['intro'])
                #                 d.append(tag)
                D.append((apiname, apiintro, label2id[label]))
    return D


wsdata = load_file("/home/user2/webservicefile/ws5.jl.txt")


# In[31]:


from sklearn.model_selection import train_test_split

train_set, test_data = train_test_split(wsdata, test_size=0.05, random_state=30)
train_data, valid_data = train_test_split(train_set, test_size=0.05, random_state=30)


# In[32]:


print(
    f"{'train':<6}: {len(train_data)}\n{'val':<6}: {len(valid_data)}\n{'test':<6}: {len(test_data)}"
)


# In[33]:


"""
预测：step4
训练：step4
"""
# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)


class data_generator(DataGenerator):
    """数据生成器
    """

    def __iter__(self, random=False):
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


# In[34]:


"""
模型√
预测：step5
训练：step7
"""
from keras.layers import Bidirectional, LSTM, Dense, Dropout

rnn_units = 128
drop_rate = 0.5
"""
后面的代码使用的是bert类型的模型，如果你用的是albert，那么前几行请改为：
model = build_transformer_model(
    config_path,
    checkpoint_path,
    model='albert',
)
output_layer = 'Transformer-FeedForward-Norm'
output = model.get_layer(output_layer).get_output_at(bert_layers - 1)
"""

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


# In[35]:


# 转换数据集
train_generator = data_generator(train_data, batch_size)
valid_generator = data_generator(valid_data, batch_size)
test_generator = data_generator(test_data, batch_size)


# In[36]:


def evaluate(data):
    total, right = 0.0, 0.0
    for x_true, y_true in data:
        y_pred = model.predict(x_true).argsort(axis=1, kind="heapsort")[:, ::-1][:, :5]
        y_true = y_true[:, 0]
        total += len(y_true)
        for inx, tag in enumerate(y_true):
            if tag in y_pred[inx, :]:
                right += 1
    return right / total


# In[37]:


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


# In[ ]:


if __name__ == "__main__":

    evaluator = Evaluator()

    model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=60,
        callbacks=[evaluator],
    )

    model.load_weights("3_best_model.weights")
    print("final test acc: %05f\n" % (evaluate(test_generator)))

else:

    model.load_weights("3_best_model.weights")


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[96]:


import numpy

demo_train = train_data[:4]
y_pred = numpy.array(
    [
        [0.1, 0.5, 0.8, 0.9, 478],
        [5.0, 347, 3.2, 0.5, 5],
        [35, 12, 32, 44, 0.5],
        [4.7, 6.7, 219, 1.5, 0.3],
    ]
)
print(y_pred)
y_pred = y_pred.argsort(axis=1, kind="heapsort")[:, ::-1][:, :2]
print(y_pred)
train_generator = data_generator(demo_train, 4)
total, right = 0.0, 0.0
for x_true, y_true in train_generator:
    #     y_pred = np.model.predict(x_true).argsort(axis=1,kind='heapsort')
    y_true = y_true[:, 0]
    y_true = [4, 5, 5, 6]
    total += len(y_true)
    print(y_true)
    for inx, tag in enumerate(y_true):
        if tag in y_pred[inx, :]:
            right += 1
print(right)
# right / total

# [[0 1 2 3 4]
#  [3 2 0 4 1]
#  [4 1 2 0 3]
#  [4 3 0 1 2]]


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


"""
预测：step6
训练：step8
"""


class KeywordsGenerator(ViterbiDecoder):
    """
    关键词生成器
    """

    def generator(self, text):
        tokens = tokenizer.tokenize(
            text
        )  # ['[CLS]', '我', '们', '变', '而', '以', '书', '会', '友', '。', '[SEP]']
        while len(tokens) > 512:
            tokens.pop(-2)  # 删除SEP之前的 也就是倒数第二个
        mapping = tokenizer.rematch(
            text, tokens
        )  # [[], [0], [1], [2], [3], [4], [5], [6], [7], [8], []]
        token_ids = tokenizer.tokens_to_ids(
            tokens
        )  # [101, 2769, 812, 1359, 5445, 809, 741, 833, 1351, 8024, 102]
        segment_ids = [0] * len(token_ids)
        nodes = model.predict([[token_ids], [segment_ids]])[0]
        labels = self.decode(nodes)
        entities, starting = [], False
        for i, label in enumerate(labels):
            if label > 0:
                if label % 2 == 1:
                    starting = True
                    entities.append([[i], id2label[(label - 1) // 2]])
                elif starting:
                    entities[-1][0].append(i)
                else:
                    starting = False
            else:
                starting = False

        return [
            (text[mapping[w[0]][0] : mapping[w[-1]][-1] + 1], l) for w, l in entities
        ]


NER = NamedEntityRecognizer(trans=K.eval(CRF.trans), starts=[0], ends=[0])


# In[ ]:


"""
训练：step9
"""


def evaluate(data):
    """评测函数
    """
    X, Y, Z = 1e-10, 1e-10, 1e-10
    for d in tqdm(data):
        text = "".join([i[0] for i in d])
        R = set(NER.recognize(text))
        T = set([tuple(i) for i in d if i[1] != "O"])
        X += len(R & T)
        Y += len(R)
        Z += len(T)
    f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
    return f1, precision, recall


class Evaluate(keras.callbacks.Callback):
    def __init__(self):
        self.best_val_f1 = 0

    def on_epoch_end(self, epoch, logs=None):
        trans = K.eval(CRF.trans)
        NER.trans = trans
        print(NER.trans)
        f1, precision, recall = evaluate(valid_data)
        # 保存最优
        if f1 >= self.best_val_f1:
            self.best_val_f1 = f1
            model.save_weights("./best_model.weights")
        print(
            "valid:  f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n"
            % (f1, precision, recall, self.best_val_f1)
        )
        f1, precision, recall = evaluate(test_data)
        print(
            "test:  f1: %.5f, precision: %.5f, recall: %.5f\n" % (f1, precision, recall)
        )


# In[ ]:


"""
训练：step10
"""
evaluator = Evaluate()
train_generator = data_generator(train_data, batch_size)

model.fit_generator(
    train_generator.forfit(),
    steps_per_epoch=len(train_generator),
    epochs=epochs,
    callbacks=[evaluator],
)


# In[ ]:


"""
预测：step7
"""
model.load_weights("./best_model.weights")
