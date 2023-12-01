import tensorflow as tf
import numpy as np
import json
import os
os.environ["TF_ENABLE_MLIR_OPTIMIZATIONS"] = "5"
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.layers import MultiHeadAttention

# print("GPU可用：" if tf.test.is_gpu_available() else "GPU不可用")

os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 指定使用GPU设备编号为0

from flask import Flask, request, jsonify

app = Flask(__name__)

import json
from transformers import XLNetTokenizer,XLNetModel

tokenizer = XLNetTokenizer.from_pretrained('embd/chinese-xlnet-mid')
xlnet = XLNetModel.from_pretrained('embd/chinese-xlnet-mid')

import re


def split_text_into_sentences(text):
    # 中文标点符号列表
    punctuation = ['。', '！', '？', '；', '……','.']

    # 根据标点符号进行分句
    sentences = re.split('([。！？；…….])', text)

    # 将分句和标点符号重新组合
    result = []
    for i in range(0, len(sentences), 2):
        sentence = sentences[i].strip()
        if sentence:
            # 在分句后添加对应的标点符号
            sentence += sentences[i + 1]
            result.append(sentence)

    return result

def process_essay(essay):
    # 分行符为'\n'，根据分行将作文分段
    paragraphs = essay.split('\n')

    result = []
    for paragraph in paragraphs:
        # 去除段落两端的空格和换行符
        paragraph = paragraph.strip()
        if paragraph:
            # 分句
            sentences = split_text_into_sentences(paragraph)
            result.append(sentences)

    return result

def process(d):
    title = d['topic']
    essay = d['text']
    processed_essay = process_essay(essay)
    return title, processed_essay

def get_essay_representation(title,input_sent):
    input = []
    EDUbreak = []
    lenss = []
    token_title = tokenizer.tokenize(title)
    input.extend(token_title)
    # print(input)
    lenss.append(len(token_title))
    input.append('SEP')
    EDUbreak.append(len(input) - 1)
    p_l = []
    for i in range(len(input_sent)):
        if len(input_sent[i])>20:
            input_sent[i] = input_sent[i][:20]
        p_l.append(len(input_sent[i]))
        for j in range(len(input_sent[i])):
            token = tokenizer.tokenize(input_sent[i][j])
            lenss.append(len(token))
            input.extend(token)
            input.append('SEP')
            EDUbreak.append(len(input)-1)
    input.append('CLS')
    print(input)
    # out_isd = tokenizer.convert_tokens_to_ids(input)
    out_isd = tokenizer.convert_tokens_to_ids(input)


    token_tensor = tf.convert_to_tensor([out_isd])
    sentence_vector = xlnet(token_tensor)

    output = sentence_vector[0].cpu().detach().numpy().tolist()

    return output,EDUbreak,p_l
# title,essay = process(d)
# all_embedding,all_Edu_breaks,p_l = get_essay_representation(title,essay)
# print('p_l',p_l)
def get_emb(EncoderOutputs,EDU_breaks):
    print('edu',EDU_breaks)
    lst = []
    if EDU_breaks[0]==0:
        EDU_breaks= EDU_breaks[1:]
    print(len(EncoderOutputs))
    print('2',len(EncoderOutputs[0]),len(EncoderOutputs[0][0]))
    for j in range(len(EDU_breaks)):
        lst.append(EncoderOutputs[0][EDU_breaks[j]])

    return lst
# all_embedding = get_emb(all_embedding,all_Edu_breaks)
# print(len(all_embedding))
def get_embedd(d):
    title, essay = process(d)
    all_embedding, all_Edu_breaks, p_l = get_essay_representation(title, essay)
    para_sent = p_l
    emb = get_emb(all_embedding,all_Edu_breaks)
    p = []
    p.append(emb[0])
    for j in range(len(para_sent)):
        e = emb[sum(para_sent[:j])+1:sum(para_sent[:j+1])+1]
        if len(e)!=768:
            p.append(e)
        else:
            p.append([e])
    for j in range(1,len(p)):
        if len(p[j])<20:
            p[j]+=[[0]*768]*(20-len(p[j]))
    if len(p)<21:
        p +=[20*[[0]*768]]*(21-len(p))
    else:
        p = p[:21]
    title = tf.convert_to_tensor(p[0])[tf.newaxis, :]
    para = tf.convert_to_tensor(p[1:])[tf.newaxis, :]
    print(title.shape,para.shape)
    return title,para



# 定义自定义模型类
class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.bilstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=256,return_sequences=True))
        # self.attention = tf.keras.layers.Attention()
        # self.attention = MultiHeadAttention(num_heads=8, key_dim=512)

        self.bilstm_over = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=256,return_sequences=True))
        self.bilstm_stru = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=256,return_sequences=True))
        self.bilstm_top = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=256,return_sequences=True))
        self.bilstm_log = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=256,return_sequences=True))
        self.bilstm_lang = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=256,return_sequences=True))
        self.w_omega = tf.keras.layers.Dense(512)
        self.u_omega = tf.keras.layers.Dense(1)
        self.v = tf.Variable(tf.random.uniform((8, 512, 1)), trainable=True)
        self.vv = tf.Variable(tf.random.uniform((20, 512, 1)), trainable=True)
        self.vvv = tf.Variable(tf.random.uniform((2, 512, 1)), trainable=True)
        self.vvvv = tf.Variable(tf.random.uniform((16, 512, 1)), trainable=True)
        self.vvvvt = tf.Variable(tf.random.uniform((1, 512, 1)), trainable=True)
        self.vvvvv = tf.Variable(tf.random.uniform((14, 512, 1)), trainable=True)



        self.dense = tf.keras.layers.Dense(64,trainable=True)
        self.dense1 = tf.keras.layers.Dense(3, activation='tanh',trainable=True)
        self.dense2 = tf.keras.layers.Dense(3, activation='tanh',trainable=True)
        self.dense3 = tf.keras.layers.Dense(3, activation='tanh',trainable=True)
        self.dense4 = tf.keras.layers.Dense(3, activation='tanh',trainable=True)
        self.dense5 = tf.keras.layers.Dense(3, activation='tanh',trainable=True)

    def call(self, inputs,idx1):
        print('inp',inputs.shape,idx1.shape)
        batch_size = tf.shape(inputs)[0]
        compressed = tf.reshape(inputs, (batch_size*inputs.shape[1], inputs.shape[2], inputs.shape[3]))
        # print('com',compressed.shape)
        lstm_output = self.bilstm(compressed)
        # print('x',lstm_output.shape)
        u = tf.keras.activations.tanh(self.w_omega(lstm_output))
        att = self.u_omega(u)
        att_score = tf.keras.activations.softmax(att, axis=1)
        # Apply element-wise multiplication
        scored_x = lstm_output * att_score
        # Sum over the time dimension
        summed_x = tf.reduce_sum(scored_x, axis=1)
        summed_x = tf.reshape(summed_x, (batch_size,inputs.shape[1],-1))
        # print('sum',summed_x.shape)
        out_over = self.bilstm_over(summed_x)
        out_overall = tf.reduce_mean(out_over, axis=1, keepdims=True)
        out_stru = self.bilstm_stru(summed_x)
        out_structure = tf.reduce_mean(out_stru, axis=1, keepdims=True)
        out_top = self.bilstm_top(summed_x)
        out_topic = tf.reduce_mean(out_top, axis=1, keepdims=True)
        out_log = self.bilstm_log(summed_x)
        out_logic = tf.reduce_mean(out_log, axis=1, keepdims=True)
        out_lang = self.bilstm_lang(summed_x)
        # print('lang',out_lang.shape)
        out_language = tf.reduce_mean(out_lang, axis=1, keepdims=True)
        # print('lang',out_lang.shape)
        ou = tf.concat([out_overall, out_structure, out_topic, out_logic, out_language], axis=1)

        o = tf.transpose(ou, perm=[0, 2, 1])
        v = tf.transpose(self.v, perm=[0, 2, 1])
        vv = tf.transpose(self.vv, perm=[0, 2, 1])
        vvv = tf.transpose(self.vvv, perm=[0, 2, 1])
        vvvv = tf.transpose(self.vvvv, perm=[0, 2, 1])
        vvvvt = tf.transpose(self.vvvvt, perm=[0, 2, 1])
        vvvvv = tf.transpose(self.vvvvv, perm=[0, 2, 1])
        #print(o.shape[0])
        if o.shape[0] == 8:
            alpha = tf.nn.softmax(tf.matmul(v, o), axis=2)  # 1*512*b2 ,b2*512*20 = 1*20
        elif o.shape[0] == 20:
            alpha = tf.nn.softmax(tf.matmul(vv, o), axis=2)  # 1*512*b2 ,b2*512*20 = 1*20
        elif o.shape[0] == 2:
            alpha = tf.nn.softmax(tf.matmul(vvv, o), axis=2)  # 1*512*b2 ,b2*512*20 = 1*20
        elif o.shape[0] == 1:
            alpha = tf.nn.softmax(tf.matmul(vvvvt, o), axis=2)  # 1*512*b2 ,b2*512*20 = 1*20
        elif o.shape[0] == 14:
            alpha = tf.nn.softmax(tf.matmul(vvvvv, o), axis=2)  # 1*512*b2 ,b2*512*20 = 1*20
        else:
            alpha = tf.nn.softmax(tf.matmul(vvvv, o), axis=2)  # 1*512*b2 ,b2*512*20 = 1*20
        alpha_1 = tf.transpose(alpha, perm=[0, 2, 1])
        output = tf.tanh(tf.matmul(o, alpha_1))  # (768*3)*(3*1)=(768*1)
        out_put = tf.squeeze(output, axis=2)
        #print('ou',output.shape)

        idx_1 = self.dense(idx1)
        out_topic = tf.concat([out_put, idx_1], axis=1)
        out0 = self.dense1(out_put)
        out1 = self.dense2(out_put)
        out2 = self.dense3(out_topic)
        out3 = self.dense4(out_put)
        out4 = self.dense5(out_put)

        return out0, out1, out2, out3, out4


# 数据批处理大小
batch_size = 16

# 创建模型实例
with tf.device('/GPU:0'):  # 将模型移动到GPU设备上
    model = MyModel()

initial_learning_rate=0.0005
# 定义损失函数和优化器
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

# 定义评估指标
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
train_accuracy1 = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy1')
train_accuracy2 = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy2')
train_accuracy3 = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy3')
train_accuracy4 = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy4')

val_loss = tf.keras.metrics.Mean(name='val_loss')
val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')
val_accuracy1 = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy1')
val_accuracy2 = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy2')
val_accuracy3 = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy3')
val_accuracy4 = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy4')


# 定义训练步骤
@tf.function
def train_step(inputs,topics, labels):
    with tf.GradientTape() as tape:
        outputs, outputs1, outputs2, outputs3, outputs4  = model(inputs,topics)
        #print(labels,outputs3)
        loss0 = loss_object([item[0] for item in labels], outputs)
        loss1 = loss_object([item[1] for item in labels], outputs1)
        loss2 = loss_object([item[2] for item in labels], outputs2)
        loss3 = loss_object([item[3] for item in labels], outputs3)
        loss4 = loss_object([item[4] for item in labels], outputs4)
        loss = (loss0 + loss2 + loss1 + loss3 + loss4) / 5

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        #print('sss',tf.convert_to_tensor([item[1] for item in labels], dtype=tf.int32).shape, tf.argmax(outputs, axis=1).shape)

        train_accuracy.update_state(tf.convert_to_tensor([item[0] for item in labels], dtype=tf.int32), outputs)
        train_accuracy1(tf.constant([item[1] for item in labels]), outputs1)
        train_accuracy2(tf.constant([item[2] for item in labels]), outputs2)
        train_accuracy3(tf.constant([item[3] for item in labels]), outputs3)
        train_accuracy4(tf.constant([item[4] for item in labels]), outputs4)


# 定义验证步骤
@tf.function
def val_step(inputs,topics, labels):
    outputs, outputs1, outputs2, outputs3, outputs4 = model(inputs, topics)
    loss0 = loss_object([item[0] for item in labels], outputs)
    loss1 = loss_object([item[1] for item in labels], outputs1)
    loss2 = loss_object([item[2] for item in labels], outputs2)
    loss3 = loss_object([item[3] for item in labels], outputs3)
    loss4 = loss_object([item[4] for item in labels], outputs4)
    loss = (loss0 + loss2 + loss1 + loss3 + loss4) / 5
    # predictions = model(inputs,topics)
    # print('val',labels,predictions)

    # loss = loss_object(labels, predictions)
    val_loss(loss)
    #print('val',outputs)
    val_accuracy(tf.constant([item[0] for item in labels]), outputs)
    val_accuracy1(tf.constant([item[1] for item in labels]), outputs1)
    val_accuracy2(tf.constant([item[2] for item in labels]), outputs2)
    val_accuracy3(tf.constant([item[3] for item in labels]), outputs3)
    val_accuracy4(tf.constant([item[4] for item in labels]), outputs4)


def get_embedding():
    f = open('al_embedding.txt','r',encoding='utf-8')
    # f.write(str(all_embedding))
    f1 = open('para_sent.txt','r',encoding='utf-8')
    # f1.write(str(para_len))
    para_sent = eval(f1.read())
    emb = eval(f.read())
    p = []
    for i in range(len(para_sent)):
        para_i = []
        para_i.append([emb[i][0]])
        for j in range(len(para_sent[i])):
            e = emb[i][sum(para_sent[i][:j])+1:sum(para_sent[i][:j+1])+1]
            if len(e)!=768:
                para_i.append(e)
            else:
                para_i.append([e])
        p.append(para_i)
    for i in range(len(p)):
        for j in range(1,len(p[i])):
            if len(p[i][j])<20:
                p[i][j]+=[[0]*768]*(20-len(p[i][j]))
        if len(p[i])<21:
            p[i] +=[20*[[0]*768]]*(21-len(p[i]))
        else:
            p[i] = p[i][:21]
    title = []
    para = []
    for item in p:
        # print(len(item))
        title.append(item[0])
        para.append(item[1:])
        # print(len(title),len(para))
    title_tensor = tf.convert_to_tensor(title)
    title_tensor = tf.squeeze(title_tensor, axis=1)

    para_tensor = tf.convert_to_tensor(para)

    return title_tensor, para_tensor
def getRandomIndex(n, x):
    # 索引范围为[0, n)，随机选x个不重复，注意replace=False才是不重复，replace=True则有可能重复
    index = np.random.choice(n, size=x, replace=False)
    return index

f = open('cv_folds.txt', 'r', encoding='utf-8')
import numpy as np
lines = f.readlines()
i = 1
line = lines[3]
line = eval(line.strip())
all = list(range(1220))
test_data_index = [i-1 for i in line]
train_data_index = [item for item in all if item not in test_data_index]
dev_data_index = getRandomIndex(train_data_index, 98)
dev_data_index = list(dev_data_index)
train_data_index = [item for item in train_data_index if item not in dev_data_index]#20 *20
f = open('score_ult.json', 'r', encoding='utf-8')
lines = f.readlines()
title,emb = get_embedding()
d = {'Bad':0,'Medium':1,'Great':2}
train_labels = [json.loads(lines[i]) for i in train_data_index]
train_labels = [[d[item['score']],d[item['stru_score']],d[item['topic_score']],d[item['logic_score']],d[item['lang_score']]] for item in train_labels]
emb_tensor = tf.constant(emb, dtype=tf.float32)
train_data_index_tensor = tf.constant(train_data_index, dtype=tf.int32)
train_data = tf.gather(emb_tensor, train_data_index_tensor)
# train_data = emb[train_data_index]
topic_tensor = tf.constant(title, dtype=tf.float32)
train_topic = tf.gather(topic_tensor, train_data_index_tensor)
test_labels = [json.loads(lines[i]) for i in test_data_index]
test_labels = [[d[item['score']],d[item['stru_score']],d[item['topic_score']],d[item['logic_score']],d[item['lang_score']]] for item in test_labels]


# test_data = emb[test_data_index]
test_data_index_tensor = tf.constant(test_data_index, dtype=tf.int32)
test_data = tf.gather(emb_tensor, test_data_index_tensor)
# test_topic = title[test_data_index]
test_topic_tensor = tf.constant(title, dtype=tf.float32)
test_topic = tf.gather(test_topic_tensor, test_data_index_tensor)
val_label = [json.loads(lines[i]) for i in dev_data_index]
val_label = [[d[item['score']],d[item['stru_score']],d[item['topic_score']],d[item['logic_score']],d[item['lang_score']]] for item in val_label]

val_data_index_tensor = tf.constant(dev_data_index, dtype=tf.int32)
val_data = tf.gather(emb_tensor, val_data_index_tensor)
# val_data = emb[dev_data_index]
val_topic_tensor = tf.constant(title, dtype=tf.float32)
val_topic = tf.gather(val_topic_tensor, val_data_index_tensor)
# val_topic = title[dev_data_index]
last_val_acc = 0
last_val_acc1 = 0
last_val_acc2 = 0
last_val_acc3 = 0
last_val_acc4 = 0
def main():
    epochs = 5
    for epoch in range(epochs):
        train_loss.reset_states()
        train_accuracy.reset_states()
        val_loss.reset_states()
        val_accuracy.reset_states()

        # 分批训练数据
        num_batches = len(train_data) // batch_size
        for batch_index in range(num_batches):
            start_index = batch_index * batch_size
            end_index = start_index + batch_size
            inputs = train_data[start_index:end_index]
            topics = train_topic[start_index:end_index]
            labels = train_labels[start_index:end_index]
            # print('1')
            train_step(inputs,topics, labels)

        # 处理最后一个批次
        if len(train_data) % batch_size != 0:
            start_index = num_batches * batch_size
            inputs = train_data[start_index:]
            topics = train_topic[start_index:]
            labels = train_labels[start_index:]
            # print('2')
            train_step(inputs, topics, labels)


        # 分批验证数据
        num_val_batches = len(val_data) // batch_size
        for val_batch_index in range(num_val_batches):
            val_start_index = val_batch_index * batch_size
            val_end_index = val_start_index + batch_size
            val_inputs = val_data[val_start_index:val_end_index]
            val_topics = val_topic[val_start_index:val_end_index]
            val_labels = val_label[val_start_index:val_end_index]
            #print('val',val_inputs.shape,val_labels.shape)
            val_step(val_inputs,val_topics, val_labels)

        print(f'Epoch {epoch+1}: Train Loss: {train_loss.result()}, Train Accuracy: {train_accuracy.result()},Train Accuracy1: {train_accuracy1.result()},\
        Train Accuracy2: {train_accuracy2.result()},Train Accuracy3: {train_accuracy3.result()},Train Accuracy4: {train_accuracy4.result()},\
         Val Loss: {val_loss.result()}, Val Accuracy: {val_accuracy.result()}, Val Accuracy1: {val_accuracy1.result()}, Val Accuracy2: {val_accuracy2.result()}, \
         Val Accuracy3: {val_accuracy3.result()}, Val Accuracy4: {val_accuracy4.result()}')
        if val_accuracy.result().numpy()>float(last_val_acc) or val_accuracy1.result().numpy()>float(last_val_acc1) or val_accuracy2.result().numpy()>float(last_val_acc2) or val_accuracy3.result().numpy()>float(last_val_acc3) or val_accuracy4.result().numpy()>float(last_val_acc4):
            last_val_acc = val_accuracy.result().numpy()
            last_val_acc1 = val_accuracy1.result().numpy()
            last_val_acc2 = val_accuracy2.result().numpy()
            last_val_acc3 = val_accuracy3.result().numpy()
            last_val_acc4 = val_accuracy4.result().numpy()
            model.save('model_final/'+'{}'.format(epoch+1))
main()
# # # 示例用法
d = {"topic":"勇气","text": """第一段是简介，介绍作文的主题和背景。哦哦哦！
第二段是论点，提出自己的观点和理由。发发发？
第三段是论证，给出事实和数据支持观点。嘻嘻嘻.
第四段是反驳，回应可能的反对意见。
最后一段是结论，总结全文，并提出展望或建议。"""}
@app.route('/ooo', methods=['POST'])
def predict():
    data = request.get_json()  # 获取请求的 JSON 数据
    topic = data.get("topic")  # 获取 JSON 中的 'topic' 字段值
    text = data.get("text")  # 获取 JSON 中的 'text' 字段值

    d = {"topic":topic,"text":text}
    print(1)
    # 指定 SavedModel 路径
    model_path = 'model/5'

    # 加载模型
    loaded_model = tf.saved_model.load(model_path)

    # 创建输入张量
    input_shape = (15, 20, 20, 768)
    inputs = tf.random.normal(input_shape)
    input_topic_shape = (15, 768)
    input_topic = tf.random.normal(input_topic_shape)
    # 推断
    test_top, test_da = get_embedd(d)
    # test_da = tf.expand_dims(test_data[0], axis=0)
    # test_top = tf.expand_dims(test_topic[0], axis=0)
    data = tf.concat([test_da, inputs], axis=0)
    topic = tf.concat([test_top, input_topic], axis=0)

    # 测试模型
    print('test',data.shape,topic.shape)
    prediction,prediction1,prediction2,prediction3,prediction4 = loaded_model(data,topic)
    print(f'Test Predictions: {prediction[0],prediction1[0],prediction2[0],prediction3[0],prediction4[0]}')
    result = {"overall_score":prediction[0],"org_score":prediction1[0],"topic_score":prediction2[0],"logic_score":prediction3[0],"langu_score":prediction4[0]}
    return jsonify(result)

#predict(d)
#if __name__ == '__main__':
#    app.run(port=8888)
