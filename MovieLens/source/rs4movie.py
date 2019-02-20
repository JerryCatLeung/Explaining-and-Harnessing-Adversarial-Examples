#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@project:MovieLens
@author: Kevin Leung
@file:rs4movie.py
@interpreter: Python3.6
@ide:PyCharm
@time:2019-02-12 16:14:48
"""
import datetime
import pickle
import random
import re
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split


def load_data():
    """
    Load Data set from File
    数据预处理
    """
    # 读取User数据
    users_title = ["UserID", "Gender", "Age", "JobID", "Zip-code"]
    users = pd.read_csv(r"./users.dat", sep="::", header=None, names=users_title, engine="python")
    users = users.filter(regex="UserID|Gender|Age|JobID")
    users_orig = users.values

    # 改变User数据中性别和年龄
    gender_map = {"F": 0, "M": 1}
    users["Gender"] = users["Gender"].map(gender_map)
    age_map = {value: index for index, value in enumerate(set(users["Age"]))}
    users["Age"] = users["Age"].map(age_map)

    # 读取Movie数据集
    movies_title = ["MovieID", "Title", "Genres"]
    movies = pd.read_csv(r"./movies.dat", sep="::", header=None, names=movies_title, engine="python")
    movies_orig = movies.values

    # 将Title中的年份去掉
    pattern = re.compile(r"^(.*)\((\d+)\)$")
    title_map = {val: pattern.match(val).group(1).strip() for ii, val in enumerate(set(movies["Title"]))}
    movies["Title"] = movies["Title"].map(title_map)

    # 电影类型转数字字典
    genres_set = set()
    for val in movies["Genres"].str.split("|"):
        genres_set.update(val)

    genres_set.add("<PAD>")
    genres2int = {val: ii for ii, val in enumerate(genres_set)}

    # 将电影类型转成等长数字列表，长度是18
    genres_map = {val: [genres2int.get(row) for row in val.split("|")] for ii, val in enumerate(set(movies["Genres"]))}
    for key in genres_map:
        for cnt in range(max(genres2int.values()) - len(genres_map[key])):
            genres_map[key].insert(len(genres_map.get(key)) + cnt, genres2int.get("<PAD>"))

    movies["Genres"] = movies["Genres"].map(genres_map)

    # 电影Title转数字字典
    title_set = set()
    for val in movies["Title"].str.split():
        title_set.update(val)

    title_set.add("<PAD>")
    title2int = {val: ii for ii, val in enumerate(title_set)}

    # 将电影Title转成等长数字列表，长度是15
    title_count = 15
    title_map = {val: [title2int.get(row) for row in val.split()] for ii, val in enumerate(set(movies["Title"]))}

    for key in title_map:
        for cnt in range(title_count - len(title_map[key])):
            title_map[key].insert(len(title_map[key]) + cnt, title2int.get("<PAD>"))

    movies["Title"] = movies["Title"].map(title_map)

    # 读取评分数据集
    ratings_title = {"UserID", "MovieID", "ratings", "timestamps"}
    ratings = pd.read_csv(r"./ratings.dat", sep='::', header=None, names=ratings_title, engine="python")
    ratings = ratings.filter(regex="UserID|MovieID|ratings")

    # 合并三个表
    data = pd.merge(pd.merge(ratings, users), movies)

    # 将数据分成X和y两张表
    target_fields = ['ratings']
    features_pd, targets_pd = data.drop(target_fields, axis=1), data[target_fields]
    features = features_pd.values
    targets_values = targets_pd.values

    return title_count, title_set, genres2int, features, targets_values, ratings, users, movies, data, movies_orig, users_orig


"""加载数据并保存到本地"""
title_count, title_set, genres2int, features, targets_values, ratings, users, movies, data, movies_orig, users_orig = load_data()
pickle.dump((title_count, title_set, genres2int, features, targets_values, ratings, users, movies, data, movies_orig,
             users_orig), open("preprocessors.p", "wb"))

"""从本地读取数据"""
title_count, title_set, genres2int, features, targets_values, ratings, users, movies, data, movies_orig, users_orig = pickle.load(
    open('preprocessors.p', mode='rb'))


def save_params(params):
    """
    Save parameters to file
    """
    pickle.dump(params, open('params.p', 'wb'))


def load_params():
    """
    Load parameters from file
    """
    return pickle.load(open('params.p', mode='rb'))


# 嵌入矩阵的维度
embed_dim = 32
# 用户ID的个数
uid_max = max(features.take(0, 1)) + 1
# 性别个数
gender_max = max(features.take(2, 1)) + 1
# 年龄类别个数
age_max = max(features.take(3, 1)) + 1
# 职业个数
job_max = max(features.take(4, 1)) + 1
# 电影ID个数
movie_id_max = max(features.take(1, 1)) + 1
# 电影类型个数
movie_categories_max = max(genres2int.values()) + 1
# 电影名单词个数
movie_title_max = len(title_set)

# 对电影类型嵌入向量做加和操作的标志，使用mean做平均
combiner = "mean"

# 电影名长度
sentences_size = title_count
# 文本卷积滑动窗口，分别滑动2, 3, 4, 5个单词
window_sizes = {2, 3, 4, 5}
# 文本卷积核数量
filter_num = 8

# 电影ID转下标的字典，数据集中电影ID跟下标不一致，比如第5行的数据电影ID不一定是5
movieid2idx = {val[0]: i for i, val in enumerate(movies.values)}

"""神经网络超参数"""
# Number of Epochs
num_epochs = 6
# Batch Size
batch_size = 256

dropout_keep = 0.5
# Learning Rate
learning_rate = 0.0001
# Show stats for every n number of batches
show_every_n_batches = 20

save_dir = r"./save"


def get_inputs():
    """定义输入的占位符"""
    uid = tf.placeholder(tf.int32, [None, 1], name="uid")
    user_gender = tf.placeholder(tf.int32, [None, 1], name="user_gender")
    user_age = tf.placeholder(tf.int32, [None, 1], name="user_age")
    user_job = tf.placeholder(tf.int32, [None, 1], name="user_job")

    movie_id = tf.placeholder(tf.int32, [None, 1], name="movie_id")
    movie_categories = tf.placeholder(tf.int32, [None, 18], name="movie_categories")
    movie_titles = tf.placeholder(tf.int32, [None, 15], name="movie_titles")
    targets = tf.placeholder(tf.int32, [None, 1], name="targets")
    LearningRate = tf.placeholder(tf.float32, name="LearningRate")
    dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

    return uid, user_gender, user_age, user_job, movie_id, movie_categories, movie_titles, targets, LearningRate, dropout_keep_prob


"""#####构建神经网络#####"""


def get_user_embedding(uid, user_gender, user_age, user_job):
    """
    定义User的Embedding矩阵
    :param uid:
    :param user_gender:
    :param user_age:
    :param user_job:
    :return:
    """
    with tf.name_scope("user_embedding"):
        uid_embed_matrix = tf.Variable(tf.random_uniform([uid_max, embed_dim], -1, 1), name="uid_embed_matrix")
        uid_embed_layer = tf.nn.embedding_lookup(uid_embed_matrix, uid, name="uid_embed_layer")

        gender_embed_matrix = tf.Variable(tf.random_uniform([gender_max, embed_dim // 2], -1, 1),
                                          name="gender_embed_matrix")
        gender_embed_layer = tf.nn.embedding_lookup(gender_embed_matrix, user_gender, name="gender_embed_layer")

        age_embed_matrix = tf.Variable(tf.random_uniform([age_max, embed_dim // 2], -1, 1), name="age_embed_matrix")
        age_embed_layer = tf.nn.embedding_lookup(age_embed_matrix, user_age, name="age_embed_layer")

        job_embed_matrix = tf.Variable(tf.random_uniform([job_max, embed_dim // 2], -1, 1), name="job_embed_matrix")
        job_embed_layer = tf.nn.embedding_lookup(job_embed_matrix, user_job, name="job_embed_layer")

    return uid_embed_layer, gender_embed_layer, age_embed_layer, job_embed_layer


def get_user_feature_layer(uid_embed_layer, gender_embed_layer, age_embed_layer, job_embed_layer):
    """
    将User的嵌入矩阵一起全连接生成User的特征
    :param uid_embed_layer:
    :param gender_embed_layer:
    :param age_embed_layer:
    :param job_embed_layer:
    :return:
    """
    with tf.name_scope("user_fc"):
        # 第一层全连接
        uid_fc_layer = tf.layers.dense(uid_embed_layer, embed_dim, name="uid_fc_layer", activation=tf.nn.relu)
        gender_fc_layer = tf.layers.dense(gender_embed_layer, embed_dim, name="gender_fc_layer", activation=tf.nn.relu)
        age_fc_layer = tf.layers.dense(age_embed_layer, embed_dim, name="age_fc_layer", activation=tf.nn.relu)
        job_fc_layer = tf.layers.dense(job_embed_layer, embed_dim, name="job_fc_layer", activation=tf.nn.relu)

        # 第二层全连接
        user_combine_layer = tf.concat([uid_fc_layer, gender_fc_layer, age_fc_layer, job_fc_layer], 2)
        # user_combine_layer = tf.contrib.layers.fully_connected(user_combine_layer, 200, tf.tanh)
        user_combine_layer = tf.layers.dense(user_combine_layer, 200, activation=tf.tanh)

        user_combine_layer_flat = tf.layers.flatten(user_combine_layer)
    return user_combine_layer, user_combine_layer_flat


def get_movie_id_embed_layer(movie_id):
    """
    定义Movie ID的embedding矩阵
    :param movie_id:
    :return:
    """
    with tf.name_scope("movie_embedding"):
        movie_id_embed_matrix = tf.Variable(tf.random_uniform([movie_id_max, embed_dim], -1, 1),
                                            name="movie_id_embed_matrix")
        movie_id_embed_layer = tf.nn.embedding_lookup(movie_id_embed_matrix, movie_id, name="movie_id_embed_layer")
    return movie_id_embed_layer


def get_movie_categories_layers(movie_categories):
    with tf.name_scope("movie_categories_layers"):
        movie_categories_embed_matrix = tf.Variable(tf.random_uniform([movie_categories_max, embed_dim], -1, 1),
                                                    name="movie_categories_embed_matrix")
        movie_categories_embed_layer = tf.nn.embedding_lookup(movie_categories_embed_matrix, movie_categories,
                                                              name="movie_categories_embed_layer")
        if combiner == "mean":
            movie_categories_embed_layer = tf.reduce_mean(movie_categories_embed_layer, axis=1, keepdims=True)
    return movie_categories_embed_layer


"""Movie Title的文本卷积网络实现"""


def get_movie_cnn_layer(movie_titles):
    """
    从嵌入矩阵中得到电影名对应的各个单词的嵌入向量
    :param movie_titles:
    :return:
    """
    with tf.name_scope("movie_embedding"):
        movie_title_embed_matrix = tf.Variable(tf.random_uniform([movie_title_max, embed_dim], -1, 1),
                                               name="movie_title_embed_matrix")
        movie_title_embed_layer = tf.nn.embedding_lookup(movie_title_embed_matrix, movie_titles,
                                                         name="movie_title_embed_layer")
        movie_title_embed_layer_expand = tf.expand_dims(movie_title_embed_layer, -1)

    # 对文本嵌入层使用不同尺寸的卷积核做卷积和最大池化
    pool_layer_lst = []
    for window_size in window_sizes:
        with tf.name_scope("movie_txt_conv_maxpool_{}".format(window_size)):
            filter_weights = tf.Variable(tf.truncated_normal([window_size, embed_dim, 1, filter_num], stddev=0.1),
                                         name="filter_weights")
            filter_bias = tf.Variable(tf.constant(0.1, shape=[filter_num]), name="filter_bias")

            conv_layer = tf.nn.conv2d(movie_title_embed_layer_expand, filter_weights, [1, 1, 1, 1], padding="VALID",
                                      name="conv_layer")
            relu_layer = tf.nn.relu(tf.nn.bias_add(conv_layer, filter_bias), name="relu_layer")

            maxpool_layer = tf.nn.max_pool(relu_layer, [1, sentences_size - window_size + 1, 1, 1], [1, 1, 1, 1],
                                           padding="VALID", name="maxpool_layer")
            pool_layer_lst.append(maxpool_layer)

    # Dropout层
    with tf.name_scope("pool_dropout"):
        pool_layer = tf.concat(pool_layer_lst, 3, name="pool_layer")
        max_num = len(window_sizes) * filter_num
        pool_layer_flat = tf.reshape(pool_layer, [-1, 1, max_num], name="pool_layer_flat")

        dropout_layer = tf.nn.dropout(pool_layer_flat, dropout_keep_prob, name="dropout_layer")
    return pool_layer_flat, dropout_layer


def get_movie_feature_layer(movie_id_embed_layer, movie_categories_embed_layer, dropout_layer):
    """
    将Movie的各个层一起做全连接
    :param movie_id_embed_layer:
    :param movie_categories_embed_layer:
    :param dropout_layer:
    :return:
    """
    with tf.name_scope("movie_fc"):
        # 第一层全连接
        movie_id_fc_layer = tf.layers.dense(movie_id_embed_layer, embed_dim, name="movie_id_fc_layer",
                                            activation=tf.nn.relu)
        movie_categories_fc_layer = tf.layers.dense(movie_categories_embed_layer, embed_dim,
                                                    name="movie_categories_fc_layer", activation=tf.nn.relu)

        # 第二层全连接
        movie_combine_layer = tf.concat([movie_id_fc_layer, movie_categories_fc_layer, dropout_layer], 2)  # (?, 1, 96)
        movie_combine_layer = tf.layers.dense(movie_combine_layer, 200, activation=tf.tanh)
        # movie_combine_layer_flat = tf.reshape(movie_combine_layer, [-1, 200])
        movie_combine_layer_flat = tf.layers.flatten(movie_combine_layer)
    return movie_combine_layer, movie_combine_layer_flat


"""构建计算图"""
tf.reset_default_graph()
train_graph = tf.Graph()
with train_graph.as_default():
    # 获取输入占位符
    uid, user_gender, user_age, user_job, movie_id, movie_categories, movie_titles, targets, lr, dropout_keep_prob = get_inputs()
    # 获取User的4个嵌入向量
    uid_embed_layer, gender_embed_layer, age_embed_layer, job_embed_layer = get_user_embedding(uid, user_gender,
                                                                                               user_age, user_job)
    # 得到用户特征
    user_combine_layer, user_combine_layer_flat = get_user_feature_layer(uid_embed_layer, gender_embed_layer,
                                                                         age_embed_layer, job_embed_layer)
    # 获取电影ID的嵌入向量
    movie_id_embed_layer = get_movie_id_embed_layer(movie_id)
    # 获取电影类型的嵌入向量
    movie_categories_embed_layer = get_movie_categories_layers(movie_categories)
    # 获取电影名的特征向量
    pool_layer_flat, dropout_layer = get_movie_cnn_layer(movie_titles)
    # 得到电影特征
    movie_combine_layer, movie_combine_layer_flat = get_movie_feature_layer(movie_id_embed_layer,
                                                                            movie_categories_embed_layer,
                                                                            dropout_layer)
    # 计算出评分，要注意两个不同的方案，inference的名字（name值）是不一样的，后面做推荐时要根据name取得tensor
    with tf.name_scope("inference"):
        # 将用户特征和电影特征作为输入，经过全连接，输出一个值的方案
        # inference_layer = tf.concat([user_combine_layer_flat, movie_combine_layer_flat], 1)  #(?, 200)
        # inference = tf.layers.dense(inference_layer, 1,
        #                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
        #                             kernel_regularizer=tf.nn.l2_loss, name="inference")
        # 简单的将用户特征和电影特征做矩阵乘法得到一个预测评分
        # inference = tf.matmul(user_combine_layer_flat, tf.transpose(movie_combine_layer_flat))
        inference = tf.reduce_sum(user_combine_layer_flat * movie_combine_layer_flat, axis=1)
        inference = tf.expand_dims(inference, axis=1)

    with tf.name_scope("loss"):
        # MSE损失，将计算值回归到评分
        cost = tf.losses.mean_squared_error(targets, inference)
        loss = tf.reduce_mean(cost)
    # 优化损失
    global_step = tf.Variable(0, name="global_step", trainable=False)
    optimizer = tf.train.AdamOptimizer(lr)
    gradients = optimizer.compute_gradients(loss)  # cost
    train_op = optimizer.apply_gradients(gradients, global_step=global_step)

print(inference)


def get_batches(Xs, ys, batch_size):
    """
    取得batch
    :param Xs:
    :param ys:
    :param batch_size:
    :return:
    """
    for start in range(0, len(Xs), batch_size):
        end = min(start + batch_size, len(Xs))
        yield Xs[start:end], ys[start:end]


"""训练网络"""
losses = {'train': [], 'test': []}
with tf.Session(graph=train_graph) as sess:
    timestamp = str(int(time.time()))

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    for epoch_i in range(num_epochs):
        # 将数据集分成训练集和测试集，随机种子不固定
        train_X, test_X, train_y, test_y = train_test_split(features,
                                                            targets_values,
                                                            test_size=0.2,
                                                            random_state=0)
        train_batches = get_batches(train_X, train_y, batch_size)
        test_batches = get_batches(test_X, test_y, batch_size)
        # 训练的迭代，保存训练损失
        for batch_i in range(len(train_X) // batch_size):
            x, y = next(train_batches)

            categories = np.zeros([batch_size, 18])
            for i in range(batch_size):
                categories[i] = x.take(6, 1)[i]
            titles = np.zeros([batch_size, sentences_size])
            for i in range(batch_size):
                titles[i] = x.take(5, 1)[i]

            feed = {
                uid: np.reshape(x.take(0, 1), [batch_size, 1]),
                user_gender: np.reshape(x.take(2, 1), [batch_size, 1]),
                user_age: np.reshape(x.take(3, 1), [batch_size, 1]),
                user_job: np.reshape(x.take(4, 1), [batch_size, 1]),
                movie_id: np.reshape(x.take(1, 1), [batch_size, 1]),
                movie_categories: categories,  # x.take(6,1)
                movie_titles: titles,  # x.take(5,1)
                targets: np.reshape(y, [batch_size, 1]),
                dropout_keep_prob: dropout_keep,  # dropout_keep
                lr: learning_rate}

            step, train_loss, _ = sess.run([global_step, loss, train_op], feed)  # cost
            losses['train'].append(train_loss)

            # Show every <show_every_n_batches> batches
            if (epoch_i * (len(train_X) // batch_size) + batch_i) % show_every_n_batches == 0:
                time_str = datetime.datetime.now().isoformat()
                print('{}: Epoch {:>3} Batch {:>4}/{}   train_loss = {:.3f}'.format(
                    time_str,
                    epoch_i,
                    batch_i,
                    (len(train_X) // batch_size),
                    train_loss))

        # 使用测试数据的迭代
        for batch_i in range(len(test_X) // batch_size):
            x, y = next(test_batches)

            categories = np.zeros([batch_size, 18])
            for i in range(batch_size):
                categories[i] = x.take(6, 1)[i]

            titles = np.zeros([batch_size, sentences_size])
            for i in range(batch_size):
                titles[i] = x.take(5, 1)[i]

            feed = {
                uid: np.reshape(x.take(0, 1), [batch_size, 1]),
                user_gender: np.reshape(x.take(2, 1), [batch_size, 1]),
                user_age: np.reshape(x.take(3, 1), [batch_size, 1]),
                user_job: np.reshape(x.take(4, 1), [batch_size, 1]),
                movie_id: np.reshape(x.take(1, 1), [batch_size, 1]),
                movie_categories: categories,  # x.take(6,1)
                movie_titles: titles,  # x.take(5,1)
                targets: np.reshape(y, [batch_size, 1]),
                dropout_keep_prob: 1,
                lr: learning_rate}

            step, test_loss = sess.run([global_step, loss], feed)  # cost

            # 保存测试损失
            losses['test'].append(test_loss)

            time_str = datetime.datetime.now().isoformat()
            if (epoch_i * (len(test_X) // batch_size) + batch_i) % show_every_n_batches == 0:
                print('{}: Epoch {:>3} Batch {:>4}/{}   test_loss = {:.3f}'.format(
                    time_str,
                    epoch_i,
                    batch_i,
                    (len(test_X) // batch_size),
                    test_loss))

    # Save Model
    saver.save(sess, save_dir)  # , global_step=epoch_i
    print('Model Trained and Saved')

save_params(save_dir)
load_dir = load_params()


def get_tensors(loaded_graph):
    """
    获取tensors，后面的推荐功能要用到
    :param loaded_graph:
    :return:
    """
    uid = loaded_graph.get_tensor_by_name("uid:0")
    user_gender = loaded_graph.get_tensor_by_name("user_gender:0")
    user_age = loaded_graph.get_tensor_by_name("user_age:0")
    user_job = loaded_graph.get_tensor_by_name("user_job:0")
    movie_id = loaded_graph.get_tensor_by_name("movie_id:0")
    movie_categories = loaded_graph.get_tensor_by_name("movie_categories:0")
    movie_titles = loaded_graph.get_tensor_by_name("movie_titles:0")
    targets = loaded_graph.get_tensor_by_name("targets:0")
    dropout_keep_prob = loaded_graph.get_tensor_by_name("dropout_keep_prob:0")
    lr = loaded_graph.get_tensor_by_name("LearningRate:0")
    # 两种不同计算预测评分的方案使用不同的name获取tensor inference
    #     inference = loaded_graph.get_tensor_by_name("inference/inference/BiasAdd:0")
    inference = loaded_graph.get_tensor_by_name("inference/ExpandDims:0")
    movie_combine_layer_flat = loaded_graph.get_tensor_by_name("movie_fc/flatten/Reshape:0")
    user_combine_layer_flat = loaded_graph.get_tensor_by_name("user_fc/flatten/Reshape:0")
    return uid, user_gender, user_age, user_job, movie_id, movie_categories, movie_titles, targets, lr, dropout_keep_prob, inference, movie_combine_layer_flat, user_combine_layer_flat


def rating_movie(user_id_val, movie_id_val):
    """
    指定用户和电影进行评分
    :param user_id_val:
    :param movie_id_val:
    :return:
    """
    loaded_graph = tf.Graph()  #
    with tf.Session(graph=loaded_graph) as sess:  #
        # Load saved model
        loader = tf.train.import_meta_graph(load_dir + '.meta')
        loader.restore(sess, load_dir)

        # Get Tensors from loaded model
        uid, user_gender, user_age, user_job, movie_id, movie_categories, movie_titles, targets, lr, dropout_keep_prob, inference, _, __ = get_tensors(
            loaded_graph)  # loaded_graph

        categories = np.zeros([1, 18])
        categories[0] = movies.values[movieid2idx[movie_id_val]][2]

        titles = np.zeros([1, sentences_size])
        titles[0] = movies.values[movieid2idx[movie_id_val]][1]

        feed = {
            uid: np.reshape(users.values[user_id_val - 1][0], [1, 1]),
            user_gender: np.reshape(users.values[user_id_val - 1][1], [1, 1]),
            user_age: np.reshape(users.values[user_id_val - 1][2], [1, 1]),
            user_job: np.reshape(users.values[user_id_val - 1][3], [1, 1]),
            movie_id: np.reshape(movies.values[movieid2idx[movie_id_val]][0], [1, 1]),
            movie_categories: categories,  # x.take(6,1)
            movie_titles: titles,  # x.take(5,1)
            dropout_keep_prob: 1}

        # Get Prediction
        inference_val = sess.run([inference], feed)
        return inference_val


print(rating_movie(234, 1401))

"""生成Movie特征矩阵将训练好的电影特征组合成电影特征矩阵并保存到本地"""
loaded_graph = tf.Graph()  #
movie_matrics = []
with tf.Session(graph=loaded_graph) as sess:  #
    # Load saved model
    loader = tf.train.import_meta_graph(load_dir + '.meta')
    loader.restore(sess, load_dir)

    # Get Tensors from loaded model
    uid, user_gender, user_age, user_job, movie_id, movie_categories, movie_titles, targets, lr, dropout_keep_prob, _, movie_combine_layer_flat, __ = get_tensors(
        loaded_graph)  # loaded_graph

    for item in movies.values:
        categories = np.zeros([1, 18])
        categories[0] = item.take(2)

        titles = np.zeros([1, sentences_size])
        titles[0] = item.take(1)

        feed = {
            movie_id: np.reshape(item.take(0), [1, 1]),
            movie_categories: categories,  # x.take(6,1)
            movie_titles: titles,  # x.take(5,1)
            dropout_keep_prob: 1}

        movie_combine_layer_flat_val = sess.run([movie_combine_layer_flat], feed)
        movie_matrics.append(movie_combine_layer_flat_val)

pickle.dump((np.array(movie_matrics).reshape(-1, 200)), open('movie_matrics.p', 'wb'))
movie_matrics = pickle.load(open('movie_matrics.p', mode='rb'))

"""生成User特征矩阵将训练好的用户特征组合成用户特征矩阵并保存到本地"""
loaded_graph = tf.Graph()  #
users_matrics = []
with tf.Session(graph=loaded_graph) as sess:  #
    # Load saved model
    loader = tf.train.import_meta_graph(load_dir + '.meta')
    loader.restore(sess, load_dir)

    # Get Tensors from loaded model
    uid, user_gender, user_age, user_job, movie_id, movie_categories, movie_titles, targets, lr, dropout_keep_prob, _, __, user_combine_layer_flat = get_tensors(
        loaded_graph)  # loaded_graph

    for item in users.values:
        feed = {
            uid: np.reshape(item.take(0), [1, 1]),
            user_gender: np.reshape(item.take(1), [1, 1]),
            user_age: np.reshape(item.take(2), [1, 1]),
            user_job: np.reshape(item.take(3), [1, 1]),
            dropout_keep_prob: 1}

        user_combine_layer_flat_val = sess.run([user_combine_layer_flat], feed)
        users_matrics.append(user_combine_layer_flat_val)

pickle.dump((np.array(users_matrics).reshape(-1, 200)), open('users_matrics.p', 'wb'))
users_matrics = pickle.load(open('users_matrics.p', mode='rb'))

"""开始推荐电影使用生产的用户特征矩阵和电影特征矩阵做电影推荐"""
"""
1、推荐同类型的电影
思路是计算当前看的电影特征向量与整个电影特征矩阵的余弦相似度，取相似度最大的top_k个，这里加了些随机选择在里面，保证每次的推荐稍稍有些不同。
"""


def recommend_same_type_movie(movie_id_val, top_k=20):
    loaded_graph = tf.Graph()  #
    with tf.Session(graph=loaded_graph) as sess:  #
        # Load saved model
        loader = tf.train.import_meta_graph(load_dir + '.meta')
        loader.restore(sess, load_dir)

        norm_movie_matrics = tf.sqrt(tf.reduce_sum(tf.square(movie_matrics), 1, keepdims=True))
        normalized_movie_matrics = movie_matrics / norm_movie_matrics

        # 推荐同类型的电影
        probs_embeddings = (movie_matrics[movieid2idx[movie_id_val]]).reshape([1, 200])
        probs_similarity = tf.matmul(probs_embeddings, tf.transpose(normalized_movie_matrics))
        sim = (probs_similarity.eval())
        #     results = (-sim[0]).argsort()[0:top_k]
        #     print(results)

        print("您看的电影是：{}".format(movies_orig[movieid2idx[movie_id_val]]))
        print("以下是给您的推荐：")
        p = np.squeeze(sim)
        p[np.argsort(p)[:-top_k]] = 0
        p = p / np.sum(p)
        results = set()
        while len(results) != 5:
            c = np.random.choice(3883, 1, p=p)[0]
            results.add(c)
        for val in results:
            print(val)
            print(movies_orig[val])
        return results


print(recommend_same_type_movie(1401, 20))

"""
2、推荐您喜欢的电影
思路是使用用户特征向量与电影特征矩阵计算所有电影的评分，取评分最高的top_k个，同样加了些随机选择部分。
"""


def recommend_your_favorite_movie(user_id_val, top_k=10):
    loaded_graph = tf.Graph()  #
    with tf.Session(graph=loaded_graph) as sess:  #
        # Load saved model
        loader = tf.train.import_meta_graph(load_dir + '.meta')
        loader.restore(sess, load_dir)

        # 推荐您喜欢的电影
        probs_embeddings = (users_matrics[user_id_val - 1]).reshape([1, 200])
        probs_similarity = tf.matmul(probs_embeddings, tf.transpose(movie_matrics))
        sim = (probs_similarity.eval())
        #     print(sim.shape)
        #     results = (-sim[0]).argsort()[0:top_k]
        #     print(results)

        #     sim_norm = probs_norm_similarity.eval()
        #     print((-sim_norm[0]).argsort()[0:top_k])

        print("以下是给您的推荐：")
        p = np.squeeze(sim)
        p[np.argsort(p)[:-top_k]] = 0
        p = p / np.sum(p)
        results = set()
        while len(results) != 5:
            c = np.random.choice(3883, 1, p=p)[0]
            results.add(c)
        for val in results:
            print(val)
            print(movies_orig[val])
        return results


print(recommend_your_favorite_movie(234, 10))

"""
3、看过这个电影的人还看了（喜欢）哪些电影
   - 首先选出喜欢某个电影的top_k个人，得到这几个人的用户特征向量。
   - 然后计算这几个人对所有电影的评分
   - 选择每个人评分最高的电影作为推荐
   - 同样加入了随机选择
"""


def recommend_other_favorite_movie(movie_id_val, top_k=20):
    loaded_graph = tf.Graph()  #
    with tf.Session(graph=loaded_graph) as sess:  #
        # Load saved model
        loader = tf.train.import_meta_graph(load_dir + '.meta')
        loader.restore(sess, load_dir)

        probs_movie_embeddings = (movie_matrics[movieid2idx[movie_id_val]]).reshape([1, 200])
        probs_user_favorite_similarity = tf.matmul(probs_movie_embeddings, tf.transpose(users_matrics))
        favorite_user_id = np.argsort(probs_user_favorite_similarity.eval())[0][-top_k:]
        #     print(normalized_users_matrics.eval().shape)
        #     print(probs_user_favorite_similarity.eval()[0][favorite_user_id])
        #     print(favorite_user_id.shape)

        print("您看的电影是：{}".format(movies_orig[movieid2idx[movie_id_val]]))

        print("喜欢看这个电影的人是：{}".format(users_orig[favorite_user_id - 1]))
        probs_users_embeddings = (users_matrics[favorite_user_id - 1]).reshape([-1, 200])
        probs_similarity = tf.matmul(probs_users_embeddings, tf.transpose(movie_matrics))
        sim = (probs_similarity.eval())
        #     results = (-sim[0]).argsort()[0:top_k]
        #     print(results)

        #     print(sim.shape)
        #     print(np.argmax(sim, 1))
        p = np.argmax(sim, 1)
        print("喜欢看这个电影的人还喜欢看：")

        results = set()
        while len(results) != 5:
            c = p[random.randrange(top_k)]
            results.add(c)
        for val in results:
            print(val)
            print(movies_orig[val])
        return results


print(recommend_other_favorite_movie(1401, 20))
