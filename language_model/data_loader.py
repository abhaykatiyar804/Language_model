# import numpy as np
import os
import collections

data_path = '/home/katiyar/PycharmProjects/LSTM_Language model/language_model/data'


def read_words(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return f.read().replace("\n", "<eos>").split()


def build_vocab(filename):
    data = read_words(filename)
    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    words, _ = list(zip(*count_pairs))
    print('-----------------------------------------------------------')
    word_to_id = dict(zip(words, range(len(words))))
    return word_to_id


def file_to_word_ids(filepath, word_to_id):
    data = read_words(filepath)
    return [word_to_id[word] for word in data if word in word_to_id]


def load_data():
    train_path = os.path.join(data_path, 'ptb.train.txt')
    test_path = os.path.join(data_path,'ptb.test.txt')
    valid_path = os.path.join(data_path,'ptb.valid.txt')

    word_to_id = build_vocab(train_path)
    train = file_to_word_ids(train_path, word_to_id)
    test = file_to_word_ids(test_path, word_to_id)
    validation = file_to_word_ids(valid_path, word_to_id)
    reverse_dict = dict(zip(word_to_id.values(), word_to_id.keys()))
    vocabulary = len(word_to_id)

    print('vocabulary length :',vocabulary)
    print('train data',train[:5])
    print('word_to_id',word_to_id)
    print('reverse_dict', ' '.join( reverse_dict[i] for i in train[100:120]))
    return train, test, validation

load_data()
