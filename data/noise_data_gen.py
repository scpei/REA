import numpy as np
from utils import loadfile, get_ent2id
import tensorflow as tf
import random
flags = tf.app.flags
FLAGS = flags.FLAGS


def load_data(dataset_str):
    names = [['ent_ids_1', 'ent_ids_2'], ['training_attrs_1', 'training_attrs_2'], ['triples_1', 'triples_2'], ['ref_ent_ids']]
    for fns in names:
        for i in range(len(fns)):
            fns[i] = 'data/'+dataset_str+'/'+fns[i]
    Es, As, Ts, ill = names
    ill = ill[0]
    ILL = loadfile(ill, 2)
    illL = len(ILL)
    np.random.shuffle(ILL)
    train = np.array(ILL[:illL // 10 * FLAGS.seed])
    test = ILL[illL // 10 * FLAGS.seed:]

    kg1 = get_ent2id([Es[0]])
    kg1_list = list(kg1.values())
    kg2 = get_ent2id([Es[1]])
    kg2_list = list(kg2.values())

    save_train_test_into_txt(np.asarray(test), 'test.txt')
    _, new_train = generate_fake(train, test, kg1_list, kg2_list)
    save_train_test_into_txt(new_train, 'new_train.txt')

def save_train_test_into_txt(z, fname):
    np.savetxt('data/zh_en/' + fname, z, fmt='%d')

def load_train_test_into_model(fname):
    return np.loadtxt('data/zh_en/' + fname, dtype=int)

def generate_fake(train, test, kg1, kg2):
    perc = 0.4
    train = train.tolist()
    replace = random.sample(train, int(len(train)*perc))
    left_train = [item for item in train if item not in replace]
    left_all = left_train + test
    linked_kg1 = [item[0] for item in left_all]
    linked_kg2 = [item[1] for item in left_all]
    new_replace = []
    for item in replace:
        l = list(set(kg2) - set(linked_kg2))
        new_item2 = random.choice(l)
        new_replace.append([item[0], new_item2])
        linked_kg1.append(item[0])
        linked_kg2.append(new_item2)

    return np.asarray(left_train), np.asarray(left_train + new_replace)

load_data("zh_en")