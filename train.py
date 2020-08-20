from __future__ import division
from __future__ import print_function

from utils import *
from metrics import *
from models import GCN_Align, Clf, Gan
import os

seed = 12306
np.random.seed(seed)
tf.set_random_seed(seed)

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('lang', 'fr_en', 'Dataset string.')  # 'zh_en', 'ja_en', 'fr_en'
flags.DEFINE_float('learning_rate', 25, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 50000, 'Number of epochs to train.')
flags.DEFINE_float('dropout', 0., 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('gamma', 3.0, 'Hyper-parameter for margin based loss.')
flags.DEFINE_integer('k', 5, 'Number of negative samples for each positive seed.')
flags.DEFINE_integer('se_dim', 200, 'Dimension for SE.')
flags.DEFINE_integer('seed', 3, 'Proportion of seeds, 3 means 30%')

dn = True
nm = 1

adj, e, train, test, train_trust, train_noise_inv = load_data(FLAGS.lang)

batch_size = len(train)*2*FLAGS.k

train_dataset = TrainSet(train, FLAGS.k)
train_dataset.init_subsampling(train_trust)
train_dataset.get_trust(train_trust)
train_dataset.get_noise_involved(train_noise_inv)

clf_train = np.asarray(random.sample(train_dataset.trust, len(train_dataset.trust)))
clf_dataset = TrainSet(clf_train, FLAGS.k)
clf_dataset.subsampling_weight = train_dataset.subsampling_weight

GAN_dataset = TrainSet(clf_train, FLAGS.k)
GAN_dataset.subsampling_weight = train_dataset.subsampling_weight

support = [preprocess_adj(adj)]
num_supports = 1
model_func = GCN_Align
k = FLAGS.k

ph_se = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.placeholder(tf.float32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder_with_default(0, shape=())
}

model_se = model_func(ph_se, input_dim=e, output_dim=FLAGS.se_dim, ILL=train_dataset, sparse_inputs=False, featureless=True, logging=True)
classifier = Clf(n_input=FLAGS.se_dim, n_hidden_1=100, n_hidden_2=30, model=model_se, batch_size= batch_size, mode="clf")
generator = Gan(n_input=FLAGS.se_dim, n_hidden_1=100, n_hidden_2=30, model=model_se, batch_size= batch_size, mode="gen")
c = tf.ConfigProto(inter_op_parallelism_threads=3, intra_op_parallelism_threads=3)
c.gpu_options.allow_growth = True
sess = tf.Session(config=c)
sess.run(tf.global_variables_initializer())

cost_val = []

t = len(train)
L = np.ones((t, k)) * (train[:, 0].reshape((t, 1)))
neg_left = L.reshape((t * k,))
L = np.ones((t, k)) * (train[:, 1].reshape((t, 1)))
neg2_right = L.reshape((t * k,))

best_hit1 = 0
best_hit5 = 0
best_mrr = 0
best_epoch = 0

for epoch in range(FLAGS.epochs):

    if epoch % 10 == 0: # for non-adversarial training
        neg2_left = np.random.choice(e, t * k)
        neg_right = np.random.choice(e, t * k)

    feed_dict_se = construct_feed_dict(1.0, support, ph_se)
    feed_dict_se.update({ph_se['dropout']: FLAGS.dropout})
    feed_dict_se.update({'neg_left:0': neg_left, 'neg_right:0': neg_right, 'neg2_left:0': neg2_left,
                         'neg2_right:0': neg2_right, 'subsampling:0': train_dataset.get_subsampling()})
    outs_se = sess.run([model_se.opt_op, model_se.loss], feed_dict=feed_dict_se)

    cost_val.append((outs_se[1]))

    if epoch % 1000 == 0:
        print("Epoch:", '%04d' % (epoch + 1), "SE_train_loss=", "{:.5f}".format(outs_se[1]))

    ## Test every 1000 epoch.
    # if epoch % 1000 == 0 and epoch != 0:
    #     feed_dict_se_test = construct_feed_dict(1.0, support, ph_se)
    #     vec_se = sess.run(model_se.outputs, feed_dict=feed_dict_se_test)
    #     hit1, hit5, mrr = get_hits(vec_se, test)

    if dn == True:
        if epoch % 1000 == 0 and epoch != 0:
            print("Optimizing the Noise Detection module....")
            for epoch_adv in range(5):

                # data for discriminator -- clf_dataset
                L_clf = np.ones((clf_dataset.len, k * nm)) * (clf_dataset.orgin_train[:, 0].reshape((clf_dataset.len, 1)))
                neg_left_clf = L_clf.reshape((clf_dataset.len * k * nm,))
                R_clf = np.ones((clf_dataset.len, k * nm)) * (clf_dataset.orgin_train[:, 1].reshape((clf_dataset.len, 1)))
                neg2_right_clf = R_clf.reshape((clf_dataset.len * k * nm,))
                neg2_left_clf = np.random.choice(e, clf_dataset.len * k * nm)
                neg_right_clf = np.random.choice(e, clf_dataset.len * k * nm)

                # data pre-processing for discrminator
                neg_head, neg_tail, row_idx, new_neg, new_pos_for_neg, pos_score, pos, vec_se = preprocess_data(model_se, feed_dict_se,
                                                                       sess, neg_left_clf, neg_right_clf, neg2_left_clf,
                                                                     neg2_right_clf, classifier)

                # Classifier training.
                new_neg, _, _ = generator.generate(neg_head, neg_tail, sess, row_idx, new_neg)
                neg = np.concatenate((new_pos_for_neg, new_neg), axis=1)
                neg_head, neg_tail = sess.run([classifier.h, classifier.t], feed_dict={classifier.model_output: vec_se,
                                                                                             classifier.input_sample:
                                                                                                 neg.astype(int)})

                neg_score = np.abs(neg_head - neg_tail)
                target = np.concatenate([np.ones(pos_score.shape[0]), np.zeros(neg_score.shape[0])], axis=0)

                for sub_epoch in range(1000):
                    clf_loss = classifier.train_classifier_step(pos_score, neg_score, target, classifier, sess)

                # data for generator -- GAN_dataset.
                L_gan = np.ones((GAN_dataset.len, k*nm)) * (GAN_dataset.orgin_train[:, 0].reshape((GAN_dataset.len, 1)))
                neg_left_gan = L_gan.reshape((GAN_dataset.len * k*nm,))
                R_gan = np.ones((GAN_dataset.len, k*nm)) * (GAN_dataset.orgin_train[:, 1].reshape((GAN_dataset.len, 1)))
                neg2_right_gan = R_gan.reshape((GAN_dataset.len * k*nm,))
                neg2_left_gan = np.random.choice(e, GAN_dataset.len * k*nm)
                neg_right_gan = np.random.choice(e, GAN_dataset.len * k*nm)

                # data pre-processing for generator.
                neg_head, neg_tail, row_idx, new_neg, new_pos_for_neg, pos_score, pos, vec_se = preprocess_data(model_se, feed_dict_se,
                                                                       sess, neg_left_gan, neg_right_gan, neg2_left_gan,
                                                                     neg2_right_gan, classifier)
                # Generator Training.
                avg_reward = 0
                for sub_epoch in range(1000):
                    epoch_reward, epoch_loss, cost = classifier.train_gan_step(neg_head, neg_tail, row_idx, new_neg, new_pos_for_neg,
                                                                                pos, generator, classifier, sess, avg_reward, vec_se)
                    avg_reward = epoch_reward / (t*2)

                classifier.find_topK_pair(model_se, feed_dict_se, classifier, train_dataset, clf_dataset,
                                                       GAN_dataset, sess)

print("Optimization Finished!")
feed_dict_se = construct_feed_dict(1.0, support, ph_se)
vec_se = sess.run(model_se.outputs, feed_dict=feed_dict_se)
print("SE")
get_hits(vec_se, test)
