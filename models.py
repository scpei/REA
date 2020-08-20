from layers import *
from metrics import *
from inits import *
from utils import *
import heapq
from sklearn.metrics import recall_score, precision_score, auc, confusion_matrix

flags = tf.app.flags
FLAGS = flags.FLAGS

class TopKHeap(object):
    def __init__(self, k):
        self.k = k
        self.data = []

    def push(self, elem):
        if len(self.data) < self.k:
            heapq.heappush(self.data, elem)
        else:
            topk_small = self.data[0]
            if elem > topk_small:
                heapq.heapreplace(self.data, elem)

    def topk(self):
        return [x for x in reversed([heapq.heappop(self.data) for x in range(len(self.data))])]


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None

        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.outputs = self.activations[-1]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)


class MLP(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(MLP, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self):
        self.layers.append(Dense(input_dim=self.input_dim,
                                 output_dim=FLAGS.hidden1,
                                 placeholders=self.placeholders,
                                 act=tf.nn.relu,
                                 dropout=True,
                                 sparse_inputs=True,
                                 logging=self.logging))

        self.layers.append(Dense(input_dim=FLAGS.hidden1,
                                 output_dim=self.output_dim,
                                 placeholders=self.placeholders,
                                 act=lambda x: x,
                                 dropout=True,
                                 logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)


class GCN(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(GCN, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self):

        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=FLAGS.hidden1,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=True,
                                            sparse_inputs=True,
                                            logging=self.logging))

        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            act=lambda x: x,
                                            dropout=True,
                                            logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)


class GCN_Align(Model):
    def __init__(self, placeholders, input_dim, output_dim, ILL, sparse_inputs=False, featureless=True, **kwargs):
        super(GCN_Align, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.placeholders = placeholders
        self.ILL = ILL
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless

        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.learning_rate)
        self.build()

    def _loss(self):
        self.loss += align_loss(self.outputs, self.ILL, FLAGS.gamma, FLAGS.k)

    def _accuracy(self):
        pass

    def _build(self):

        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            act=tf.nn.leaky_relu,
                                            dropout=False,
                                            featureless=self.featureless,
                                            sparse_inputs=self.sparse_inputs,
                                            transform=False,
                                            init=trunc_normal,
                                            logging=self.logging))

        self.layers.append(GraphConvolution(input_dim=self.output_dim,
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            act=tf.nn.leaky_relu,
                                            dropout=False,
                                            transform=False,
                                            logging=self.logging))

class Gan(object):
    def __init__(self, n_input, n_hidden_1, n_hidden_2, model, batch_size, mode):
        self.model = model
        self.n_input = n_input
        self.n_hidden_1 = n_hidden_1
        self.n_hidden_2 = n_hidden_2
        self.batch_size = batch_size
        self.opt = None
        self.cost = None
        self.mode = mode
        self.build()

    def get_embedding(self, outputs, sample):
        h = tf.nn.embedding_lookup(outputs, sample[:, 0])
        t = tf.nn.embedding_lookup(outputs, sample[:, 1])
        return h, t

    def forward(self, in_sample):
        layer_1 = tf.nn.relu(tf.add(tf.matmul(in_sample, self.weights['h1']), self.biases['b1']))
        # layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, self.weights['h2']), self.biases['b2']))
        pred = tf.matmul(layer_1, self.weights['out']) + self.biases['out']
        return pred

    def build(self):

        self.weights = {
            'h1': tf.Variable(tf.random_normal([self.n_input, self.n_hidden_1])),
            # 'h2': tf.Variable(tf.random_normal([self.n_hidden_1, self.n_hidden_2])),
            'out': tf.Variable(tf.random_normal([self.n_hidden_1, 1]))
        }
        self.biases = {
            'b1': tf.Variable(tf.random_normal([self.n_hidden_1])),
            # 'b2': tf.Variable(tf.random_normal([self.n_hidden_2])),
            'out': tf.Variable(tf.random_normal([1]))
        }

        self.sample_pos = tf.placeholder(shape=[None, None, FLAGS.se_dim], dtype=tf.float32)
        self.sample_neg = tf.placeholder(shape=[None, None, FLAGS.se_dim], dtype=tf.float32)
        self.idx = tf.placeholder(shape=[None, 2], dtype=tf.int32)
        self.reward = tf.placeholder(shape=[None, 1], dtype=tf.float32)
        self.n_sample = tf.placeholder(dtype=tf.int32)
        self.nn = tf.placeholder(dtype=tf.int32)
        scores = self.forward(self.sample_pos - self.sample_neg)
        self.scores = tf.reshape(scores, (-1, FLAGS.k*self.nn))
        probs = tf.nn.softmax(self.scores, dim=1)
        self.sample = tf.multinomial(probs, self.n_sample)
        self.log_probs = tf.nn.log_softmax(self.scores, dim=1)

        self.cost = - tf.reduce_mean(tf.multiply(tf.transpose(self.reward), tf.gather_nd(self.log_probs, self.idx)))

        self.opt = tf.train.AdamOptimizer(learning_rate=0.01).minimize(self.cost)

    def generate(self, neg_head, neg_tail, sess, row_idx, new_neg, n_sample=1, nn=1):
        sample, scores = sess.run([self.sample, self.scores],
                                    feed_dict={self.sample_pos: neg_head,
                                               self.sample_neg: neg_tail,
                                               self.n_sample: n_sample,
                                               self.nn: nn})
        neg_list = []
        idx_list = []
        for i in range(sample.shape[1]):
            idx = np.transpose(row_idx)[0]
            s = np.transpose(sample[:, i])
            neg = new_neg[idx, s]
            neg_list.append(neg)
            idx_ = np.concatenate((row_idx, np.expand_dims(sample[:, i], axis=1).astype(int)), axis=1)
            idx_list.append(idx_)

        neg_out = np.reshape(np.expand_dims(np.transpose(neg_list), axis=-1), (-1, 1))
        idx_out = np.concatenate(idx_list, axis=0)
        return neg_out, idx_out, scores

class Clf(object):
    def __init__(self, n_input, n_hidden_1, n_hidden_2, model, batch_size, mode):

        self.model = model
        self.n_input = n_input
        self.n_hidden_1 = n_hidden_1
        self.n_hidden_2 = n_hidden_2
        self.batch_size = batch_size
        self.opt = None
        self.cost = None
        self.mode = mode
        self.build()

    def get_embedding(self, outputs, sample):
        h = tf.nn.embedding_lookup(outputs, sample[:, 0])
        t = tf.nn.embedding_lookup(outputs, sample[:, 1])
        return h, t

    def forward(self, in_sample):
        layer_1 = tf.nn.relu(tf.add(tf.matmul(in_sample, self.weights['h1']), self.biases['b1']))
        # layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, self.weights['h2']), self.biases['b2']))
        pred = tf.matmul(layer_1, self.weights['out']) + self.biases['out']
        return pred

    def build(self):

        self.weights = {
            'h1': tf.Variable(tf.random_normal([self.n_input, self.n_hidden_1])),
            # 'h2': tf.Variable(tf.random_normal([self.n_hidden_1, self.n_hidden_2])),
            'out': tf.Variable(tf.random_normal([self.n_hidden_1, 1]))
        }
        self.biases = {
            'b1': tf.Variable(tf.random_normal([self.n_hidden_1])),
            # 'b2': tf.Variable(tf.random_normal([self.n_hidden_2])),
            'out': tf.Variable(tf.random_normal([1]))
        }
        #
        self.model_output = tf.placeholder(shape=[None, FLAGS.se_dim], dtype=tf.float32)
        self.input_sample = tf.placeholder(shape=[None, 2], dtype=tf.int32)
        self.h = tf.nn.embedding_lookup(self.model_output, self.input_sample[:, 0])
        self.t = tf.nn.embedding_lookup(self.model_output, self.input_sample[:, 1])
        #

        self.sample_pos = tf.placeholder(shape=[None, FLAGS.se_dim], dtype=tf.float32)
        self.sample_neg = tf.placeholder(shape=[None, FLAGS.se_dim], dtype=tf.float32)
        self.label = tf.placeholder(shape=[None], dtype=tf.float32)
        self.pos_score = self.forward(self.sample_pos)
        self.neg_score = self.forward(self.sample_neg)
        self.sample = tf.concat([self.pos_score, self.neg_score], axis=0)

        self.sig_score = tf.nn.sigmoid(self.pos_score)
        self.neg_score_probs = tf.nn.sigmoid(self.neg_score)
        ###
        self.out_neg_score_probs = tf.log(1 + tf.exp(self.neg_score))

        self.cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.sample,
                                                                           labels=tf.expand_dims(self.label, axis=1)))
        self.opt = tf.train.AdamOptimizer(learning_rate=0.01).minimize(self.cost, var_list=[self.weights, self.biases])

    def train_classifier_step(self, pos_score, neg_score, target, classifier, sess):
        _, loss, sample_ = sess.run([classifier.opt, classifier.cost, classifier.sample],
                           feed_dict={classifier.sample_pos: pos_score,
                                      classifier.sample_neg: neg_score,
                                      classifier.label: target})

        return loss

    def discriminator_step(self, pos, neg, discriminator, vec_se, sess):
        pos_head, pos_tail = sess.run([discriminator.h, discriminator.t], feed_dict={discriminator.model_output: vec_se,
                                                                                     discriminator.input_sample:
                                                                                         pos.astype(int)})
        neg_head, neg_tail = sess.run([discriminator.h, discriminator.t], feed_dict={discriminator.model_output: vec_se,
                                                                                     discriminator.input_sample:
                                                                                         neg.astype(int)})
        pos_score = np.abs(pos_head - pos_tail)
        neg_score = np.abs(neg_head - neg_tail)
        target = np.concatenate([np.ones(pos_score.shape[0]), np.zeros(neg_score.shape[0])], axis=0)

        loss, pos_, neg_prob, neg_pred = sess.run([discriminator.cost, discriminator.pos_score,
                                                   discriminator.out_neg_score_probs, discriminator.neg_score],
                           feed_dict={discriminator.sample_pos: pos_score,
                                      discriminator.sample_neg: neg_score,
                                      discriminator.label: target})

        return loss, neg_prob

    def train_gan_step(self, neg_head, neg_tail, row_idx, new_neg, new_pos_for_neg, pos,
                       generator, discriminator, sess, avg_reward, vec_se):

        epoch_reward = 0
        epoch_loss = 0

        new_neg, row_sample_idx, score = generator.generate(neg_head, neg_tail, sess, row_idx, new_neg)
        neg = np.concatenate((new_pos_for_neg, new_neg), axis=1)
        loss, rewards = discriminator.discriminator_step(pos, neg, discriminator, vec_se, sess)
        epoch_reward += np.sum(rewards)
        epoch_loss += loss
        rewards = rewards - avg_reward
        _, cost_ = sess.run([generator.opt, generator.cost],
                                    feed_dict={generator.sample_pos: neg_head,
                                               generator.sample_neg: neg_tail,
                                               generator.idx: row_sample_idx,
                                               generator.reward: rewards,
                                               generator.nn: 1})
        return epoch_reward, epoch_loss, cost_

    def find_topK_pair(self, model, feed_dict_se, classifier, train_dataset, clf_dataset, gan_dataset, sess, k = None, k1 = None):

        if k is None:
            k = clf_dataset.len
            k1 = gan_dataset.len

        all_weight = 0
        bottomK_heap = TopKHeap(k1)

        new_set_discrim = []

        vec_se = sess.run(model.outputs, feed_dict=feed_dict_se)
        i = 0
        weight_list = []
        while i < len(train_dataset.train):
            j = min(i + 2000, len(train_dataset.train))
            sample = train_dataset.train[i:j]
            head, tail = sess.run([classifier.h, classifier.t], feed_dict={classifier.model_output: vec_se,
                                                                           classifier.input_sample: sample})
            orig_weight = sess.run([classifier.sig_score], feed_dict={classifier.sample_pos: np.abs(head - tail)})
            weight = (np.asarray(orig_weight) >= 0.01).astype(float)
            weight_ = np.reshape(weight, (weight.shape[1],))
            weight_list.extend(weight_.tolist())
            all_weight += np.sum(weight)
            score = - np.linalg.norm(head - tail, ord=1, axis=1)

            for x, pair in enumerate(train_dataset.train[i:j]):
                if pair not in train_dataset.trust:
                    train_dataset.subsampling_weight[pair] = weight[0][x][0]

                    if weight[0][x][0] > 0.5:
                        new_set_discrim.append(pair)

                bottomK_heap.push((-score[x], pair))
            i = j

        bottomK_list = bottomK_heap.topk()
        clf_dataset.train = train_dataset.trust + new_set_discrim
        clf_dataset.len = len(clf_dataset.train)
        clf_dataset.orgin_train = np.asarray([[item[0], item[1]] for item in clf_dataset.train])
        _, gan_dataset.train = list(zip(*bottomK_list))
        gan_dataset.orgin_train = np.asarray([[item[0], item[1]] for item in gan_dataset.train])






