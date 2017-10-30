import numpy as np
import pandas as pd
import tensorflow as tf
from itertools import product
from tensorflow.contrib.data import Iterator
from sklearn.metrics import classification_report


class Model:
    def __init__(self, input_shape, num_classes, verbose=True):
        tf.reset_default_graph()
        self.input_shape = input_shape
        self.num_classes = num_classes

        self.X = tf.placeholder('float', [None, self.input_shape])
        self.Y = tf.placeholder('float', [None, self.num_classes])
        self.model = self.create_model()

        self.verbose = verbose
        self.losses = []

    def __copy__(self):
        model = Model(self.input_shape, self.num_classes, self.verbose)
        model.set_loss_params(weights=self.loss_weights, beta=self.beta)
        model.set_optimise_params(learning_rate=self.learning_rate, decay=self.decay, momentum=self.momentum,
                                  epsilon=self.epsilon, use_locking=self.use_locking, centered=self.centered)
        return model

    def create_model(self):
        self.weights = {
            'h1': tf.Variable(tf.truncated_normal([self.input_shape, 256])),
            'h2': tf.Variable(tf.truncated_normal([256, 256])),
            'h3': tf.Variable(tf.truncated_normal([256, 256])),
            'out': tf.Variable(tf.truncated_normal([256, self.num_classes]))
        }
        # Initialises the biases
        self.biases = {
            'b1': tf.Variable(tf.truncated_normal([256])),
            'b2': tf.Variable(tf.truncated_normal([256])),
            'b3': tf.Variable(tf.truncated_normal([256])),
            'out': tf.Variable(tf.truncated_normal([self.num_classes]))
        }

        # Initialises the graph acording to an architecture
        layer_1 = tf.add(tf.matmul(self.X, self.weights['h1']), self.biases['b1'])
        layer_2 = tf.add(tf.matmul(layer_1, self.weights['h2']), self.biases['b2'])
        layer_3 = tf.nn.softsign(tf.add(tf.matmul(layer_2, self.weights['h3']), self.biases['b3']))
        return tf.add(tf.matmul(layer_3, self.weights['out']), self.biases['out'])

    def loss(self):
        if not self.loss_weights.shape == np.ones((0, 0)).shape:
            nb_cl = len(self.loss_weights)
            final_mask = tf.zeros_like(self.model[..., 0])
            y_pred_max = tf.reduce_max(self.model, axis=-1)
            y_pred_max = tf.expand_dims(y_pred_max, axis=-1)
            y_pred_max_mat = tf.equal(self.model, y_pred_max)

            for c_p, c_t in product(range(nb_cl), range(nb_cl)):
                w = tf.cast(self.loss_weights[c_t, c_p], 'float32')
                y_p = tf.cast(y_pred_max_mat[..., c_p], 'float32')
                y_t = tf.cast(y_pred_max_mat[..., c_t], 'float32')
                final_mask += w * y_p * y_t

            loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.model, labels=self.Y) * final_mask
        else:
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.model, labels=self.Y)

        loss += self.beta + tf.nn.l2_loss(self.weights['h2']) + self.beta + tf.nn.l2_loss(self.biases['b2'])
        loss += self.beta + tf.nn.l2_loss(self.weights['h3']) + self.beta + tf.nn.l2_loss(self.biases['b3'])
        loss += self.beta + tf.nn.l2_loss(self.weights['out']) + self.beta + tf.nn.l2_loss(self.biases['out'])
        return tf.reduce_mean(loss)

    def set_loss_params(self, weights=np.ones((0, 0)), beta=0.1):
        self.beta = beta
        self.loss_weights = weights

    def optimise(self, loss):
        optimiser = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate, decay=self.decay,
                                              momentum=self.momentum, use_locking=self.use_locking,
                                              centered=self.centered)
        return optimiser.minimize(loss)

    def set_optimise_params(self, learning_rate=0.001, decay=0.9, momentum=0.0, epsilon=1e-10, use_locking=False,
                            centered=False):
        self.learning_rate = learning_rate
        self.decay = decay
        self.momentum = momentum
        self.epsilon = epsilon
        self.use_locking = use_locking
        self.centered = centered

    def train(self, train_data, test_data, val_data, epochs=-1, batch_size=100, intervals=10):
        loss = self.loss()
        optimizer = self.optimise(loss)
        correct_pred = tf.equal(tf.argmax(self.model, 1), tf.argmax(self.Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        train_data.batch(batch_size)
        test_data.batch(batch_size)
        val_data.batch(batch_size)

        init = tf.global_variables_initializer()
        iterator = Iterator.from_structure(train_data.output_types, train_data.output_shapes)
        next_batch = iterator.get_next()

        train_batches_per_epoch = int(np.floor(train_data.data_size / batch_size))
        test_batches_per_epoch = int(np.floor(test_data.data_size / batch_size))
        val_batches_per_epoch = int(np.floor(val_data.data_size / batch_size))

        training_init_op = iterator.make_initializer(train_data)
        testing_init_op = iterator.make_initializer(test_data)
        validation_init_op = iterator.make_initializer(val_data)

        with tf.Session as sess:
            sess.run(init)
            epoch = 0
            while not self.converged(epochs) and epoch != epochs:
                sess.run(training_init_op)
                train_loss = 0
                for step in range(train_batches_per_epoch):
                    img_batch, label_batch = sess.run(next_batch)
                    _, cost = sess.run([optimizer, loss], feed_dict={self.X: img_batch, self.Y: label_batch})
                    train_loss += cost / train_batches_per_epoch
                epoch += 1
                sess.run(validation_init_op)
                val_loss, val_acc = 0, 0
                for step in range(val_batches_per_epoch):
                    img_batch, label_batch = sess.run(next_batch)
                    acc, cost = sess.run([accuracy, loss], feed_dict={self.X: img_batch, self.Y: label_batch})
                    val_loss += cost / val_batches_per_epoch
                    val_acc += cost / val_batches_per_epoch
                self.losses.append(val_loss)
                if self.verbose and epoch % intervals == 0:
                    message = 'Epoch: ' + str(epoch) + ' Training Loss: ' + '{:.4f}'.format(train_loss)
                    message += ' Validation Accuracy: ' + '{:.3f}'.format(val_acc)
                    message += ' Validation Loss: ' + '{:.4f}'.format(val_loss)
                    print(message)
            sess.run(testing_init_op)
            test_acc = 0
            for step in range(test_batches_per_epoch):
                img_batch, label_batch = sess.run(next_batch)
                accuracy = sess.run(accuracy, feed_dict={self.X: img_batch, self.Y: label_batch})
                test_acc += accuracy / test_batches_per_epoch
        return test_acc


    @staticmethod
    def confusion_matrix(predictions, labels):
        y_actu = np.zeros(len(labels))
        for i in range(len(labels)):
            for j in range(len(labels[i])):
                if labels[i][j] == 1.00:
                    y_actu[i] = j
        y_pred = np.zeros(len(predictions))
        for i in range(len(predictions)):
            y_pred[i] = np.argmax(predictions[i])

        p_labels = pd.Series(y_pred)
        t_labels = pd.Series(y_actu)
        df_confusion = pd.crosstab(t_labels, p_labels, rownames=['Actual'], colnames=['Predicted'], margins=True)

        print(df_confusion)
        print(classification_report(y_actu, y_pred, digits=4))

    def converged(self, epochs, min_epochs=50, diff=0.5, converge_len=10):
        if len(self.losses) > min_epochs and epochs == -1:
            losses = self.losses[-converge_len:]

            for loss in losses[: (converge_len - 1)]:
                if abs(losses[-1] - loss) > diff:
                    return False
            return True
        return False
