import numpy as np
import tensorflow as tf
from itertools import product
from sklearn.metrics import f1_score
import tensorflow.contrib.data as tf_data


class Model:
    def __init__(self, input_shape, num_classes, verbose=True):
        tf.reset_default_graph()
        self.input_shape = input_shape
        self.num_classes = num_classes

        self.X = tf.placeholder('float', [None] + input_shape)
        self.Y = tf.placeholder('float', [None, self.num_classes])
        self.model = self.create_model()

        self.verbose = verbose
        self.losses = []

        self.set_loss_params()
        self.set_optimise_params()

    def __copy__(self):
        model = Model(self.input_shape, self.num_classes, self.verbose)
        model.set_loss_params(self.loss_weights, self.beta)
        return model

    def create_model(self):
        self.weights = {
            'h1': tf.Variable(tf.truncated_normal(self.input_shape + [256])),
            'h2': tf.Variable(tf.truncated_normal([256, 256])),
            'h3': tf.Variable(tf.truncated_normal([256, 256])),
            'out': tf.Variable(tf.truncated_normal([256, self.num_classes]))
        }
        self.biases = {
            'b1': tf.Variable(tf.truncated_normal([256])),
            'b2': tf.Variable(tf.truncated_normal([256])),
            'b3': tf.Variable(tf.truncated_normal([256])),
            'out': tf.Variable(tf.truncated_normal([self.num_classes]))
        }

        layer1 = tf.add(tf.matmul(self.X, self.weights['h1']), self.biases['b1'])
        layer2 = tf.add(tf.matmul(layer1, self.weights['h2']), self.biases['b2'])
        layer3 = tf.nn.softsign(tf.add(tf.matmul(layer2, self.weights['h3']), self.biases['b3']))
        return tf.add(tf.matmul(layer3, self.weights['out']), self.biases['out'])

    def set_loss_params(self, weights=np.zeros((0, 0)), beta=0.1):
        self.beta = 0.1
        self.loss_weights = weights

    def set_optimise_params(self, learning_rate=0.001, decay=0.9, momentum=0.0, epsilon=1e-10,
                            use_locking=False, centered=False):
        self.learning_rate = learning_rate
        self.decay = decay
        self.momentum = momentum
        self.epsilon = epsilon
        self.use_locking = use_locking
        self.centered = centered

    def loss(self):
        if not self.loss_weight.shape == np.ones((0, 0)).shape:
            nb_cl = len(self.loss_weights)
            final_mask = tf.zeros_like(self.model[..., 0])
            y_pred_max = tf.reduce_max(self.model, axis=-1)
            y_pred_max = tf.expand_dims(y_pred_max, axis=-1)
            y_pred_max_mat = tf.equal(self.model, y_pred_max)

            for c_p, c_t in product(range(nb_cl), range(nb_cl)):
                w = tf.cast(self.loss_weights[c_t, c_p], 'float32')
                y_p = tf.cast(y_pred_max_mat[..., c_p], 'float32')
                y_t = tf.cast(y_pred_max_mat[..., c_t], 'flaot32')
                final_mask += w * y_p * y_t
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.model, labels=self.Y) * final_mask
        else:
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.model, labels=self.Y)
        loss += self.beta * tf.nn.l2_loss(self.weights['h2']) + self.beta * tf.nn.l2_loss(self.biases['b2'])
        loss += self.beta * tf.nn.l2_loss(self.weights['h3']) + self.beta * tf.nn.l2_loss(self.biases['b3'])
        loss += self.beta * tf.nn.l2_loss(self.weights['out']) + self.beta * tf.nn.l2_loss(self.biases['out'])
        return loss

    def optimise(self, loss):
        optimiser = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate, decay=self.decay,
                                              momentum=self.momentum, use_locking=self.use_locking,
                                              centered=self.centered)
        return optimiser.minimize(loss)

    def converged(self, epochs, min_epochs=10, diff=0.1, converge_len=10):
        if len(self.losses) > min_epochs and epochs == -1:
            losses = self.losses[-converge_len:]
            for loss in losses[: (converge_len - 1)]:
                if abs(losses[-1] - loss) > diff:
                    return False
            return True
        return False

    def train(self, data, epochs=-1, batch_size=100, intervals=10):
        train_data, test_data, val_data = data.get_datasets(4, 10000, batch_size, batch_size * 10)
        train_batches, test_batches, val_batches = data.get_num_batches(batch_size, batch_size * 10)

        train_iterator = tf_data.Iterator.from_structure(train_data.output_types, train_data.output_shapes)
        train_next_batch = train_iterator.get_next()
        training_init_op = train_iterator.make_initializer(train_data)

        val_iterator = tf_data.Iterator.from_structure(val_data.output_types, test_data.output_shapes)
        val_next_batch = val_iterator.get_next()
        val_init_op = val_iterator.make_initializer(val_data)

        test_iterator = tf_data.Iterator.from_structure(test_data.output_types, test_data.output_shapes)
        test_next_batch = test_iterator.get_next()
        testing_init_op = test_iterator.make_initializer(test_data)

        loss = self.loss()
        optimiser = self.optimise(loss)
        correct_pred = tf.equal(tf.argmax(self.model, 1), tf.argmax(self.Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        init = tf.global_variables_initializer()

        with tf.Session as sess:
            sess.run(init)
            epoch = 0
            while not self.converged(epochs) and epoch != epochs:
                sess.run(training_init_op)
                train_loss = 0
                for _ in range(train_batches):
                    image_batch, label_batch = sess.run(train_next_batch)
                    _, cost = sess.run([optimiser, loss], feed_dict={self.X: image_batch, self.Y: label_batch})
                    train_loss += cost / train_batches
                epoch += 1
                sess.run(val_init_op)
                val_loss, val_acc = 0, 0
                for _ in range(val_batches):
                    image_batch, label_batch = sess.run(val_next_batch)
                    acc, cost = sess.run([optimiser, loss], feed_dict={self.X: image_batch, self.Y: label_batch})
                    val_loss += cost / val_batches
                    val_acc += acc / val_batches
                self.losses.append(val_loss)
                if self.verbose and epoch % intervals == 0:
                    message = 'Epoch: ' + str(epoch).zfill(4) + ' Training Loss: {:.4f}'.format(train_loss)
                    message += ' Validation Accuracy: {:.3f}'.format(val_acc)
                    message += ' Validation Loss: {:.4f}'.format(val_loss)
                    print(message)
            sess.run(testing_init_op)
            test_acc = 0
            predictions, labels = [], []
            for step in range(test_batches):
                image_batch, label_batch = sess.run(test_next_batch)
                acc, y_pred = sess.run([accuracy, tf.nn.softmax(self.model)], feed_dict={self.X:image_batch,
                                                                                         self.Y: label_batch})
                test_acc += acc / test_batches
                for i in range(len(label_batch)):
                    labels.append(np.argmax(label_batch[i]))
                    predictions.append(np.argmax(y_pred[i]))
        return test_acc, f1_score(labels, predictions)

    def predit(self, data):
        slice_predictions = []
        for i in range(len(data.data_y)):
            predict_data = data.get_predict_dataset(4, 10000, 10000, i)
            iterator = tf_data.Iterator.from_structure(predict_data.output_types, predict_data.output_shapes)
            next_batch = iterator.get_next()
            batches = data.get_num_predict_batches(10000, i)
            data_init_op = iterator.make_ititializer(predict_data)

            predictions = []
            init = tf.global_variables_initializer()
            with tf.Session() as sess:
                sess.run(init)
                sess.run(data_init_op)
                for step in range(batches):
                    image_batch = sess.run(next_batch)
                    predictions += sess.run(tf.nn.softmax(self.model), feed_dict={self.X: image_batch}).tolist()
            slice_predictions.append(predictions)
        return slice_predictions
