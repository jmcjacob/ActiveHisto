import numpy as np
import tensorflow as tf
from itertools import product
from sklearn.metrics import f1_score
import tensorflow.contrib.data as tf_data


class Model:
    def __init__(self, input_shape, num_classes, verbose=True, bootstrap=False):
        tf.reset_default_graph()
        self.input_shape = input_shape
        self.num_classes = num_classes

        with tf.device('/device:GPU:0'):
            self.X = tf.placeholder('float', [None] + input_shape, name='X')
            self.Y = tf.placeholder('float', [None, self.num_classes], name='Y')
            self.Drop = tf.placeholder('float', name='Dropout')
            self.model = self.create_model()

        self.set_loss_params()
        self.set_optimise_params()

        self.bootstrap = bootstrap
        self.verbose = verbose
        self.losses = []

    def __copy__(self):
        model = Model(self.input_shape, self.num_classes, self.verbose)
        model.set_loss_params(self.loss_weights, self.beta)
        model.set_optimise_params(self.learning_rate, self.decay, self.momentum, self.epsilon, self.use_locking,
                                  self.centered)
        return model

    def create_model(self):
        self.weights = {
            'c1': tf.Variable(tf.random_normal([3, 3, 3, 64])),
            'c2': tf.Variable(tf.random_normal([3, 3, 64, 64])),
            'c3': tf.Variable(tf.random_normal([3, 3, 64, 128])),
            'c4': tf.Variable(tf.random_normal([3, 3, 128, 128])),
            'c5': tf.Variable(tf.random_normal([3, 3, 128, 256])),
            'c6': tf.Variable(tf.random_normal([3, 3, 256, 256])),
            'c7': tf.Variable(tf.random_normal([3, 3, 256, 256])),
            'c8': tf.Variable(tf.random_normal([3, 3, 256, 512])),
            'c9': tf.Variable(tf.random_normal([3, 3, 512, 512])),
            'c10': tf.Variable(tf.random_normal([3, 3, 512, 512])),
            'c11': tf.Variable(tf.random_normal([3, 3, 512, 512])),
            'c12': tf.Variable(tf.random_normal([3, 3, 512, 512])),
            'c13': tf.Variable(tf.random_normal([3, 3, 512, 512])),
            'fw1': tf.Variable(tf.random_normal([512, 4096]), name='fw1'),
            'fw2': tf.Variable(tf.random_normal([4096, 4096]), name='fw2'),
            'outw': tf.Variable(tf.random_normal([4096, self.num_classes]), name='outw')
        }
        self.biases = {
            'b1': tf.Variable(tf.random_normal([64])),
            'b2': tf.Variable(tf.random_normal([64])),
            'b3': tf.Variable(tf.random_normal([128])),
            'b4': tf.Variable(tf.random_normal([128])),
            'b5': tf.Variable(tf.random_normal([256])),
            'b6': tf.Variable(tf.random_normal([256])),
            'b7': tf.Variable(tf.random_normal([256])),
            'b8': tf.Variable(tf.random_normal([512])),
            'b9': tf.Variable(tf.random_normal([512])),
            'b10': tf.Variable(tf.random_normal([512])),
            'b11': tf.Variable(tf.random_normal([512])),
            'b12': tf.Variable(tf.random_normal([512])),
            'b13': tf.Variable(tf.random_normal([512])),
            'fb1': tf.Variable(tf.random_normal([4096]), name='fb1'),
            'fb2': tf.Variable(tf.random_normal([4096]), name='fb2'),
            'outb': tf.Variable(tf.random_normal([self.num_classes]), name='outb')
        }

        conv_1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(self.X, self.weights['c1'], [1,1,1,1], 'SAME'),
                                           self.biases['b1']))
        conv_2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv_1, self.weights['c2'], [1,1,1,1], 'SAME'),
                                           self.biases['b2']))
        pool_1 = tf.nn.max_pool(conv_2, [1,2,2,1], [1,2,2,1], 'SAME')

        conv_3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(pool_1, self.weights['c3'], [1,1,1,1], 'SAME'),
                                           self.biases['b3']))
        conv_4 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv_3, self.weights['c4'], [1,1,1,1], 'SAME'),
                                           self.biases['b4']))
        pool_2 = tf.nn.max_pool(conv_4, [1,2,2,1], [1,2,2,1], 'VALID')

        conv_5 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(pool_2, self.weights['c5'], [1,1,1,1], 'SAME'),
                                           self.biases['b5']))
        conv_6 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv_5, self.weights['c6'], [1,1,1,1], 'SAME'),
                                           self.biases['b6']))
        conv_7 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv_6, self.weights['c7'], [1,1,1,1], 'SAME'),
                                           self.biases['b7']))
        pool_3 = tf.nn.max_pool(conv_7, [1,2,2,1], [1,2,2,1], 'SAME')

        conv_8 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(pool_3, self.weights['c8'], [1,1,1,1], 'SAME'),
                                           self.biases['b8']))
        conv_9 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv_8, self.weights['c9'], [1,1,1,1], 'SAME'),
                                           self.biases['b9']))
        conv_10 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv_9, self.weights['c10'], [1,1,1,1], 'SAME'),
                                            self.biases['b10']))
        pool_4 = tf.nn.max_pool(conv_10, [1,2,2,1], [1,2,2,1], 'SAME')

        conv_11 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(pool_4, self.weights['c11'], [1,1,1,1], 'SAME'),
                                            self.biases['b11']))
        conv_12 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv_11, self.weights['c12'], [1,1,1,1], 'SAME'),
                                            self.biases['b12']))
        conv_13 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv_12, self.weights['c13'], [1,1,1,1], 'SAME'),
                                            self.biases['b13']))
        pool_5 = tf.nn.max_pool(conv_13, [1,2,2,1], [1,2,2,1], 'SAME')

        fc1 = tf.reshape(pool_5, [-1, self.weights['fw1'].get_shape().as_list()[0]])
        full_1 = tf.nn.dropout(tf.nn.relu(tf.nn.bias_add(tf.matmul(fc1, self.weights['fw1']), self.biases['fb1'])),
                                   self.Drop)
        full_2 = tf.nn.dropout(tf.nn.relu(tf.nn.bias_add(tf.matmul(full_1, self.weights['fw2']), self.biases['fb2'])),
                                   self.Drop)
        output = tf.nn.bias_add(tf.matmul(full_2, self.weights['outw']), self.biases['outb'])
        return output

    def set_loss_params(self, weights=None, beta=0.1):
        self.beta = beta
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
        with tf.device('/device:GPU:0'):
            if self.loss_weights is not None:
                print(self.loss_weights)
                return tf.nn.weighted_cross_entropy_with_logits(logits=tf.nn.softmax(self.model),
                                                                targets=self.Y, pos_weight=self.loss_weights)
            else:
                return tf.nn.softmax_cross_entropy_with_logits(logits=self.model, labels=self.Y)

    def optimise(self, loss):
        with tf.device('/device:GPU:0'):
            optimiser = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate, decay=self.decay,
                                                  momentum=self.momentum, use_locking=self.use_locking,
                                                  centered=self.centered)
        if self.bootstrap:
            final_vars = [self.weights['fw1'], self.weights['fw2'], self.weights['outw'],
                    self.biases['fb1'], self.biases['fb2'], self.biases['outb']]
            return optimiser.minimize(loss, var_list=final_vars)
        else:
            return optimiser.minimize(loss)

    def converged(self, epochs, min_epochs=2, diff=5.0, converge_len=5):
        if self.bootstrap:
            diff = 0.5
            converge_len = 10
        if len(self.losses) > min_epochs and epochs == -1:
            losses = self.losses[-converge_len:]
            for loss in losses[: (converge_len - 1)]:
                if abs(losses[-1] - loss) > diff:
                    return False
            return True
        return False

    def train(self, data, epochs=-1, batch_size=100, intervals=10):
        with tf.device('/cpu:0'):
            train_data, test_data, val_data = data.get_datasets(4, 100, batch_size, batch_size * 10)
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
            saver = tf.train.Saver()

        with tf.device('/device:GPU:0'):
            loss = self.loss()
            optimiser = self.optimise(loss)
            correct_pred = tf.equal(tf.argmax(self.model, 1), tf.argmax(self.Y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)
            if self.bootstrap:
                saver.restore(sess, 'tmp/model.ckpt')
            epoch = 0
            while not self.converged(epochs) and epoch != epochs:
                sess.run(training_init_op)
                train_loss = 0
                for step in range(train_batches):
                    image_batch, label_batch = sess.run(train_next_batch)
                    _, cost = sess.run([optimiser, loss], feed_dict={self.X: image_batch, self.Y: label_batch, self.Drop: .5})
                    # print(str(step).zfill(5) + '/' + str(train_batches) + ' Loss: ' + str((sum(cost) / len(cost))))
                    train_loss += sum(sum(cost) / len(cost)) / 2
                epoch += 1
                sess.run(val_init_op)
                val_loss, val_acc = 0, 0
                for step in range(val_batches):
                    image_batch, label_batch = sess.run(val_next_batch)
                    acc, cost = sess.run([accuracy, loss], feed_dict={self.X: image_batch, self.Y: label_batch, self.Drop:1.})
                    val_loss += sum(sum(cost) / len(cost)) / 2
                    val_acc += acc
                self.losses.append(val_loss)
                if self.verbose and epoch % intervals == 0:
                    message = 'Epoch: ' + str(epoch).zfill(4)
                    message += ' Training Loss: {:.4f}'.format(train_loss / train_batches)
                    message += ' Validation Accuracy: {:.3f}'.format(val_acc / val_batches)
                    message += ' Validation Loss: {:.4f}'.format(val_loss / val_batches)
                    print(message)
                    print(message, file=open('results.txt', 'a'))
            if not self.bootstrap:
                saver.save(sess, 'tmp/model.ckpt')
            sess.run(testing_init_op)
            test_acc = 0
            predictions, labels = [], []
            for step in range(test_batches):
                image_batch, label_batch = sess.run(test_next_batch)
                acc, y_pred = sess.run([accuracy, tf.nn.softmax(self.model)], feed_dict={self.X:image_batch,
                                                                                         self.Y: label_batch,
                                                                                         self.Drop: 1.})
                test_acc += acc
                for i in range(len(label_batch) - 1):
                    labels.append(np.argmax(label_batch[i]))
                    predictions.append(np.argmax(y_pred[i]))
        accuracy = test_acc / test_batches
        f1 = f1_score(labels, predictions)
        print('Model trained with an Accuracy: {:.4f}'.format(accuracy) + ' F1-Score: {:.4f}'.format(f1) + ' in ' +
              str(epoch) + ' epochs', file=open('results.txt', 'a'))
        print('Model trained with an Accuracy: {:.4f}'.format(accuracy) + ' F1-Score: {:.4f}'.format(f1) + ' in ' +
              str(epoch) + ' epochs')
        return accuracy, f1

    def predict(self, data):
        slice_predictions = []
        predictions = []
        with tf.device('/cpu:0'):
            predict_data, indices = data.get_predict_dataset(4, 1000, 1000)
            iterator = tf_data.Iterator.from_structure(predict_data.output_types, predict_data.output_shapes)
            next_batch = iterator.get_next()
            batches = data.get_num_predict_batches(1000)
            data_init_op = iterator.make_initializer(predict_data)
        with tf.device('/device:GPU:0'):
            init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            sess.run(data_init_op)
            for step in range(batches):
                image_batch = sess.run(next_batch)
                predictions += sess.run(tf.nn.softmax(self.model),
                                        feed_dict={self.X: image_batch, self.Drop:1.}).tolist()
        slice_predictions.append(predictions[0:indices[0]])
        for i in range(1, len(indices)):
            slice_predictions.append(predictions[indices[i-1
            ]:indices[i]])
        for i in range(len(slice_predictions)):
            if slice_predictions[i] == []:
                print(indices, file=open('thing.txt', 'a'))
                print(data.data_x[i])
                print(i)
                print(slice_predictions[i])

        return slice_predictions
