import os
import numpy as np
import pandas as pd
import tensorflow as tf
import sklearn.metrics as metrics
import tensorflow.contrib.data as tf_data


class Model:
    def __init__(self, input_shape, num_classes, verbose=True):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.verbose = verbose
        self.val_losses, self.train_losses = [], []

        with tf.device('/device:GPU:0'):
            self.X = tf.placeholder('float', [None] + input_shape, name='X')
            self.Y = tf.placeholder('float', [None, self.num_classes], name='Y')
            self.drop = tf.placeholder('float', name='Dropout')
            self.model = self.create_model()

    def __copy__(self):
        tf.reset_default_graph()
        return Model(self.input_shape, self.num_classes, self.verbose)

    def create_model(self):
        self.weights = {
            'c1': tf.Variable(tf.truncated_normal([3, 3, 3, 64])),
            'c2': tf.Variable(tf.truncated_normal([3, 3, 64, 64])),
            'c3': tf.Variable(tf.truncated_normal([3, 3, 64, 64])),
            'f1': tf.Variable(tf.truncated_normal([256, 512])),
            'f2': tf.Variable(tf.truncated_normal([512, 512])),
            'out': tf.Variable(tf.truncated_normal([512, self.num_classes]))
        }
        self.biases = {
            'c1': tf.Variable(tf.truncated_normal([64])),
            'c2': tf.Variable(tf.truncated_normal([64])),
            'c3': tf.Variable(tf.truncated_normal([64])),
            'f1': tf.Variable(tf.truncated_normal([512])),
            'f2': tf.Variable(tf.truncated_normal([512])),
            'out': tf.Variable(tf.truncated_normal([self.num_classes]))
        }

        conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(self.X, self.weights['c1'], [1,1,1,1], 'SAME'), self.biases['c1']))
        pool1 = tf.nn.max_pool(conv1, [1,2,2,1], [1,2,2,1], 'SAME')

        conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(pool1, self.weights['c2'], [1,1,1,1], 'SAME'), self.biases['c2']))
        pool2 = tf.nn.max_pool(conv2, [1,2,2,1], [1,2,2,1], 'SAME')

        conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(pool2, self.weights['c3'], [1,1,1,1], 'SAME'), self.biases['c3']))
        pool3 = tf.nn.max_pool(conv3, [1,2,2,1], [1,2,2,1], 'SAME')

        flat = tf.reshape(pool3, [-1, 256])
        full1 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(flat, self.weights['f1']), self.biases['f1'])), self.drop)
        full2 = tf.nn.dropout(tf.nn.softsign(tf.add(tf.matmul(full1, self.weights['f2']), self.biases['f2'])), self.drop)
        output = tf.add(tf.matmul(full2, self.weights['out']), self.biases['out'])
        return output

    def loss(self):
        with tf.device('/device:GPU:0'):
            loss = tf.losses.hinge_loss(labels=self.Y, logits=tf.nn.softmax(self.model))
            # loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.model, labels=self.Y)
        return loss

    def optimise(self, loss):
        optimiser = tf.train.AdamOptimizer(learning_rate=0.01)
        return optimiser.minimize(loss)

    def converged(self, epochs, epoch):
        if epochs == -1 and len(self.val_losses) > 5:
            if epoch >= 1000:
                return False
            else:
                if epoch % 5 == 0:
                    g_loss = 100 * ((self.val_losses[-1] / min(self.val_losses[:-1])) - 1)
                    progress = 1000 * ((sum(self.train_losses[-6:-1]) / (5 * min(self.train_losses[-6:-1])))-1)
                    print('Generalised Loss: {:.4f}'.format(g_loss))
                    print('Training Progress : {:.4f}'.format(progress))
                    print('Progress Quality: {:.4f}'.format(g_loss / progress))
                    if abs(g_loss / progress) > 2:
                        return False
                    else:
                        return True
                else:
                    return True
        elif epoch == epochs:
            return False
        else:
            return True

    def train(self, data, dir, epochs=-1, batch_size=100, intervals=1):
        with tf.device('/cpu:0'):
            train_data, test_data = data.get_datasets(4, 1000, batch_size, batch_size*10)
            train_batches, test_batches, val_batches = data.get_num_batches(batch_size, batch_size*10)

            train_iterator = tf_data.Iterator.from_structure(train_data.output_types, train_data.output_shapes)
            train_next_batch = train_iterator.get_next()
            training_init_op = train_iterator.make_initializer(train_data)

            # val_iterator = tf_data.Iterator.from_structure(val_data.output_types, val_data.output_shapes)
            # val_next_batch = val_iterator.get_next()
            # val_init_op = val_iterator.make_initializer(val_data)
            saver = tf.train.Saver()


        with tf.device('/device:GPU:0'):
            loss = self.loss()
            optimiser = self.optimise(loss)
            init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)
            epoch = 0
            if os.path.isdir(dir) and dir != 'supervised':
                print('Loaded weights from ' + dir + '\n')
                saver.restore(sess, dir + '/weights.ckpt')
            while self.converged(epochs, epoch):
                sess.run(training_init_op)
                train_loss = np.array([])
                for step in range(train_batches):
                    image_batch, label_batch = sess.run(train_next_batch)
                    _, cost = sess.run([optimiser, loss], feed_dict={self.X: image_batch, self.Y: label_batch,
                                                                     self.drop: 0.5})
                    train_loss = np.append(train_loss, cost)
                epoch += 1
                # sess.run(val_init_op)
                # val_loss, val_acc = np.array([]), 0
                # predictions, labels = [], []
                # for step in range(val_batches):
                #     image_batch, label_batch = sess.run(val_next_batch)
                #     cost, y_pred = sess.run([loss, tf.nn.softmax(self.model)], feed_dict={self.X: image_batch,
                #                                                                   self.Y: label_batch, self.drop: 0.5})
                #     for i in range(len(label_batch) - 1):
                #         labels.append(np.argmax(label_batch[i]))
                #         predictions.append(np.argmax(y_pred[i]))
                #     val_loss = np.append(val_loss, cost)
                # cmat = metrics.confusion_matrix(labels, predictions)
                # val_acc = np.mean(cmat.diagonal() / cmat.sum(axis=1))
                # self.val_losses.append(np.average(val_loss))
                self.train_losses.append(np.average(train_loss))
                if self.verbose and epoch % intervals == 0:
                    message = 'Epoch: ' + str(epoch).zfill(4)
                    message += ' Training Loss: {:.4f}'.format(np.average(train_loss))
                    # message += ' Validation Accuracy: {:.3f}'.format(val_acc)
                    # message += ' Validation Loss: {:.4f}'.format(np.average(val_loss))
                    print(message)

            saver.save(sess, dir + '/weights.ckpt')

            test_iterator = tf_data.Iterator.from_structure(test_data.output_types, test_data.output_shapes)
            test_next_batch = test_iterator.get_next()
            testing_init_op = test_iterator.make_initializer(test_data)

            sess.run(testing_init_op)
            predictions, prediction_scores, labels = [], [], []
            test_loss, test_acc = 0, 0
            for step in range(test_batches):
                image_batch, label_batch = sess.run(test_next_batch)
                y_pred, cost = sess.run([tf.nn.softmax(self.model), loss], feed_dict={self.X: image_batch, self.Y: label_batch,
                                                                        self.drop: 1.0})
                test_loss += np.average(cost)
                for i in range(len(label_batch) - 1):
                    labels.append(np.argmax(label_batch[i]))
                    predictions.append(np.argmax(y_pred[i]))
                    prediction_scores.append(max(y_pred[i]))
            f1 = metrics.f1_score(labels, predictions)
            roc = metrics.roc_auc_score(labels, prediction_scores)
            self.confusion_matrix(labels, predictions)
            cmat = metrics.confusion_matrix(labels, predictions)
            accuracy = np.mean(cmat.diagonal() / cmat.sum(axis=1))
            loss = test_loss / test_batches
            print('Model trained with an Accuracy: {:.4f}'.format(accuracy) + ' F1-Score: {:.4f}'.format(f1) +
                  ' ROC: {:.4f}'.format(roc) + ' Loss: {:.4f}'.format(loss) + ' in ' + str(epoch) + ' epochs\n\n')
            return accuracy, f1, roc, loss

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
                predictions += sess.run(tf.nn.softmax(self.model), feed_dict={self.X: image_batch, self.drop:1}).tolist()
        slice_predictions.append(predictions[0:indices[0]])
        for i in range(1, len(indices)):
            slice_predictions.append(predictions[indices[i-1]:indices[i]])
        return slice_predictions


    def confusion_matrix(self, labels, predictions):
        y_actu = labels
        y_pred = predictions
        p_labels = pd.Series(predictions)
        t_labels = pd.Series(y_actu)
        df_confusion = pd.crosstab(t_labels, p_labels, rownames=['Actual'], colnames=['Predicted'], margins=True)
        print(df_confusion)
        print(' ')
        print(metrics.classification_report(y_actu, y_pred, digits=5))
