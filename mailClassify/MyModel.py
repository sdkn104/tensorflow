import tensorflow as tf
import MyData
#import random
#import codecs, csv

class MyModel:
    def __init__(self, my_data_):
        self.hidden1_size = 500
        self.hidden2_size = 100
        self.my_data = my_data_

    # inference
    def inference(self, x):
        Wh1 = tf.Variable(tf.random_normal([self.my_data.word_dict_size, self.hidden1_size], mean=0.0, stddev=0.05), name="Wh1")
        bh1 = tf.Variable(tf.zeros([self.hidden1_size]), name="bh1")
        Wh2 = tf.Variable(tf.random_normal([self.hidden1_size, self.hidden2_size], mean=0.0, stddev=0.05), name="Wh2")
        bh2 = tf.Variable(tf.zeros([self.hidden2_size]), name="bh2")
        W = tf.Variable(tf.random_normal([self.hidden2_size, self.my_data.label_dict_size], mean=0.0, stddev=0.05), name="W")
        b = tf.Variable(tf.zeros([self.my_data.label_dict_size]), name="b")
        h1 = tf.nn.relu(tf.matmul(x, Wh1) + bh1)
        h2 = tf.nn.relu(tf.matmul(h1, Wh2) + bh2)
        y = tf.nn.relu(tf.matmul(h2, W) + b)
        return y

    # loss
    def loss(self, y, y_):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                             labels=y_, logits=y, name='xentropy')
        loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
        tf.summary.scalar('loss', loss)
        return loss

    # train
    def train(self, loss):
        optimizer = tf.train.GradientDescentOptimizer(0.6)
        train_op = optimizer.minimize(loss)
        return train_op

    # evaluate
    def evaluate(self, y, y_):
        id_y = tf.argmax(y, 1)
        correct = tf.equal(id_y, tf.argmax(y_, 1))
        correct_count = tf.reduce_sum(tf.cast(correct, tf.int32))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        tf.summary.scalar('accuracy', accuracy)
        return accuracy, correct, id_y



def main():
    # prepare data
    my_data = MyData("./data0.csv", max_freq=0.999, min_freq=0.001)
    my_data.import_data(None)

    # data
    x = tf.placeholder(tf.float32, [None, my_data.word_dict_size]) # batch
    # label
    y_ = tf.placeholder(tf.int32, [None, my_data.label_dict_size]) # batch

    # model
    my_model = MyModel(my_data)
    y = my_model.inference(x)
    loss = my_model.loss(y, y_)
    train_op = my_model.train(loss)
    accuracy = my_model.evaluate(y, y_)

    summary = tf.summary.merge_all()
    saver = tf.train.Saver()

    # run
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init) 

    summary_writer = tf.summary.FileWriter("./log", sess.graph)

    batch_size = 500
    for epoch in range(1):
        my_data.read_rewind()
        for step in range(int(my_data.data_size / batch_size)-1):
            word_batch, label_batch = my_data.read_next_batch(batch_size)
            _, losses = sess.run([train_op, loss], feed_dict={x:word_batch, y_:label_batch})

            evals = sess.run(accuracy, feed_dict={x:word_batch,  y_:label_batch})
            print("TRAINING(%02d, %03d): %.0f%%" % (epoch, step, (evals * 100.0)))

            summary_str = sess.run(summary, feed_dict={x:word_batch, y_:label_batch})
            summary_writer.add_summary(summary_str, step)
            summary_writer.flush()

    word_batch, label_batch = my_data.read_next_batch(batch_size)
    evals = sess.run(accuracy, feed_dict={x:word_batch,  y_:label_batch})
    print("TEST: %.0f%%" % (evals * 100.0))

    saver.save(sess, "save/model.ckpt")

if __name__ == "__main__":
    main()


