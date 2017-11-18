import tensorflow as tf
import create_dict
import random
import codecs, csv

class MyData:
    def __init__(self, csv_name, max_freq=1.0, min_freq=0.0):
        self.csv_file_name = csv_name
        self.word_dict, self.label_dict = create_dict.createDict(csv_name, max_freq, min_freq)
        self.word_dict_size = len(self.word_dict)
        self.label_dict_size = len(self.label_dict)
        self.current_idx = 0

    def import_data(self, csv_name):
        print("reading csv")
        csvfile = codecs.open(csv_name or self.csv_file_name, 'r', encoding="utf-8")
        csvReader = csv.reader(csvfile)
        self.csv_data = [ row for row in csvReader ]
        csvfile.close()
        random.shuffle(self.csv_data)
        self.data_size = len(self.csv_data)

    def read_next(self):
       if self.current_idx >= len(self.csv_data):
           return None
       row = self.csv_data[self.current_idx]
       self.current_idx += 1
       word_vec, label_dec = create_dict.vectorize(row, self.word_dict, self.label_dict)
       return word_vec, label_dec

    def read_next_batch(self, batch_size):
       word_batch = []
       label_batch = []
       for i in range(batch_size):
           w, l = self.read_next()
           word_batch.append(w)
           label_batch.append(l)
       return word_batch, label_batch

def main():
    # prepare data
    my_data = MyData("./data0.csv", max_freq=0.999, min_freq=0.001)
    my_data.import_data(None)

    # data
    x = tf.placeholder(tf.float32, [None, my_data.word_dict_size]) # batch
    # label
    y_ = tf.placeholder(tf.int32, [None, my_data.label_dict_size]) # batch
 
    # inference
    W = tf.Variable(tf.random_normal([my_data.word_dict_size, my_data.label_dict_size], mean=0.0, stddev=0.05))
    b = tf.Variable(tf.zeros([my_data.label_dict_size]))
    y = tf.nn.softmax(tf.matmul(x, W) + b)

    # loss
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                         labels=y_, logits=y, name='xentropy')
    loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    tf.summary.scalar('loss', loss)

    # train
    optimizer = tf.train.GradientDescentOptimizer(0.5)
    train_op = optimizer.minimize(loss)

    # evaluate
    correct = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    correct_count = tf.reduce_sum(tf.cast(correct, tf.int32))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    summary = tf.summary.merge_all()

    # run
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init) 

    summary_writer = tf.summary.FileWriter("./log", sess.graph)

    batch_size = 500
    for step in range(int(my_data.data_size / batch_size)):
        word_batch, label_batch = my_data.read_next_batch(batch_size)
        _, losses = sess.run([train_op, loss], feed_dict={x:word_batch, y_:label_batch})

        evals = sess.run(accuracy, feed_dict={x:word_batch,  y_:label_batch})
        print("TRAINING(%03d): %.0f%%" % (step, (evals * 100.0)))

        summary_str = sess.run(summary, feed_dict={x:word_batch, y_:label_batch})
        summary_writer.add_summary(summary_str, step)
        summary_writer.flush()

if __name__ == "__main__":
    main()


