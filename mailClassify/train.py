import tensorflow as tf
from MyData import MyData
from MyModel import MyModel
#import random


def main():
    batch_size = 500
    epoch_num = 50
    random_seed = 3000000
    exec_key = str(epoch_num)+"epc"+str(random_seed)

    # prepare data
    my_data = MyData("./data/data0.csv", max_freq=0.999, min_freq=0.001)
    my_data.import_data(None, random_seed)

    # data
    x = tf.placeholder(tf.float32, [None, my_data.word_dict_size]) # batch
    # label
    y_ = tf.placeholder(tf.int32, [None, my_data.label_dict_size]) # batch

    # model
    my_model = MyModel(my_data)
    y = my_model.inference(x)
    loss = my_model.loss(y, y_)
    train_op = my_model.train(loss)
    accuracy, correct, id_y = my_model.evaluate(y, y_)

    summary = tf.summary.merge_all()
    saver = tf.train.Saver()

    # run
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init) 

    summary_writer = tf.summary.FileWriter("./data/log-"+exec_key, sess.graph)

    # training
    for epoch in range(epoch_num):
        my_data.read_rewind()
        for step in range(int(my_data.data_size / batch_size)-1):
            _, word_batch, label_batch = my_data.read_next_batch(batch_size)
            sess.run([train_op, loss], feed_dict={x:word_batch, y_:label_batch})

            acc = sess.run(accuracy, feed_dict={x:word_batch,  y_:label_batch})
            print("TRAINING(%02d, %03d): %.0f%%" % (epoch, step, (acc * 100.0)))

            summary_str = sess.run(summary, feed_dict={x:word_batch, y_:label_batch})
            summary_writer.add_summary(summary_str, step)
            summary_writer.flush()

    # evaluate
    id_batch, word_batch, label_batch = my_data.read_next_batch(batch_size)
    acc = sess.run(accuracy, feed_dict={x:word_batch,  y_:label_batch})
    print("TEST: %.0f%%" % (acc * 100.0))

    saver.save(sess, "data/save-"+exec_key+"/model.ckpt")


if __name__ == "__main__":
    main()


