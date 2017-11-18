import tensorflow as tf
from MyData import MyData
from MyModel import MyModel


def main():
    batch_size = 500
    epoch_num = 50
    random_seed = 1000000
    exec_key = str(epoch_num)+"epc"+str(random_seed)

    # prepare data
    my_data = MyData("./data0.csv", max_freq=0.999, min_freq=0.001)
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

    saver.restore(sess, "save-"+exec_key+"/model.ckpt")

    # evaluate training
    s = int(my_data.data_size / batch_size)-1
    s = s * batch_size
    id_batch, word_batch, label_batch = my_data.read_next_batch(s)
    acc, corr = sess.run([accuracy, correct], feed_dict={x:word_batch,  y_:label_batch})
    print("EVAL: %.0f%%" % (acc * 100.0))

    # evaluate test
    id_batch, word_batch, label_batch = my_data.read_next_batch(batch_size)
    acc, corr, idy = sess.run([accuracy, correct, id_y], feed_dict={x:word_batch,  y_:label_batch})
    print("TEST: %.0f%%" % (acc * 100.0))
    for i in range(len(corr)):
        if not corr[i]:
            print("rand[%d] %s(%d) data[%d] %s %s" % (i, my_data.label_list[idy[i]], idy[i], id_batch[i], my_data.csv_data[id_batch[i]], "ss"))


if __name__ == "__main__":
    main()


