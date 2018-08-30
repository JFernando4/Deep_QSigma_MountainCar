import tensorflow as tf


def save_graph(sourcepath, tf_sess):
    saver = tf.train.Saver()
    save_path = saver.save(tf_sess, sourcepath+".ckpt")
    print("Model Saved in file: %s" % save_path)


def restore_graph(sourcepath, tf_sess):
    saver = tf.train.Saver()
    saver.restore(tf_sess, sourcepath+".ckpt")
    print("Model restored from file: %s" % sourcepath)
