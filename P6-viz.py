import os, sys
from tqdm import tqdm
import tensorflow as tf

batch_size = 200
steps_per_epoch = 5000//batch_size
summary = True
iter_routing = 3

def mnist_reader(part):
    for i in range(10):
        with open('Part3_{}_{}.csv'.format(i, part)) as f:
            for line in f:
                yield line.strip(), i

def mnist_decoder(csv_line, label):
    FIELD_DEFAULTS = [[0.0]]*(28*28)
    with tf.variable_scope('DataSource'):
        fields = tf.decode_csv(csv_line, FIELD_DEFAULTS)
        im = tf.stack(fields)
        im = tf.reshape(im, (28, 28, 1))
        return im[:14, :14, :], im[:14, 14:, :] ,im[14:, :14, :], im[14:, 14:, :], tf.one_hot(label, depth=10)

tf.reset_default_graph()

with tf.variable_scope('DataSource'):
    dataset_val = tf.data.Dataset.from_generator(lambda: mnist_reader('Test'),
        (tf.string, tf.int32),
        (tf.TensorShape([]), tf.TensorShape([])
    )).map(mnist_decoder, num_parallel_calls=2) \
      .batch(batch_size) \
      .prefetch(1)

    iter_handle = tf.placeholder(tf.string, shape=[])
    data_iterator = tf.data.Iterator.from_string_handle(iter_handle, dataset_val.output_types, dataset_val.output_shapes)
    val_iterator = dataset_val.make_initializable_iterator()
    val_init_op = data_iterator.make_initializer(dataset_val)
    im1, im2, im3, im4, onehot_labels = data_iterator.get_next()

with tf.variable_scope('CNN'):
    conv77 = tf.keras.layers.Conv2D(16, (7,7), activation='relu')
    convmaps = [
        conv77(im)
            for im in [im1, im2, im3, im4]
    ]

from capsLayer import CapsLayer

with tf.variable_scope('QuadrantCaps'):
    l1caps = []
    for cm in convmaps:
        quadrantCaps = CapsLayer(num_outputs=1, vec_len=8, iter_routing=0, batch_size=batch_size, input_shape=(batch_size, 16, 8, 8), layer_type='CONV')
        l1caps.append(quadrantCaps(cm, kernel_size=7, stride=1))
    caps1 = tf.keras.layers.Concatenate(axis=1)(l1caps)
with tf.variable_scope('ClassCaps'):
    digitCaps = CapsLayer(num_outputs=10, vec_len=16, iter_routing=iter_routing, batch_size=batch_size, input_shape=(batch_size, 16, 8, 1), layer_type='FC')
    caps2 = digitCaps(caps1)

ctx_batch_size = batch_size
ctx_nclasses = 10
ctx_lambda_val = 0.5
ctx_m_plus = 0.9
ctx_m_minus = 0.1

with tf.variable_scope('Masking'):
    epsilon = 1e-9
    v_length = tf.sqrt(tf.reduce_sum(tf.square(caps2),
                                        axis=2, keepdims=True) + epsilon)
    softmax_v = tf.nn.softmax(v_length, axis=1)
    argmax_idx = tf.to_int32(tf.argmax(softmax_v, axis=1))
    argmax_idx = tf.reshape(argmax_idx, shape=(ctx_batch_size, ))
    masked_v = tf.multiply(tf.squeeze(caps2), tf.reshape(onehot_labels, (-1, ctx_nclasses, 1)))
    v_length = tf.sqrt(tf.reduce_sum(tf.square(caps2), axis=2, keepdims=True) + epsilon)

with tf.variable_scope('Loss'):
    max_l = tf.square(tf.maximum(0., ctx_m_plus - v_length))
    max_r = tf.square(tf.maximum(0., v_length - ctx_m_minus))
    max_l = tf.reshape(max_l, shape=(ctx_batch_size, -1))
    max_r = tf.reshape(max_r, shape=(ctx_batch_size, -1))
    L_c = onehot_labels * max_l + ctx_lambda_val * (1 - onehot_labels) * max_r
    margin_loss = tf.reduce_mean(tf.reduce_sum(L_c, axis=1))
    total_loss = margin_loss

with tf.variable_scope('Metrics'):
    labels = tf.to_int32(tf.argmax(onehot_labels, axis=1))
    # argmax_idx = tf.to_int32(tf.argmax(pred, axis=1))
    epoch_loss_avg, epoch_loss_avg_update = tf.metrics.mean(total_loss)
    epoch_accuracy, epoch_accuracy_update = tf.metrics.accuracy(labels, argmax_idx)

run_dir = os.path.join('/tmp/logdir', 'quadcaps-{}'.format(iter_routing))
assert os.path.exists(run_dir)

with tf.variable_scope('Initializer'):
    saver = tf.train.Saver(tf.trainable_variables())

config = tf.ConfigProto(
    # intra_op_parallelism_threads=2,
    # inter_op_parallelism_threads=2,
    allow_soft_placement=True)

# from tensorflow.python import debug as tf_debug
with tf.Session(config = config) as sess:
    # sess = tf_debug.TensorBoardDebugWrapperSession(sess, 'localhost:7000')
    saver.restore(sess, os.path.join(run_dir, 'model'))
    val_handle = sess.run(val_iterator.string_handle())

    sess.run(val_init_op, feed_dict={iter_handle: val_handle})
    sess.run(tf.local_variables_initializer())
    for _ in tqdm(range(steps_per_epoch)):
        sess.run([total_loss, epoch_loss_avg_update, epoch_accuracy_update], feed_dict={iter_handle: val_handle})
    print("vLoss: {:.5f} vAcc: {:.5f}".format(sess.run(epoch_loss_avg), sess.run(epoch_accuracy)))
