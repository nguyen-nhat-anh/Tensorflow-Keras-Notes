import tensorflow as tf
import pandas as pd
from tensorflow.python.lib.io import file_io


CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth',
                    'PetalLength', 'PetalWidth', 'Species']

FEATURE_COLUMNS = [tf.feature_column.numeric_column(key='SepalLength'),
                   tf.feature_column.numeric_column(key='SepalWidth'),
                   tf.feature_column.numeric_column(key='PetalLength'),
                   tf.feature_column.numeric_column(key='PetalWidth')]


def load_data(train_path, test_path, y_name='Species'):
    with file_io.FileIO(train_path, 'r') as f:
        train = pd.read_csv(f, names=CSV_COLUMN_NAMES, header=0)
    train_x, train_y = train, train.pop(y_name)

    with file_io.FileIO(test_path, 'r') as f:
        test = pd.read_csv(f, names=CSV_COLUMN_NAMES, header=0)
    test_x, test_y = test, test.pop(y_name)

    return (train_x, train_y), (test_x, test_y)


def train_input_fn(features, labels, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)
    return dataset


def eval_input_fn(features, labels, batch_size):
    features = dict(features)
    if labels is None:
        inputs = features
    else:
        inputs = (features, labels)

    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    return dataset


def my_model(features, labels, mode, params):
    input_layer = tf.feature_column.input_layer(features, params['feature_columns'])
    hidden_layer_1 = tf.layers.dense(input_layer, units=10, activation=tf.nn.relu)
    hidden_layer_2 = tf.layers.dense(hidden_layer_1, units=10, activation=tf.nn.relu)
    logits = tf.layers.dense(hidden_layer_2, units=params['n_classes'], activation=None)

    # Predict
    predicted_class = tf.argmax(logits, 1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_class,
            'probabilities': tf.nn.softmax(logits),
            'logits': logits
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Train and Eval
    ## Compute loss
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    ## Compute evaluation metrics
    accuracy = tf.metrics.accuracy(labels=labels, predictions=predicted_class, name='acc_op')
    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])

    ## Eval
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    ## Train
    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def json_serving_input_fn():
    inputs = {}
    for feat in FEATURE_COLUMNS:
        inputs[feat.name] = tf.placeholder(shape=[None], dtype=feat.dtype)
    return tf.estimator.export.ServingInputReceiver(inputs, inputs)