import argparse
import tensorflow as tf
import model
import os


def train_and_evaluate(args):
    (train_x, train_y), (test_x, test_y) = model.load_data(args.train_files, args.eval_files)
    feature_columns = model.FEATURE_COLUMNS
    classifier = tf.estimator.Estimator(
        model_fn=model.my_model,
        model_dir=args.job_dir,
        params={
            'feature_columns': feature_columns,
            'n_classes': 3
        }
    )
    classifier.train(
        input_fn=lambda: model.train_input_fn(train_x, train_y, batch_size=32),
        steps=1000)

    eval_result = classifier.evaluate(
        input_fn=lambda: model.eval_input_fn(test_x, test_y, batch_size=32))

    classifier.export_savedmodel(os.path.join(args.job_dir, 'export'), model.json_serving_input_fn)
    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train-files'
    )
    parser.add_argument(
        '--eval-files'
    )
    parser.add_argument(
        '--job-dir'
    )
    args, _ = parser.parse_known_args()
    tf.logging.set_verbosity(tf.logging.INFO)
    train_and_evaluate(args)