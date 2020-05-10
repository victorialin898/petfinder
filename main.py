import os
import argparse
import tensorflow as tf
from model import cnn
import hyperparameters as hp
from preprocess import Datasets, create_sets
from data_vis import ImageLabelingLogger, ConfusionMatrixLogger, ConfusionMatrixLogger_nocallback

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Added this from the project 4 code to give us parsing functionality
def parse_args():
    """ Perform command-line argument parsing. """

    parser = argparse.ArgumentParser(
        description="Let's train some neural nets!")
    parser.add_argument(
        '--data',
        default=os.getcwd() + '/data/',
        help='Location where the dataset is stored.')
    parser.add_argument(
        '--load-checkpoint',
        default=None,
        help='''Path to model checkpoint file (should end with the
        extension .h5). Checkpoints are automatically saved when you
        train your model. If you want to continue training from where
        you left off, this is how you would load your weights. In
        the case of task 2, passing a checkpoint path will disable
        the loading of VGG weights.''')
    parser.add_argument(
        '--confusion',
        action='store_true',
        help='''Log a confusion matrix at the end of each
        epoch (viewable in Tensorboard). This is turned off
        by default as it takes a little bit of time to complete.''')
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='''Skips training and evaluates on the test set once.
        You can use this to test an already trained model by loading
        its checkpoint.''')

    return parser.parse_args()

def train(model, datasets, checkpoint_path):
    """
    Trains our model, handles checkpoints as well
    """
    callback_list = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path + \
                    "weights.e{epoch:02d}-" + \
                    "acc{val_sparse_categorical_accuracy:.4f}.h5",
            monitor='val_sparse_categorical_accuracy',
            save_best_only=True,
            save_weights_only=True),
        tf.keras.callbacks.TensorBoard(
            update_freq='batch',
            profile_batch=0),
        ImageLabelingLogger(datasets)
    ]

    if ARGS.confusion:
        callback_list.append(ConfusionMatrixLogger(datasets))

    # Fit model to test data
    model.fit(
        x=datasets.train_data,
        validation_data=datasets.test_data,
        epochs=hp.num_epochs,
        batch_size=None,
        callbacks=callback_list,
    )

def test(model, datasets):
    model.evaluate(
        x=datasets.test_data,
        verbose=1,
    )

    ConfusionMatrixLogger_nocallback(model, datasets)


def main():
    # Sets up the train and test directories according to flow_from_directory. Aborts if they are already present
    create_sets(ARGS.data, train_ratio=0.9)

    # Our datasets here
    datasets = Datasets(ARGS.data)

    model = cnn()
    checkpoint_path = "./your_model_checkpoints/"

    if ARGS.load_checkpoint is not None:
        model.load_weights(ARGS.load_checkpoint)

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    # Compile model graph
    model.compile(
        optimizer='sgd',
        loss='sparse_categorical_crossentropy',
        metrics=["sparse_categorical_accuracy"])

    if ARGS.evaluate:
        test(model, datasets)
    else:
        train(model, datasets, checkpoint_path)


ARGS = parse_args()

main()
