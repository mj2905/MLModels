"""
Contains utility methods
"""
import pickle as pk
import numpy as np
import matplotlib.pyplot as plt


def save_pkl(obj, filename):
    """
    Save an object to a pkl file
    :param obj: the object to persist
    :param filename: the file
    :return: None
    """
    with open(filename, 'wb') as f:
        pk.dump(obj, f)


def load_pkl(filename):
    """
    Load a persisted object for a pkl file
    :param filename: the filename
    :return: object
    """
    with open(filename, 'rb') as f:
        return pk.load(f)


def create_submission_file_from_keras_model(model, test_data_pkl, filename):
    """
    Create a submission file from a Keras model
    :param model: the model from which make the predictions
    :param test_data_pkl: the filename of persisted (and preprocessed) test_data
    :param filename: where to write the submission file
    :return: None
    """
    x_test, ids_test = load_pkl(test_data_pkl)
    assert(len(x_test) == 10000 and len(ids_test) == 10000)

    predictions = model.predict(x_test)
    predictions = [-1 if p[0] > p[1] else 1 for p in predictions]
    indices = np.argsort(ids_test)
    with open(filename, 'w', encoding='utf8') as f:
        f.write('Id,Prediction\n')
        for i in indices:
            f.write('{},{}\n'.format(str(int(ids_test[i])), str(int(predictions[i]))))
    print('{} submissions written to file: {}'.format(str(len(ids_test)), filename))


def plot_keras_accuracy_graph(history, filename):
    """
    Create the accuracy plot from the history object returned by model.fit()
    :param history: the history object
    :param filename: where to save the *.png file
    :return: None
    """
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.grid()
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(filename)
    print('Training graph written to : {}'.format(filename))
