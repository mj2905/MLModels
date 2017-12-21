# CS-433 Project 2: Text classification task

### Dependencies
The following libraries are required (even to make predictions from pretrained model):
- Python3 (tested with version 3.6)
- Numpy and Scipy
- Keras2
- Tensorflow

To install all dependencies (also the ones required for other models) in one step, type: `pip install -r requirements.txt`

We built and ran this project on OSX (last version) and CentOS 7.

## Running the model
### Without training, using pretrained model
1. Uncompress the [tweets files](https://www.kaggle.com/c/epfml17-text/data) in the twitter-datasets folder.
2. Run the script `python3 run.py` with no arguments.
3. A file `submission.csv` is generated in the same folder as "run.py", upload it to [Kaggle submission page](https://www.kaggle.com/c/epfml17-text/submit) for grading.

### Retrain the model from scratch (optional, ~ 2 hours on a machine with recent GPU)
1. Download the [tweets files](https://www.kaggle.com/c/epfml17-text/data)
2. Download the [Glove words-embeddings](http://nlp.stanford.edu/data/glove.twitter.27B.zip) and unzip them in the `data` folder
3. In the file `configuration.py`: set `'trainable': True` for the `submitted_model
4. Run the script `python3 run.py` with no arguments.

## Directories structure
### `data` folder
Contains the "temporary" data.
You need to put the pretrained embeddings files in this folder.

`words-by-frequency.txt`: dictionnary of words ordered by decreasing frequency (used to split hashtags)

`empty_embeddings.txt`: dummy file used to train embeddings from scratch

#### `data/model_xyz` folders
Each time a model is trained, the temporary files are written in this folder.
- The keras model is persisted at each epochs ends `model_name_EPOCH_NB.hdf5`
- A graph of training is saved in `graph.png`
- The preprocessed tweets are saved in `test_data.pkl` (used generating predictions) 

### `twitter-datasets`
Where to uncompress the twitter dataset downloaded from Kaggle.

### `utils`
Contains python classes and scripts implemented for this project.

`utils.py`: contains utility methods (create submission file, plot training graph, ...)
`tweets_embeddings_builder.py`: class used to parse tweet files, build word index, vocabulary and finally the Keras embedding layer (see comment in the file)
`tweets_cleanup.py`: methods used to preprocess the tweets (see comment in the file)

## Description of important files and folders

### `run.py`
This is the main script used to run convolutionnal neural network that produced the file uploaded on Kaggle.

It can take an argument for specifying the model to run: `python3 run.py cnn_tests_with_2_dense_layers`
All the parameters of the models are defined in `configuration.py`.

To enable the training phase for the submitted model, simple set `'training': True` in the `submitted_model`. Be aware that this step take a really long time, more than one hour on a computer with a decent GPU.

### `configuration.py`
Contain the parameters of the Keras model we used for submission and also other models we tried.

### `keras_models.py`
Contain the generic implementations of the different models we tried, the parameters of the models are defined in `configuration.py`
