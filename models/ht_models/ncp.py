import tensorflow as tf
import scipy.stats as stats
from kerasncp.tf import LTCCell
from kerashypetune import KerasRandomSearch
from gensim import models
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization


def get_model(param, maxlen=25, vocab_size=vocab_size):
    wiring = kncp.wirings.NCP(
                                inter_neurons=12,  
                                command_neurons=8,  
                                motor_neurons=1,  
                                sensory_fanout=4,  
                                inter_fanout=4,  
                                recurrent_command_synapses=4,  
                                motor_fanin=6,
                              )
    ltc_cell = LTCCell(wiring) 
    model = keras.Sequential(
        [
            keras.layers.InputLayer(input_shape=maxlen),            
            keras.layers.Embedding( input_dim = vocab_size, 
                                    output_dim = 300, 
                                    embeddings_initializer=keras.initializers.Constant(matrix),
                                    trainable=True),
            keras.layers.Dropout(param['dropout']),
            keras.layers.RNN(ltc_cell, return_sequences=False),
            keras.layers.Dense(1, activation="sigmoid")

        ]
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=param['lr']),
                loss="binary_crossentropy", metrics=["accuracy", tf.keras.metrics.AUC(name='auc')])

    return model
    
from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=0)
for train_index, test_index in sss.split(df['comment'], df['label']):
    X_train, X_test = df['comment'].iloc[train_index], df['comment'].iloc[test_index]
    y_train, y_test = df['label'].iloc[train_index], df['label'].iloc[test_index]

param_grid = {
    'dropout': [0.1, 0.2, 0.3, 0.4, 0.5],
    'lr': stats.uniform(1e-4, 1e-2),
    'epochs': 5,
    'batch_size': [32, 64, 128]
}

X_train = vectorizer(np.array([[s] for s in X_train])).numpy()
X_test = vectorizer(np.array([[s] for s in X_test])).numpy()
X_train = keras.preprocessing.sequence.pad_sequences(X_train, maxlen=25)
X_test =  keras.preprocessing.sequence.pad_sequences(X_test, maxlen=25)
kgs = KerasRandomSearch(get_model, param_grid, monitor='val_auc', greater_is_better=True,
                        n_iter=15)
kgs.search(X_train, y_train, validation_data=(X_test, y_test))
print(kgs.best_params)



def get_model(param, maxlen=25, vocab_size=n_word_unique + 1):
    wiring = kncp.wirings.NCP(
                                inter_neurons=12, 
                                command_neurons=8,  
                                motor_neurons=1,  
                                sensory_fanout=4,  
                                inter_fanout=4,  
                                recurrent_command_synapses=4, 
                                motor_fanin=6,
                              )
    ltc_cell = LTCCell(wiring) # Create LTC model
    
    model = keras.Sequential(
        [
            keras.layers.InputLayer(input_shape=(maxlen,)),            
            keras.layers.Embedding(input_dim = vocab_size, 
                                               output_dim = param['embed_dim'], 
                                               input_length=maxlen),
            keras.layers.Dropout(param['dropout']),
            keras.layers.RNN(ltc_cell, return_sequences=False),
            keras.layers.Dense(1, activation="sigmoid")

        ]
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=param['lr']),
                loss="binary_crossentropy", metrics=["accuracy", tf.keras.metrics.AUC(name='auc')])

    return model
    
from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=0)
for train_index, test_index in sss.split(df['comment'], df['label']):
    X_train, X_test = df['comment'].iloc[train_index], df['comment'].iloc[test_index]
    y_train, y_test = df['label'].iloc[train_index], df['label'].iloc[test_index]

param_grid = {
    'dropout': [0.1, 0.2, 0.3, 0.4, 0.5],
    'embed_dim': [32, 64, 128],
    'lr': stats.uniform(1e-4, 1e-2),
    'epochs': 5,
    'batch_size': [32, 64, 128]
}

X_train = keras.preprocessing.sequence.pad_sequences(X_train, maxlen=25)
X_test =  keras.preprocessing.sequence.pad_sequences(X_test, maxlen=25)


kgs = KerasRandomSearch(get_model, param_grid, monitor='val_auc', greater_is_better=True,
                        n_iter=15)
kgs.search(X_train, y_train, validation_data=(X_test, y_test))
print(kgs.best_params)   
