import tensorflow as tf
import scipy.stats as stats
from kerashypetune import KerasRandomSearch
from gensim import models
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization


##model for FastText pretrained embeddings (embed_dim set to 300)
def get_model(param, maxlen=25, vocab_size=vocab_size, embed_dim=300):
    
    inputs = layers.Input(shape=(maxlen,))
    embedding_layer = layers.Embedding(input_dim = vocab_size, output_dim=embed_dim, 
                                        input_length=maxlen)
    x = embedding_layer(inputs)
    x = layers.Conv1D(param['unit_1'], kernel_size=2, input_dim=(None, embed_dim), 
                      padding='valid', kernel_initializer='he_uniform')(x)
    x = layers.Dropout(param['dropout'])(x)
    avg_pool = layers.GlobalAveragePooling1D()(x)
    max_pool = layers.GlobalMaxPooling1D()(x)
    x = layers.concatenate([avg_pool, max_pool])
    outputs = layers.Dense(1, activation="sigmoid")(x)

    
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=param['lr']), 
                  loss="binary_crossentropy", metrics=["accuracy", tf.keras.metrics.AUC(name='auc')])
    return model

from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=0)
for train_index, test_index in sss.split(data['comment'], data['label']):
    X_train, X_test = data['comment'].iloc[train_index], data['comment'].iloc[test_index]
    y_train, y_test = data['label'].iloc[train_index], data['label'].iloc[test_index]
    

param_grid = {
    'unit_1': [32, 64, 128],
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


##############################################################################
##model without FastText Embeddings
ddef get_model(param, maxlen=25, vocab_size=n_word_unique + 1):
    
    inputs = layers.Input(shape=(maxlen,))
    embedding_layer = layers.Embedding(input_dim = vocab_size, output_dim=param['embed_dim'], 
                                        input_length=maxlen)
    x = embedding_layer(inputs)
    x = layers.Conv1D(param['unit_1'], kernel_size=2, input_dim=(None, param['embed_dim']), 
                      padding='valid', kernel_initializer='he_uniform')(x)
    x = layers.Dropout(param['dropout'])(x)
    avg_pool = layers.GlobalAveragePooling1D()(x)
    max_pool = layers.GlobalMaxPooling1D()(x)
    x = layers.concatenate([avg_pool, max_pool])
    outputs = layers.Dense(1, activation="sigmoid")(x)

    
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=param['lr']), 
                  loss="binary_crossentropy", metrics=["accuracy", tf.keras.metrics.AUC(name='auc')])
    return model

from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=0)
for train_index, test_index in sss.split(data['comment'], data['label']):
    X_train, X_test = data['comment'].iloc[train_index], data['comment'].iloc[test_index]
    y_train, y_test = data['label'].iloc[train_index], data['label'].iloc[test_index]
    
param_grid = {
    'unit_1': [32, 64, 128], 
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
