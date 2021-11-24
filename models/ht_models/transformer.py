import tensorflow as tf
import scipy.stats as stats
from kerashypetune import KerasRandomSearch
from gensim import models
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.2):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions
    

##model for FastText pretrained embeddings (embed_dim set to 300)
def get_model(param, maxlen=25, vocab_size=vocab_size, num_heads=8, embed_dim=300):
    
    inputs = layers.Input(shape=(maxlen,))
    embedding_layer = layers.Embedding(input_dim = vocab_size, output_dim=embed_dim, 
                                        input_length=maxlen)
    x = embedding_layer(inputs)
    transformer_block = TransformerBlock(embed_dim, num_heads, param['unit_1'])
    x = transformer_block(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(param['dropout'])(x)
    x = layers.BatchNormalization()(x)
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
def get_model(param, maxlen=25, vocab_size=n_word_unique + 1, num_heads=8):
    
    inputs = layers.Input(shape=(maxlen,))
    embedding_layer = layers.Embedding(input_dim = vocab_size, output_dim=param['embed_dim'], 
                                        input_length=maxlen)
    x = embedding_layer(inputs)
    transformer_block = TransformerBlock(param['embed_dim'], num_heads, param['unit_1'])
    x = transformer_block(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(param['dropout'])(x)
    x = layers.BatchNormalization()(x)
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
