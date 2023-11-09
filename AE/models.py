RSEED = 42

import numpy as np 


from keras.layers import Dense, GRU, BatchNormalization, Dropout, Input, RepeatVector, TimeDistributed
from tensorflow.keras import regularizers
from tensorflow import keras



import tensorflow as tf


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
keras.backend.clear_session() 
tf.random.set_seed(RSEED)
np.random.seed(RSEED)


print('In  models file')


def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    # y_true: true labels (binary), y_pred: predicted probabilities
    p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
    loss = -alpha * (1 - p_t)**gamma * tf.math.log(tf.clip_by_value(p_t, 1e-7, 1.0))
    return tf.reduce_mean(loss)


def build_model(feature_array, ae, hp):

    if feature_array.ndim == 3:
        n_features = feature_array.shape[2]
    else:
        n_features = feature_array.shape[1]

    hp_units = hp.Int('units', min_value=8, max_value=n_features, step=8)
    hp_units2 = hp.Int('units2', min_value=8, max_value=hp_units, step=8)
    hp_dropout = hp.Float('dropout', min_value=0.2, max_value=0.8, step=0.1)
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4, 1e-5])

    autoencoder, encoder = ae(feature_array, hp_units, hp_units2, hp_dropout, hp_learning_rate)

    return autoencoder


# def AutoencoderStatic(feature_array, hp_units, hp_units2, hp_dropout, lr):
#     input =  keras.layers.Input(shape=(feature_array.shape[1],), name='main_input')
#     encoded = Dense(hp_units, activation=tf.keras.layers.PReLU(),  kernel_initializer='he_normal', activity_regularizer=keras.regularizers.l1(10e-5))(input)
#     #
#     encoded = BatchNormalization()(encoded)
#     encoded = Dropout(hp_dropout/4)(encoded)
#     encoded = Dense(4, kernel_initializer='he_normal', activation=tf.keras.layers.PReLU())(encoded)
#     decoded = Dense(hp_units,kernel_initializer='he_normal', activation=tf.keras.layers.PReLU())(encoded)
#     decoded = BatchNormalization()(decoded)  # Adding BatchNormalization
#     decoded = Dropout(hp_dropout/2)(decoded)  # Adding Dropout
#     decoded = Dense(feature_array.shape[1], kernel_initializer='he_normal', activation='sigmoid')(decoded)  # Sigmoid for binary data

#     autoencoder = keras.Model(input, decoded)
#     encoder = keras.Model(input, encoded)

#     optimizer = keras.optimizers.Adam(lr=lr)
#     #loss =  tf.keras.losses.SigmoidCrossEntropy()

#     autoencoder.compile(optimizer=optimizer, loss='huber_loss')
#     #'binary_crossentropy'



#     return autoencoder, encoder



def AutoencoderStatic(feature_array, hp_units, hp_units2, hp_dropout, lr, pretrained_weights=None):
    input = tf.keras.layers.Input(shape=(feature_array.shape[1],), name='main_input')

    # Define the encoder layers with PReLU activation
    encoder = tf.keras.layers.Dense(hp_units, activation=tf.keras.layers.PReLU(), kernel_initializer='he_normal', activity_regularizer=tf.keras.regularizers.l1(10e-5))(input)
    encoder = tf.keras.layers.BatchNormalization()(encoder)
    encoder = tf.keras.layers.Dropout(hp_dropout/4)(encoder)
    encoder = tf.keras.layers.Dense(4, activation=tf.keras.layers.PReLU(), kernel_initializer='he_normal')(encoder)

    # Define the decoder layers with PReLU activation
    decoder = tf.keras.layers.Dense(hp_units, activation=tf.keras.layers.PReLU(), kernel_initializer='he_normal')(encoder)
    decoder = tf.keras.layers.BatchNormalization()(decoder)
    decoder = tf.keras.layers.Dropout(hp_dropout/2)(decoder)
    decoder = tf.keras.layers.Dense(feature_array.shape[1], activation='sigmoid', kernel_initializer='he_normal')(decoder)

    autoencoder = tf.keras.models.Model(input, decoder)

    # Load pretrained weights if provided
    if pretrained_weights is not None:
        autoencoder.load_weights(pretrained_weights)

    # Compile the autoencoder with Huber loss
    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='huber_loss')

    encoder = tf.keras.models.Model(input, encoder)

    return autoencoder, encoder


def AutoencoderGRU(feature_array, hp_units, hp_units2, hp_dropout, lr, pretrained_weights=None):

    n_timesteps = feature_array.shape[1]
    n_features = feature_array.shape[2]
    input = keras.Input(shape=(n_timesteps, n_features))
    encoded = GRU(n_features, activation="tanh", dropout=hp_dropout, return_sequences=True)(input)
    encoded = GRU(hp_units, activation="tanh", dropout=hp_dropout/2, return_sequences=True)(encoded)
    encoded = GRU(hp_units2, activation="tanh", dropout=hp_dropout/3, return_sequences=False)(encoded)

    decoded = RepeatVector(n_timesteps)(encoded)

    decoded = GRU(hp_units2, activation="tanh", return_sequences=True)(decoded)
    decoded = GRU(hp_units, activation="tanh", return_sequences=True)(decoded)
    decoded = GRU(n_features, activation="tanh", return_sequences=True)(decoded)
    decoded = TimeDistributed(Dense(n_features))(decoded)

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    autoencoder = keras.Model(input, decoded)
    encoder = keras.Model(input, encoded)
    # Calculate the Huber loss

    #Load pretrained weights if provided
    if pretrained_weights is not None:
        autoencoder.load_weights(pretrained_weights)

    autoencoder.compile(optimizer=optimizer, loss='huber_loss')
    #'mean_squared_error'

    return autoencoder, encoder


def build_model_2d(feature_array, hp):
    static_shape = feature_array['static_input'].shape[1]
    timesteps = feature_array['time_series_input'].shape[1]
    ts_features = feature_array['time_series_input'].shape[2]

    hp_neuros_st = hp.Int('hp_neuros_st', min_value=16, max_value=128, step=16)
    hp_dropout_rate_st = hp.Float('hp_dropout_rate_st', min_value=0.2, max_value=0.8, step=0.1)
    hp_neuros_time = hp.Int('hp_neuros_time', min_value=16, max_value=128, step=16)
    hp_dropout_rate_time = hp.Float('hp_dropout_rate_time', min_value=0.2, max_value=0.8, step=0.1)
    lr = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4, 1e-5])
    loss_weight_st = hp.Float('loss_weight_st', min_value=0.0, max_value=1.0)
    loss_weight_time = hp.Float('loss_weight_time', min_value=0.0, max_value=1.0)

    autoencoder, encoder = AutoenconderMultiModal_2d(
        feature_array,
        hp_neuros_st,
        hp_dropout_rate_st,
        hp_neuros_time,
        hp_dropout_rate_time,
        lr,
        loss_weight_st=loss_weight_st,
        loss_weight_time=loss_weight_time
    )

    return autoencoder


def build_model_2d(feature_array, hp):
    static_shape = feature_array['static_input'].shape[1]
    timesteps = feature_array['time_series_input'].shape[1]
    ts_features = feature_array['time_series_input'].shape[2]

    hp_neuros_st_1 = hp.Int('hp_neuros_st_1', min_value=16, max_value=128, step=8)
    # Define hp_neuros_st_2 within the desired range relative to hp_neuros_st_1
    hp_neuros_st_2 = hp.Int('hp_neuros_st_2', min_value=8, max_value=hp_neuros_st_1, step=8)
    hp_neuros_time_1 = hp.Int('hp_neuros_time_1', min_value=16, max_value=128, step=8)
    hp_neuros_time_2 = hp.Int('hp_neuros_time_2', min_value=8, max_value=hp_neuros_time_1, step=8)

    hp_dropout_rate_st = hp.Float('hp_dropout_rate_st', min_value=0.2, max_value=0.8, step=0.1)
    hp_dropout_rate_time = hp.Float('hp_dropout_rate_time', min_value=0.2, max_value=0.8, step=0.1)
    lr = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4, 1e-5])
    loss_weight_st = hp.Float('loss_weight_st', min_value=0.0, max_value=1.0)
    loss_weight_time = hp.Float('loss_weight_time', min_value=0.0, max_value=1.0)

    autoencoder, encoder = AutoenconderMultiModal_2d(
        feature_array,
        hp_neuros_st_1,
        hp_neuros_st_2,
        hp_neuros_time_1,
        hp_neuros_time_2,
        hp_dropout_rate_st,
        hp_dropout_rate_time,
        lr,
        loss_weight_st,
        loss_weight_time
    )

    return autoencoder




def AutoenconderMultiModal_2d(feature_array, hp_neuros_st_1, hp_neuros_st_2, hp_neuros_time_1, hp_neuros_time_2, hp_dropout_rate_st, hp_dropout_rate_time, lr, loss_weight_st, loss_weight_time, pretrained_weights=None):
    static_shape = feature_array['static_input'].shape[1]
    timesteps = feature_array['time_series_input'].shape[1]
    ts_features = feature_array['time_series_input'].shape[2]

    input_1 = Input(shape=(static_shape,), name="static_input")
    input_2 = Input(shape=(timesteps, ts_features), name="time_series_input")

    encoded_1 = build_static_encoder(input_1, hp_neuros_st_1,hp_neuros_st_2, hp_dropout_rate_st)
    decoded_1 = build_static_decoder(encoded_1, static_shape, hp_neuros_st_1, hp_neuros_st_2, hp_dropout_rate_st)  # Pass hp_neuros_st

    encoded_2 = build_time_series_encoder(input_2, ts_features, hp_neuros_time_1,hp_neuros_time_2, hp_dropout_rate_time)
    decoded_2 = build_time_series_decoder(encoded_2, timesteps, ts_features, hp_neuros_time_1,hp_neuros_time_2)  # Pass hp_neuros_time

    concatenated_encoded = keras.layers.Concatenate()([encoded_1, encoded_2])

    autoencoder = keras.Model(inputs=[input_1, input_2], outputs=[decoded_1, decoded_2])
    encoder = keras.Model(inputs=[input_1, input_2], outputs=concatenated_encoded)

    # loss_1 = 'binary_crossentropy'
    # loss_2 = 'mae'

    loss_1 = 'huber_loss'
    loss_2 = 'huber_loss'
    #huber_loss
    optimizer = keras.optimizers.Adam(learning_rate=lr)

    #Load pretrained weights if provided
    if pretrained_weights is not None:
        autoencoder.load_weights(pretrained_weights)


    autoencoder.compile(optimizer=optimizer, loss=[loss_1, loss_2], loss_weights=[0.5, 1])
    #loss_weight_st, loss_weight_time
    #0.05, 0.5

    return autoencoder, encoder

def build_static_encoder(input_layer, hp_neuros_st_1, hp_neuros_st_2, hp_dropout_rate_st):
    encoded = Dense(hp_neuros_st_1, kernel_initializer='he_normal', activation=tf.keras.layers.PReLU())(input_layer)
    encoded = BatchNormalization()(encoded)
    encoded = Dropout(hp_dropout_rate_st)(encoded)
    encoded = Dense(2, kernel_initializer='he_normal', activation=tf.keras.layers.PReLU())(encoded)
    return encoded

def build_static_decoder(encoded, static_shape, hp_neuros_st_1, hp_neuros_st_2, hp_dropout_rate_st):
    decoded = Dense(hp_neuros_st_1, kernel_initializer='he_normal', activation=tf.keras.layers.PReLU())(encoded)
    decoded = BatchNormalization()(decoded)
    decoded = Dropout(hp_dropout_rate_st)(decoded)
    decoded = Dense(static_shape, kernel_initializer='he_normal', activation='sigmoid', name='static')(decoded)
    return decoded

def build_time_series_encoder(input_layer, ts_features,hp_neuros_time_1,hp_neuros_time_2, hp_dropout_rate_time):
    encoded = GRU(ts_features, activation="tanh", dropout=hp_dropout_rate_time, return_sequences=True)(input_layer)
    encoded = GRU(hp_neuros_time_1, activation="tanh", dropout=hp_dropout_rate_time/2, return_sequences=True)(encoded)
    encoded = GRU(hp_neuros_time_2, activation="tanh", dropout=hp_dropout_rate_time/3, return_sequences=False)(encoded)
    return encoded

def build_time_series_decoder(encoded, timesteps, ts_features, hp_neuros_time_1,hp_neuros_time_2):
    decoded = RepeatVector(timesteps)(encoded)
    decoded = GRU(hp_neuros_time_2, activation="tanh", return_sequences=True)(decoded)
    decoded = GRU(hp_neuros_time_1, activation="tanh", return_sequences=True)(decoded)
    decoded = GRU(ts_features, activation="tanh", return_sequences=True)(decoded)
    decoded = TimeDistributed(Dense(ts_features), name='time_series')(decoded)
    return decoded

