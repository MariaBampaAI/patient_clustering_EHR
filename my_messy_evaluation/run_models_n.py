RSEED = 42
import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
from functools import partial
import keras_tuner
from tensorflow import keras
#import absl.logging

from kerastuner.tuners import RandomSearch
from tqdm import tqdm




# Append relevant paths to sys.path
sys.path.append('/Users/maba4574/Desktop/Work/Projects/MIMIC/patient_clustering_EHR/AE')
import models
sys.path.append('/Users/maba4574/Desktop/Work/Projects/MIMIC/patient_clustering_EHR/helper_functions')
import helpers

# Set the checkpoint path
checkpoint_path = "AE/saved_models/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Set random seeds
tf.random.set_seed(RSEED)
np.random.seed(RSEED)

# Define a learning rate scheduler
def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * 0.1
    #tf.math.exp(-0.1)

callbacks = [
    keras.callbacks.TensorBoard(
        log_dir='my_log_dir',
        histogram_freq=1,
        embeddings_freq=1,
    ),
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True
    ),
    #keras.callbacks.LearningRateScheduler(scheduler),
    keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_loss',
        save_best_only=True
    )
]

print('In run_models python file')




def represent(AEmodels, modalities, cohort, datasets, model_run, epochs=100, path_results='RESULTS', pretrain=True):
    """
    Represent the data using multiple autoencoders.

    Args:
        AEmodels (list of functions): A list of autoencoder model functions.
        modalities (list of str): Names of the modalities (e.g., 'static', 'time_series').
        cohort (str): The application problem (e.g., 'sepsis' or 'cardiovascular').
        datasets (list): A list of datasets for different modalities.
        model_run (int): The current model run number.
        epochs (int): Number of training epochs (default is 100).
        path_results (str): Path to store the results (default is 'MAY_RESULTS').

    Returns:
        Tuple: Encoded representations for different modalities.
    """
    # if not os.path.isdir(path_results):
    #     os.makedirs(path_results)

    saved_models_dir = os.path.join(path_results, 'saved_models_avg')
    losses_dir = os.path.join(path_results, 'losses_avg')

    # for dir_path in [saved_models_dir, losses_dir]:
    #     if not os.path.isdir(dir_path):
    #         os.makedirs(dir_path)
    print(f'[INFO] Training modality: {modalities[2]}')

    tf.keras.backend.clear_session()

    # Hyperparameter tuning for MMAE
    
    partial_model = partial(models.build_model_2d, datasets[2])

    tuner = RandomSearch(
        hypermodel=partial_model,
        objective='val_loss',
        max_trials=30,
        executions_per_trial=5,
        directory=os.path.join(path_results, f'hp_dir_2_{cohort}'),
        project_name=f'hp_{modalities[2]}',
        overwrite=False
    )

    tuner.search(datasets[2], datasets[3], validation_split=0.1, epochs=10)

    best_hps = tuner.get_best_hyperparameters()[0]
    best_hps_values = best_hps.values
    print(best_hps.values)
    pd.DataFrame.from_dict(data=best_hps_values, orient='index').to_csv(f'{path_results}/hp_{cohort}_{modalities[2]}.csv', header=False)
    
    
    mm_autoencoder_path = f'{saved_models_dir}/{modalities[2]}_model_run_{model_run}.h5'

    autoencoder_mm, encoder_mm = AEmodels[2](
        feature_array=datasets[2],
        hp_neuros_st_1=best_hps['hp_neuros_st_1'],
        hp_neuros_st_2=best_hps['hp_neuros_st_2'],
        hp_neuros_time_1=best_hps['hp_neuros_time_1'],
        hp_neuros_time_2=best_hps['hp_neuros_time_2'],
        hp_dropout_rate_st=best_hps['hp_dropout_rate_st'],
        hp_dropout_rate_time=best_hps['hp_dropout_rate_time'],
        lr=best_hps['learning_rate'],
        loss_weight_st=best_hps['loss_weight_st'],
        loss_weight_time=best_hps['loss_weight_time'],
        #pretrained_weights = mm_autoencoder_path
    )

    history = autoencoder_mm.fit(
        datasets[2],
        datasets[3],
        epochs=epochs,
        batch_size=32,
        #32 angus
        verbose=1,
        shuffle=False,
        validation_split=0.2,
        callbacks=[callbacks]
    )

    if pretrain==False:
        autoencoder_mm.save(f'{saved_models_dir}/{modalities[2]}_model_run_{model_run}.h5')
        encoder_mm.save(f'{saved_models_dir}/{modalities[2]}_encoder_model_run_{model_run}.h5')
    

        ae_val_loss = history.history['val_loss']
        ae_train_loss = history.history['loss']

        title = f'{modalities[2]}_run_{model_run}'
        helpers.plot_model_loss(ae_val_loss, ae_train_loss, title, losses_dir)
    else:
        autoencoder_mm.save(f'{saved_models_dir}/{modalities[2]}_model_run_{model_run}_PT.h5')
        encoder_mm.save(f'{saved_models_dir}/{modalities[2]}_encoder_model_run_{model_run}_PT.h5')
        ae_val_loss = history.history['val_loss']
        ae_train_loss = history.history['loss']

        title = f'{modalities[2]}_run_{model_run}_PT'
        helpers.plot_model_loss(ae_val_loss, ae_train_loss, title, losses_dir)      



    encoded_mm = encoder_mm.predict(datasets[2])
    print('latent shape', encoded_mm.shape)
    print(f'[INFO] Training modality: {modalities[0]}')

    tf.keras.backend.clear_session()

    # Hyperparameter tuning for static modality
    partial_model = partial(models.build_model, datasets[0], AEmodels[0])
    tuner = RandomSearch(
        hypermodel=partial_model,
        objective='val_loss',
        max_trials=100,
        executions_per_trial=10,
        directory=os.path.join(path_results, f'hp_dir_2_{cohort}'),
        project_name=f'hp_{modalities[0]}',
        overwrite=False
    )

    tuner.search(datasets[0], datasets[0], validation_split=0.1, epochs=10)

    best_hps = tuner.get_best_hyperparameters()[0]
    best_hps_values = best_hps.values
    print(best_hps.values)
    pd.DataFrame.from_dict(data=best_hps_values, orient='index').to_csv(f'{path_results}/hp_{cohort}_{modalities[0]}.csv', header=False)

    static_autoencoder_path = f'{saved_models_dir}/{modalities[0]}_model_run_{model_run}.h5'
    static_autoencoder, static_encoder = AEmodels[0](
        feature_array=datasets[0],
        hp_units=best_hps['units'],
        hp_units2=best_hps['units2'],
        hp_dropout=best_hps['dropout'],
        #lr=best_hps['learning_rate'],
        lr=best_hps['learning_rate'],
        #pretrained_weights = static_autoencoder_path
    )
    
    history = static_autoencoder.fit(
        datasets[0],
        datasets[0],
        epochs=epochs,
        batch_size=128,
        verbose=1,
        shuffle=False,
        validation_split=0.2,
        callbacks=[callbacks]
    )



    if pretrain==False:
        static_autoencoder.save(f'{saved_models_dir}/{modalities[0]}_model_run_{model_run}.h5')
        static_encoder.save(f'{saved_models_dir}/{modalities[0]}_encoder_model_run_{model_run}.h5')
    

        ae_val_loss = history.history['val_loss']
        ae_train_loss = history.history['loss']

        title = f'{modalities[0]}_run_{model_run}'
        helpers.plot_model_loss(ae_val_loss, ae_train_loss, title, losses_dir)
    else:
        static_autoencoder.save(f'{saved_models_dir}/{modalities[0]}_model_run_{model_run}_PT.h5')
        static_encoder.save(f'{saved_models_dir}/{modalities[0]}_encoder_model_run_{model_run}_PT.h5')
        ae_val_loss = history.history['val_loss']
        ae_train_loss = history.history['loss']

        title = f'{modalities[0]}_run_{model_run}_PT'
        helpers.plot_model_loss(ae_val_loss, ae_train_loss, title, losses_dir)     


    encoded_static = static_encoder.predict(datasets[0])

    print(f'[INFO] Training modality: {modalities[1]}')

    # Train the model
    tf.keras.backend.clear_session()

    partial_model = partial(models.build_model, datasets[1], AEmodels[1])

    tuner = keras_tuner.RandomSearch(
            hypermodel=partial_model,
            objective='val_loss',
            max_trials=15,
            executions_per_trial=10,
            directory=f'{path_results}/hp_dir_2_{cohort}',
            project_name=f'hp_{modalities[1]}',
            overwrite=False
        )
    tuner.search(datasets[1], datasets[1], validation_split=0.1, epochs=10)


    # Get the hyperparameters.
    best_hps = tuner.get_best_hyperparameters()[0]
    # Build the model with the best hp.
    print(best_hps.values)
    (pd.DataFrame.from_dict(data=best_hps.values, orient='index').to_csv(f'{path_results}/hp_{cohort}_{modalities[1]}.csv', header=False))
    
    time_series_autoencoder_path = f'{saved_models_dir}/{modalities[1]}_model_run_{model_run}.h5'
    
    autoencoder_time_series, encoder_time_series = AEmodels[1](
        feature_array=datasets[1],  
        hp_units =best_hps['units'], 
        hp_units2=best_hps['units2'], 
        hp_dropout=best_hps['dropout'], 
        lr=best_hps['learning_rate'],
        #pretrained_weights = time_series_autoencoder_path
        )
    
        


    history = autoencoder_time_series.fit(
        datasets[1], 
        datasets[1],
        epochs=epochs,
        batch_size=128,
        verbose=1,
        shuffle=False,     
        validation_split=0.2,
        callbacks=[callbacks]
        )


    if pretrain==False:
        autoencoder_time_series.save(f'{saved_models_dir}/{modalities[1]}_model_run_{model_run}.h5')
        encoder_time_series.save(f'{saved_models_dir}/{modalities[1]}_encoder_model_run_{model_run}.h5')
    

        ae_val_loss = history.history['val_loss']
        ae_train_loss = history.history['loss']

        title = f'{modalities[1]}_run_{model_run}'
        helpers.plot_model_loss(ae_val_loss, ae_train_loss, title, losses_dir)
    else:
        autoencoder_time_series.save(f'{saved_models_dir}/{modalities[1]}_model_run_{model_run}_PT.h5')
        encoder_time_series.save(f'{saved_models_dir}/{modalities[1]}_encoder_model_run_{model_run}_PT.h5')
        ae_val_loss = history.history['val_loss']
        ae_train_loss = history.history['loss']

        title = f'{modalities[1]}_run_{model_run}_PT'
        helpers.plot_model_loss(ae_val_loss, ae_train_loss, title, losses_dir)     

    encoded_time_series = encoder_time_series.predict(datasets[1])

    del static_autoencoder
    del static_encoder
    del tuner

    tf.keras.backend.clear_session()

    return encoded_static, encoded_time_series, encoded_mm


if __name__ == "__main__":
    # #sepsis3
    time_series_2d = np.load('preprocessing/representations/time_series_2d_scaled.npy')
    time_series_3d = np.load('preprocessing/representations/time_series_3d_scaled.npy')

    static = pd.read_csv('preprocessing/representations/static_eq.csv')
    time_series_2d_df = pd.read_csv('preprocessing/representations/time_series_2d_scaled_df.csv')
    static = static.set_index('icustay_id')
    static = static.loc[:, static.gt(0).mean() >= .1]  

    static = static[['vent', 'F', 'M', 'adults', 'seniors', 'race_white', 'race_other',
       'gcs_13_15', 'gcs_9_12', 'gcs_<8', 'congestive_heart_failure',
       'cardiac_arrhythmias', 'hypertension', 'other_neurological',
       'chronic_pulmonary',  'hypothyroidism',
       'renal_failure', 'liver_disease', 'coagulopathy', 'fluid_electrolyte',
       'alcohol_abuse', 'depression', 'diabetes', 'circulatory',
       'circulatory_infectious_congenital', 'nervous_and_sense',
       'endocrinal_nutritional', 'neoplasms', 'blood_and_blood_forming',
       'mental']]


    #angus
    # time_series_2d = np.load('preprocessing/representations/time_series_2d_scaled_angus.npy')
    # time_series_3d = np.load('preprocessing/representations/time_series_3d_scaled_angus.npy')

    # static = pd.read_csv('preprocessing/representations/static_eq_angus.csv')
    # time_series_2d_df = pd.read_csv('preprocessing/representations/time_series_2d_scaled_df_angus.csv')

    # static = static.set_index('icustay_id')
    # static = static.loc[:, static.gt(0).mean() >= .1]  

    # static = static[['vent', 'F', 'M', 'adults', 'seniors', 'race_white', 'gcs_13_15',
    #    'gcs_9_12', 'gcs_<8', 'congestive_heart_failure', 'cardiac_arrhythmias',
    #    'hypertension', 'other_neurological', 'chronic_pulmonary',
    #    'hypothyroidism', 'renal_failure',
    #    'liver_disease', 'coagulopathy', 'fluid_electrolyte', 'alcohol_abuse',
    #    'depression', 'diabetes', 'circulatory',
    #    'circulatory_infectious_congenital', 'nervous_and_sense',
    #    'endocrinal_nutritional', 'neoplasms', 'blood_and_blood_forming',
    #     'mental']]
    

    static = static.fillna(0)
    static = static.astype(float)  
    print('static shape after dropping empty values', static.shape)


    time_series_3d = time_series_3d.astype(np.float32)


    dataset, dataset2 = helpers.create_dicts_MultiModal(static.values, time_series_3d)
    
    path = f'results_EQ' 


    if not os.path.isdir(f"{path}/losses_avg"):

        os.makedirs(f"{path}/losses_avg")

    if not os.path.isdir(f"{path}/saved_models_avg"):

        os.makedirs(f"{path}/saved_models_avg")
    
    # Modify 'train' variable if needed
    train = True
    PT = False

    # Specify cohort
    cohort = 'sepsis3'

    # Run the models 10 times with different random seeds
    if train:
        RSEED=42
        num_runs = 10
        #all_encoded_static = []
        #all_encoded_time_series = []
        #all_encoded_mm = []

        for run in tqdm(range(10,20), desc="Training Autoencoders"):
            current_seed = RSEED + run  # Use a different seed for each run

            tf.random.set_seed(current_seed)
            np.random.seed(current_seed)
            print(run, current_seed)
            encoded_static, encoded_time_series, encoded_mm = \
            represent([models.AutoencoderStatic, models.AutoencoderGRU, models.AutoenconderMultiModal_2d],\
                                            ['static', 'time_series', 'MultiModal_2d', 'late_fusion'], \
                                                cohort, [static.values, time_series_3d, dataset, dataset2], \
                                                    f'latent_avg_{run}', epochs=300, path_results=path, pretrain=PT)

            # all_encoded_static.append(encoded_static)
            # all_encoded_time_series.append(encoded_time_series)
            # all_encoded_mm.append(encoded_mm)

        # Compute the average of the encoded representations
        # average_encoded_static = np.mean(all_encoded_static, axis=0)
        # average_encoded_time_series = np.mean(all_encoded_time_series, axis=0)
        # average_encoded_mm = np.mean(all_encoded_mm, axis=0)

    else:
        print('Models are trained. You can load the encoded spaces.')

