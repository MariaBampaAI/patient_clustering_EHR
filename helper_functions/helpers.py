import pandas as pd, numpy as np, datetime as dt, matplotlib.pyplot as plt 
import tensorflow as tf
import keras

def categorize_age(age):

    """
    Categorize age into predefined groups.

    This function categorizes a person's age into predefined groups based on common age groupings.
    Age is categorized into three groups: 'youth' for ages less than 25, 'adults' for ages between 25 and 64,
    and 'seniors' for ages greater than or equal to 65.

    Args:
        age (int or float): The age of the person to be categorized.

    Returns:
        str: The age group category to which the age belongs.

    Example:
        age_group = categorize_age(30)
        # Result: 'adults'
    """


    #from: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3825015/
    if  age < 25: 
        cat = 'youth'
    elif age >= 25 and age <= 64:
        cat = 'adults'
    elif age > 64 :
        cat = 'seniors'

    return cat


def simple_imputer(df, ID_COLS):
    """
    Simple imputation for missing values in a DataFrame.

    This function performs a simple imputation of missing values in a DataFrame.
    It fills missing values with the last available value within each group defined by 'ID_COLS',
    and then fills remaining missing values with the mean of each group.

    Args:
        df (pandas.DataFrame): The input DataFrame with missing values.
        ID_COLS (list): A list of column names used to group data for imputation.

    Returns:
        pandas.DataFrame: The DataFrame with missing values imputed using the specified strategy.

    Example:
        df = pd.DataFrame({'PatientID': [1, 1, 2, 2, 3], 'Value': [10, np.nan, 20, 30, np.nan]})
        imputed_df = simple_imputer(df, ['PatientID'])
    """
    df_out = df.copy()
    icustay_means = df_out.groupby(ID_COLS).mean()
    
    df_out = df_out.groupby(ID_COLS).fillna(
        method='ffill'
    ).groupby(ID_COLS).fillna(icustay_means)
    #.fillna(-1)
    
    df_out.sort_index(axis=1, inplace=True)
    return df_out


# Define a function to reindex each patient's data with a complete range of hours
def fill_missing_hours(group):

    """
    Reindex patient's data with a complete range of hours.

    This function reindexes a patient's data to include a complete range of hours (0 to 23) and fills
    missing hours with NaN values. It ensures that each patient's data covers all 24 hours.

    Args:
        group (pandas.DataFrame): A DataFrame containing patient data with an 'hours_in' column.

    Returns:
        pandas.DataFrame: A DataFrame with a complete range of hours (0 to 23) and filled with data or NaN values.

    Example:
        patient_data = pd.DataFrame({"hours_in": [0, 2, 3, 5], "value": [10, 20, 30, 40]})
        complete_data = fill_missing_hours(patient_data)
    """

    complete_range = pd.DataFrame({"hours_in": range(24)})
    return complete_range.merge(group, on="hours_in", how="left")


def plot_model_loss(val, train, model,path_results):
    """
    Plot and save the loss curve for a model.

    This function plots and saves the loss curve for a model during training. It displays both the training
    and validation loss over epochs.

    Args:
        val (list): A list of validation loss values over epochs.
        train (list): A list of training loss values over epochs.
        model (str): The name or identifier of the model.
        path_results (str): The directory path where the loss curve image will be saved.

    Example:
        train_loss = [0.1, 0.08, 0.06, 0.05]
        val_loss = [0.12, 0.1, 0.09, 0.08]
        plot_model_loss(val_loss, train_loss, 'my_model', '/path/to/results')
    """
    import matplotlib.pyplot as plt 
    plt.plot(train)
    plt.plot(val)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(f'{path_results}/loss_{model}.png')
    plt.clf()
    plt.cla()
    plt.close()


def create_dicts_MultiModal(static, timeseries):
    """
    Create dictionaries for static and time series data as TensorFlow tensors.

    This function takes static and time series data and converts them into TensorFlow tensors.
    It then stores them in two dictionaries, 'dataset' and 'dataset2', with specific keys.

    Args:
        static (numpy.ndarray): A 2D numpy array representing static data (samples, features).
        timeseries (numpy.ndarray): A 3D numpy array representing time series data (samples, timesteps, features).

    Returns:
        tuple: A tuple containing two dictionaries:
            - dataset (dict): Contains TensorFlow tensors for static and time series data.
            - dataset2 (dict): An alternative dictionary with keys 'static' and 'time_series'.

    Example:
        static_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        timeseries_data = np.array([[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], [[0.7, 0.8], [0.9, 1.0], [1.1, 1.2]]])
        dataset, dataset2 = create_dicts_MultiModal(static_data, timeseries_data)
    """
    dataset = {} 

    dataset['static_input'] = static
    dataset['time_series_input'] = timeseries



    st_train_tensor = tf.convert_to_tensor(dataset['static_input'], name='static')
    ts1_train_tensor = tf.convert_to_tensor(dataset['time_series_input'], name='time_series')

    dataset = {} 
    dataset2 = {}
    

    dataset['static_input'] = st_train_tensor
    dataset['time_series_input'] = ts1_train_tensor

    dataset2['static'] = st_train_tensor 
    dataset2['time_series'] = ts1_train_tensor
    

    return dataset, dataset2


def load_encoder_model(model_path, input_data):
    """
    Load a pre-trained encoder model from the specified path and use it to generate latent representations.

    Args:
        model_path (str): The path to the pre-trained encoder model file.
        input_data (numpy.ndarray or tf.Tensor): Input data for which latent representations will be generated.

    Returns:
        numpy.ndarray: Latent representations generated by the encoder model.

    Note:
        This function loads a pre-trained encoder model from the given file path and applies it to the input data
        to produce latent representations. It then clears the TensorFlow session to release resources.

    Example:
        latent_representations = load_encoder_model('path/to/encoder_model.h5', input_data)
    """

    encoder_model = tf.keras.models.load_model(model_path)
    encoder = keras.Model(encoder_model.input, encoder_model.layers[-1].output)
    latent_output = encoder.predict(input_data)
    tf.keras.backend.clear_session()
    return latent_output

def load_encoded_spaces(time_series, static, model_run, path):
    """
    Load pre-trained encoder models and generate latent representations for time series, static data, and MultiModal data.

    Args:
        time_series (tf.Tensor or numpy.ndarray): Time series data for which latent representation will be generated.
        static (pd.DataFrame or numpy.ndarray): Static data for which latent representation will be generated.
        model_run (int): The run/model number.
        path (str): The base path for saved encoder models.

    Returns:
        tuple: A tuple containing latent representations for time series, static, and MultiModal data.

    Note:
        This function loads pre-trained encoder models for static, time series, and MultiModal data, and then uses
        these models to generate latent representations. The generated latent representations are returned as a tuple
        (latent_ts, latent_st, latent_mm).

    Example:
        latent_ts, latent_st, latent_mm = load_encoded_spaces(time_series_data, static_data, model_run=1, path='path/to/models')
    """

    
    static_encoder_path = f'{path}/static_encoder_model_run_{model_run}.h5'
    mm_encoder_path = f'{path}/MultiModal_2d_encoder_model_run_{model_run}.h5'
    ts_encoder_path = f'{path}/time_series_encoder_model_run_{model_run}.h5'

    latent_st = load_encoder_model(static_encoder_path, static.to_numpy())
    print('STATIC AE SHAPE:', latent_st.shape)


    latent_ts = load_encoder_model(ts_encoder_path, time_series)
    print('GRU SHAPE:', latent_ts.shape)

    
    dataset, dataset2 = create_dicts_MultiModal(static, time_series)

    latent_mm = load_encoder_model(mm_encoder_path, dataset)
    print('MM SHAPE:', latent_mm.shape)


    return latent_ts, latent_st, latent_mm
