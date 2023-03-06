from time_utils import *
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from data_analyst_result import *


def load_data(path='data/weblog.csv'):
    data = pd.read_csv(path)
    data.rename(columns={"URL": "log", "IP": "person_id", "Time": "time"}, inplace=True)
    data.drop(['Staus'], axis=1, inplace=True)  # we don't need that right now !
    # ------------------------------------------------------------------------
    shouldRemove = data['person_id'].value_counts()[data['person_id'].value_counts() < 100]  # removing Chert!! ids :))
    for i in range(len(shouldRemove)):
        data.drop(data.loc[data['person_id'] == shouldRemove.index[i]].index,
                  inplace=True)  # it Automatically deletes time missing values
    # ------------------------------------------------------------------------
    data['time'] = data['time'].apply(time_formatter)  # changing time format to smother one
    data['timestamp'] = data['time'].apply(to_time_stamp)  # time stamp for best sorting by time
    data.sort_values(by='timestamp', ascending=True, inplace=True)  # sorting by time
    data.drop(['timestamp'], axis=1, inplace=True)  # removing time stamp , we do not need that anymore !
    # ------------------------------------------------------------------------
    data.reset_index(inplace=True)  # reset index
    data.drop(['index'], axis=1, inplace=True)  # remove old index
    # ------------------------------------------------------------------------
    return data


def vector_to_one_hot(vec, max_len):
    oh = np.zeros((vec.size, max_len))
    oh[np.arange(vec.size), vec] = 1
    return oh


def matrix_to_one_hot(vec, work_week_type='work'):
    number_of_vectors = vec.shape[0]  # number of vectors
    len_of_vectors = vec.shape[1]  # len of each vector
    # len of one_hot matrix
    if work_week_type == 'week':
        lenOfOh = 8  # correspond to 7 days of week + 1 for null days
    else:
        lenOfOh = vec.max() + 1
    # ---------------------------------------------
    mat = np.zeros((len_of_vectors, lenOfOh))  # matrix !
    # ---------------------------------------------
    for n in range(number_of_vectors):
        mat = np.concatenate((mat, vector_to_one_hot(vec[n], lenOfOh)))
        # -----------------------------------------
    mat = mat[len_of_vectors:]
    mat = mat.reshape(number_of_vectors, len_of_vectors, lenOfOh)
    return mat


def xy(dataset, tx):
    """ dataset named workData as Input """
    line = dataset.shape[0]
    n_values = dataset.shape[1]
    dataset_len = line - tx + 1
    # ---------------------------------------------------
    X = np.zeros((dataset_len, tx, n_values))
    Y = np.zeros((dataset_len, tx, n_values))
    # ---------------------------------------------------
    for i in range(1, dataset_len):
        one_slice = np.expand_dims(dataset[i:i + tx], axis=0)
        X[i, :, :] = one_slice
    # ---------------------------------------------------
    for i in range(0, dataset_len - 1):
        one_slice = np.expand_dims(dataset[i:i + tx], axis=0)
        Y[i, :, :] = one_slice
    # ---------------------------------------------------
    X = X[1:, :, :]
    Y = Y[:-1, :, :]
    # ---------------------------------------------------------------------------------------------
    Y = np.swapaxes(Y, 0, 1)  # but why you should do this ? for (m,tx,n_values) to (Ty,m,n_values)
    return X, Y


def xy_small_model(dataset, tx):
    line = dataset.shape[0]
    n_values = dataset.shape[1]
    dataset_len = line - tx + 1
    # ---------------------------------------------------
    X = np.zeros((dataset_len - 1, tx, n_values))
    Y = np.zeros((dataset_len - 1, n_values))
    # ---------------------------------------------------
    for i in range(0, dataset_len - 1):
        one_slice = np.expand_dims(dataset[i:i + tx], axis=0)
        X[i, :, :] = one_slice
        Y[i] = dataset[i + tx]
    return X, Y


def time_serious_feature_adder(data):
    dayArray = data['time'].apply(to_days)
    uniqueDayArray = np.array(dayArray.unique())
    assigned_array = np.arange(0, len(uniqueDayArray))
    for i in range(len(uniqueDayArray)):
        dayArray[dayArray == uniqueDayArray[i]] = assigned_array[i]
    data['day'] = dayArray
    # -----------------------------------------------------
    # it takes a date and give us week_day! like Mon,Thu,Wed
    data['weekday'] = data['time'].apply(week_day)
    # -----------------------------------------------------
    data['timeSerious'] = data['time'].apply(to_time_space)
    # -----------------------------------------------------
    return data


def work_to_number(data):
    allWorks = np.array(data['log'].unique())
    # -----------------------------------------------------
    rangeVector = np.arange(1, len(allWorks) + 1)
    # -----------------------------------------------------
    dictionary = {}
    for i in range(len(allWorks)):
        dictionary[allWorks[i]] = rangeVector[i]

    # -----------------------------------------------------
    def work_to_number_apply(string):
        return dictionary[string]

    # -----------------------------------------------------
    data['work_to_number'] = data['log'].apply(work_to_number_apply)
    # -----------------------------------------------------
    data.drop('log', axis='columns', inplace=True)
    data.drop('time', axis='columns', inplace=True)
    # -----------------------------------------------------
    return data


def week_to_number(data):
    allWeeks = np.array(data['weekday'].unique())
    # -----------------------------------------------------
    rangeVector = np.arange(1, len(allWeeks) + 1)
    # -----------------------------------------------------
    WeeksDictionary = {}
    for i in range(len(allWeeks)):
        WeeksDictionary[allWeeks[i]] = rangeVector[i]

    # -----------------------------------------------------
    def week_to_number_apply(string):
        return WeeksDictionary[string]

    # -----------------------------------------------------
    data['week_to_number'] = data['weekday'].apply(week_to_number_apply)
    data.drop('weekday', axis='columns', inplace=True)
    # -----------------------------------------------------
    return data


def data_writer(data, Tx=30):
    Persons = data['person_id'].unique()  # array of all unique ids in other hand array of all different persons
    lenPersons = len(data['person_id'].unique())  # number of all unique ids
    n_values = max(data['work_to_number'])  # max_len of work numbers ...
    # ------------------------------------------------------------------
    for idx in range(lenPersons):
        currentId = Persons[idx]
        # -------------------------------------------------
        dataOfCurrentId = data.loc[data['person_id'] == currentId]
        workData = dataOfCurrentId['work_to_number'].to_numpy()
        weekData = dataOfCurrentId['week_to_number'].to_numpy()
        workData = tf.one_hot(workData, n_values + 1)
        weekData = tf.one_hot(weekData, 8)
        # -------------------------------------------------
        X_work, Y_work = xy(workData, Tx)
        X_week, Y_week = xy(weekData, Tx)
        # ----------------------------------------------------------------------------
        # writing part
        with open('data/model_data/X_work_id_' + str(currentId) + '.npy', 'wb') as f:
            np.save(f, X_work)
        print('writen in ' + 'data/model_data/X_work_id_' + str(currentId) + '.npy')
        # ----------------------------------------------------------------------------
        with open('data/model_data/Y_work_id_' + str(currentId) + '.npy', 'wb') as f:
            np.save(f, Y_work)
        print('writen in ' + 'data/model_data/Y_work_id_' + str(currentId) + '.npy')
        # ----------------------------------------------------------------------------
        with open('data/model_data/X_week_id_' + str(currentId) + '.npy', 'wb') as f:
            np.save(f, X_week)
        print('writen in ' + 'data/model_data/X_week_id_' + str(currentId) + '.npy')
        # ----------------------------------------------------------------------------
        with open('data/model_data/Y_week_id_' + str(currentId) + '.npy', 'wb') as f:
            np.save(f, Y_week)
        print('writen in ' + 'data/model_data/Y_week_id_' + str(currentId) + '.npy')


def small_data_writer(data, Tx=30):
    Persons = data['person_id'].unique()  # array of all unique ids in other hand array of all different persons
    lenPersons = len(data['person_id'].unique())  # number of all unique ids
    n_values = max(data['work_to_number'])  # max_len of work numbers ...
    # ------------------------------------------------------------------
    for idx in range(lenPersons):
        currentId = Persons[idx]
        # -------------------------------------------------
        dataOfCurrentId = data.loc[data['person_id'] == currentId]
        workData = dataOfCurrentId['work_to_number'].to_numpy()
        weekData = dataOfCurrentId['week_to_number'].to_numpy()
        workData = tf.one_hot(workData, n_values + 1)
        weekData = tf.one_hot(weekData, 8)
        # -------------------------------------------------
        X_work, Y_work = xy_small_model(workData, Tx)
        X_week, Y_week = xy_small_model(weekData, Tx)
        # ----------------------------------------------------------------------------
        # writing part
        with open('data/smallModel_data/X_work_id_' + str(currentId) + '.npy', 'wb') as f:
            np.save(f, X_work)
        print('writen in ' + 'data/smallModel_data/X_work_id_' + str(currentId) + '.npy')
        # ----------------------------------------------------------------------------
        with open('data/smallModel_data/Y_work_id_' + str(currentId) + '.npy', 'wb') as f:
            np.save(f, Y_work)
        print('writen in ' + 'data/smallModel_data/Y_work_id_' + str(currentId) + '.npy')
        # ----------------------------------------------------------------------------
        with open('data/smallModel_data/X_week_id_' + str(currentId) + '.npy', 'wb') as f:
            np.save(f, X_week)
        print('writen in ' + 'data/smallModel_data/X_week_id_' + str(currentId) + '.npy')
        # ----------------------------------------------------------------------------
        with open('data/smallModel_data/Y_week_id_' + str(currentId) + '.npy', 'wb') as f:
            np.save(f, Y_week)
        print('writen in ' + 'data/smallModel_data/Y_week_id_' + str(currentId) + '.npy')


def reader(PersonId):
    with open('data/model_data/X_work_id_' + str(PersonId) + '.npy', 'rb') as f:
        X_work = np.load(f)
    print('saved as X_work')
    # ----------------------------------------------------------------------------
    with open('data/model_data/Y_work_id_' + str(PersonId) + '.npy', 'rb') as f:
        Y_work = np.load(f)
    print('saved as Y_work')
    # ----------------------------------------------------------------------------
    with open('data/model_data/X_week_id_' + str(PersonId) + '.npy', 'rb') as f:
        X_week = np.load(f)
    print('saved as X_week')
    # ----------------------------------------------------------------------------
    with open('data/model_data/Y_week_id_' + str(PersonId) + '.npy', 'rb') as f:
        Y_week = np.load(f)
    print('saved as Y_week')
    # ------------------------------------
    print('X_work.shape =>', X_work.shape)
    print('Y_work.shape =>', Y_work.shape)
    print('X_week.shape =>', X_week.shape)
    print('Y_week.shape =>', Y_week.shape)
    # ------------------------------------
    return X_work, Y_work, X_week, Y_week


def small_reader(PersonId):
    with open('data/smallModel_data/X_work_id_' + str(PersonId) + '.npy', 'rb') as f:
        X_work = np.load(f)
    print('saved as X_work')
    # ----------------------------------------------------------------------------
    with open('data/smallModel_data/Y_work_id_' + str(PersonId) + '.npy', 'rb') as f:
        Y_work = np.load(f)
    print('saved as Y_work')
    # ----------------------------------------------------------------------------
    with open('data/smallModel_data/X_week_id_' + str(PersonId) + '.npy', 'rb') as f:
        X_week = np.load(f)
    print('saved as X_week')
    # ----------------------------------------------------------------------------
    with open('data/smallModel_data/Y_week_id_' + str(PersonId) + '.npy', 'rb') as f:
        Y_week = np.load(f)
    print('saved as Y_week')
    # ------------------------------------
    print('X_work.shape =>', X_work.shape)
    print('Y_work.shape =>', Y_work.shape)
    print('X_week.shape =>', X_week.shape)
    print('Y_week.shape =>', Y_week.shape)
    # ------------------------------------
    return X_work, Y_work, X_week, Y_week


def writer(tx=30):
    """preprocessing and writing"""
    data = load_data()
    data = time_serious_feature_adder(data)
    data = work_to_number(data)
    data = week_to_number(data)
    data = outlier_remover(data)
    data_writer(data, tx)


def small_writer(tx=30):
    """preprocessing and writing"""
    data = load_data()
    data = time_serious_feature_adder(data)
    data = work_to_number(data)
    data = week_to_number(data)
    data = outlier_remover(data)
    small_data_writer(data, tx)
