import datetime
from datetime import datetime
from datetime import timedelta
import time
import random
import numpy as np
import warnings
import pandas as pd
warnings.simplefilter(action='ignore')


# time sampling for a person that she started her job at 09/01/2022 and working 9 hours a day with some tasks
# ## Helper Functions for time sampling


def to_time_space(string, start='00:00:00'):
    # string work_week_type is  "%m/%d/%Y %H:%M:%S"
    start_minutes = datetime.strptime(start, '%H:%M:%S').minute + datetime.strptime(start, '%H:%M:%S').hour * 60
    splitted = string.split()
    hour = splitted[1]
    splitted = hour.split(':')
    hour = splitted[0]
    minute = splitted[1]
    # score = 60 * (int(splitted[0]) - datetime.strptime(start, '%H:%M:%S').hour ) + int(splitted[1])
    score = (60 * int(hour) + int(minute)) - start_minutes
    return score


def id_generator(num):
    bias = 1000
    array = np.arange(0, num) + bias
    array = array.astype(int)
    return array


def clock(string, worksPerDay=9, start='09:00:00', stop='18:00:00'):
    # -----------------------------------------------------------------
    datetime_object = datetime.strptime(string, '%m/%d/%Y %H:%M:%S')
    # -----------------------------------------------------------------
    stop_minutes = datetime.strptime(stop, '%H:%M:%S').minute + datetime.strptime(stop, '%H:%M:%S').hour * 60
    start_minutes = datetime.strptime(start, '%H:%M:%S').minute + datetime.strptime(start, '%H:%M:%S').hour * 60
    # -----------------------------------------------------------------
    rdn = random.randint(1, int((stop_minutes - start_minutes) / worksPerDay))
    value = datetime_object + timedelta(minutes=rdn)
    if value.hour >= datetime.strptime(stop, '%H:%M:%S').hour:
        value = value - timedelta(minutes=value.minute)
        value = value + timedelta(hours=15)
        datetime_object = value
    else:
        datetime_object = value

    if value.strftime('%a') == 'Fri':
        datetime_object = datetime_object - timedelta(minutes=value.minute)
        datetime_object = datetime_object + timedelta(days=1)
        # -----------------------------------------------------------------
    return datetime_object.strftime("%m/%d/%Y %H:%M:%S")


# dictionary contains all paths that you can add or remove
dirs = list({'/Desktop/folder9',
             '/Desktop/folder8',
             '/Desktop/folder1',
             '/Desktop/folder2',
             '/Desktop/folder3',
             '/Desktop/folder4',
             '/Desktop/folder5',
             '/Desktop/folder6',
             '/Desktop/data/folder7'})
print('dictionary contains all paths ==>')
# random choose of works ...
print(dirs[random.randint(0, len(dirs) - 1)].split("/")[-1])


def sampling(persons, worksPerDay, lenOfData):
    data = pd.DataFrame({'person_id': [0], 'time': [0], 'log': ['/']})
    ids = iter(id_generator(lenOfData))
    # -----------------------------------------------------
    for i in range(persons):
        worksPerDay = int(np.random.normal(worksPerDay, 0.3, 1))
        init = '09/01/2022 09:00:00'
        thisId = next(ids)
        df = pd.DataFrame(
            {'person_id': thisId, 'time': [init], 'log': dirs[random.randint(0, len(dirs) - 1)].split("/")[-1]}
        )
        for j in range(int(lenOfData / persons)):
            init = clock(init, worksPerDay)
            df_new = pd.DataFrame(
                {'person_id': thisId, 'time': [init], 'log': dirs[random.randint(0, len(dirs) - 1)].split("/")[-1]})
            df = pd.concat([df, df_new], ignore_index=True)
        data = pd.concat([data, df], ignore_index=True)
    # -----------------------------------------------------
    # dropping first row and re indexing
    data.drop(index=0, inplace=True)
    data.index = data.index - 1

    data.sort_values(by='time', inplace=True)
    data.index = np.arange(0, len(data))

    for i in range(len(data) - lenOfData):
        data.drop(index=(len(data) - 1), inplace=True)

    # saving data to file in csv
    data.to_csv('data.csv', index=True)
    return data


print(sampling(persons=10, worksPerDay=12, lenOfData=600))
