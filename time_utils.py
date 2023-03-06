# helper function for time processing named as time Utils

import datetime
from datetime import datetime, timedelta
import time
import numpy as np


def to_days(date, startDate='11/07/2017 9:00:00'):
    # it takes a date and give us how many mitunes away from everyday starting date
    date = datetime.strptime(date, '%m/%d/%Y %H:%M:%S')
    startDate = datetime.strptime(startDate, '%m/%d/%Y %H:%M:%S')
    return (date.year - startDate.year) * 366 + (date.month - startDate.month) * 31 + (date.day - startDate.day)


def week_day(datetime_str):
    # it takes a date and give us week_day! like Mon,Thu,Wed
    date = datetime.strptime(datetime_str, '%m/%d/%Y %H:%M:%S')
    return date.strftime("%a")


def time_formatter(string):
    # remove '[' and %d/%b/%Y:%H:%M:%S to %m/%d/%Y %H:%M:%S string to string
    try:
        string = string.replace('[', '')
        stringToDate = datetime.strptime(string, "%d/%b/%Y:%H:%M:%S")
        string = stringToDate.strftime('%m/%d/%Y %H:%M:%S')
        return string
    except:
        return '?'


def to_time_stamp(string):
    return time.mktime(datetime.strptime(string, "%m/%d/%Y %H:%M:%S").timetuple())


def to_time_space(string, start='00:00:00'):
    # string work_week_type is  "%m/%d/%Y %H:%M:%S"
    start_minutes = datetime.strptime(start, '%H:%M:%S').minute + datetime.strptime(start, '%H:%M:%S').hour * 60
    splited = string.split()
    hour = splited[1]
    splited = hour.split(':')
    hour = splited[0]
    minute = splited[1]
    # score = 60 * (int(splited[0]) - datetime.strptime(start, '%H:%M:%S').hour ) + int(splited[1])
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
