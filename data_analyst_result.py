import pandas as pd


def outlier_remover(data):
    data.drop(data.loc[(data['person_id'] == '10.130.2.1') & (data['day'] == 40)].index, inplace=True)
    data.drop(data.loc[(data['person_id'] == '10.130.2.1') & (data['day'] == 21)].index, inplace=True)
    # it seems that you should delete day 21th for person_id :'10.129.2.1' !!
    data.drop(data.loc[(data['person_id'] == '10.129.2.1') & (data['day'] == 21)].index, inplace=True)
    # it seems that you should delete days 40th and 21th for person_id :'10.128.2.1' !!
    data.drop(data.loc[(data['person_id'] == '10.128.2.1') & (data['day'] == 40)].index, inplace=True)
    data.drop(data.loc[(data['person_id'] == '10.128.2.1') & (data['day'] == 21)].index, inplace=True)
    # it seems that you should delete days 40th and 21th for person_id :'10.128.2.1' !!
    data.drop(data.loc[(data['person_id'] == '10.128.2.1') & (data['day'] == 40)].index, inplace=True)
    data.drop(data.loc[(data['person_id'] == '10.128.2.1') & (data['day'] == 21)].index, inplace=True)
    # it seems that you should delete days 40th and 21th for person_id :'10.128.2.1' !!
    data.drop(data.loc[(data['person_id'] == '10.131.0.1') & (data['day'] == 40)].index, inplace=True)
    data.drop(data.loc[(data['person_id'] == '10.131.0.1') & (data['day'] == 21)].index, inplace=True)
    # it seems that you should delete day 21th for person_id :'10.129.2.1' !!
    data.drop(data.loc[(data['person_id'] == '10.131.2.1') & (data['day'] == 21)].index, inplace=True)
    # -----------------------------------------------------------------------------------
    return data
