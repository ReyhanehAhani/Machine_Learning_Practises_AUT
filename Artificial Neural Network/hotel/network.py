import pandas as pd

from keras.models import load_model

from sklearn.preprocessing import MinMaxScaler

test = pd.read_csv('./H1.csv')

categorical = ['IsCanceled', 'ArrivalDateYear', 'ArrivalDateMonth', 'Meal', 'Country', 'MarketSegment', 'DistributionChannel', 'IsRepeatedGuest', 'ReservedRoomType', 'AssignedRoomType', 'DepositType', 'CustomerType', 'ReservationStatus']

test[categorical] = test[categorical].astype('category')
test[categorical] = test[categorical].apply(lambda x: x.cat.codes)

scaler = MinMaxScaler()

test_x = scaler.fit_transform(test[test.columns.drop(['ADR'])])
test_y = test['ADR'].to_numpy()

model = load_model('model.h5')
y_pred = model.predict(test_x)


import csv
with open('results.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)

    writer.writerow(['pred', 'act', 'diff'])
   
    for i in range(y_pred.shape[0]):
        pred = y_pred[i][0]
        act = test_y[i]
        diff = pred - act
        writer.writerow([pred, act, diff])
