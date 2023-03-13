import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential
from keras.layers import BatchNormalization

def scale_volume(raw_value):
    if str(raw_value) == '': 
        return 0
    postfix = str(raw_value)[-1:]
    try:
        actual_volume = float(str(raw_value)[:-1])
    except:
        return 0
    multiplier = 1
    if postfix == 'K':
        multiplier = 1000
    elif postfix == 'M':
        multiplier = 1000000
    elif postfix == 'B':
        multiplier = 1000000000
    elif postfix.isnumeric():
        return int(raw_value)

    return int(actual_volume * multiplier)


def load_dataset():
    raw_data = pd.read_csv('ETH_USD_price_2019-2023.csv')
    print('Number of rows and columns:', raw_data.shape)

    raw_data['date'] = pd.to_datetime(raw_data['date'], format='%Y-%m-%d')
    raw_data['close'] = raw_data['close'].apply(lambda x: float(str(x).replace(',', '')))
    raw_data['open'] = raw_data['open'].apply(lambda x: float(str(x).replace(',', '')))
    raw_data['high'] = raw_data['high'].apply(lambda x: float(str(x).replace(',', '')))
    raw_data['low'] = raw_data['low'].apply(lambda x: float(str(x).replace(',', '')))
    raw_data['volume'] = raw_data['volume'].apply(lambda x: scale_volume(x))

    # 모델 정확성을 위한 normalize 처리
    number_data = raw_data.iloc[:, 1:6]

    min_num = np.min(number_data)
    max_num = np.max(number_data)

    raw_data.iloc[:, 1:6] = (number_data - min_num) /  (max_num - min_num)

    # 2번째 인자 31은 30일간 데이터를 모델이 볼 것이라는 뜻이고, 3번쨰 인자 4는 모델이 보고 판단할 인자의 갯수다.
    array_data = np.zeros(shape=(raw_data.shape[0] - 1, 31, 5)) 
    for i in range(raw_data.shape[0] - 31):
        for j in range(31):
            if (i - j < 0): continue
            array_data[i][j] = raw_data.iloc[i - j, 1:6]

    print(raw_data.sample(5))
    test_size = 365

    training_set = array_data[test_size:]
    test_set = array_data[31:test_size]
    test_set_dates = raw_data['date'].iloc[31:test_size]
    
    training_set_x = training_set[:, :-1, 1:]
    training_set_y = training_set[:, 1, 1]

    test_set_x = test_set[:, :-1, 1:]
    test_set_y = test_set[:, :1, 1]

    print(training_set_x.shape)
    print(training_set_y.shape)
    return ((training_set_x, training_set_y), (test_set_x, test_set_y), test_set_dates, ((max_num - min_num) + min_num)[0])

# 목표: 최근 30일의 데이터를 보고 다음날의 종가 가격 예측을 시도
def define_model():
    
    model = Sequential()
    
    # 첫 LSTM 레이어 유닛 갯수가 30인 이유는 최근 30일을 모델의 입력값으로 줄 예정이기 때문
    model.add(LSTM(30, return_sequences=True, input_shape=(30, 4)))
    model.add(LSTM(128, return_sequences=False))
    # 출력값은 다음날 종가 가격 하나뿐이니 유닛 카운트가 1이어야 함 
    model.add(Dense(1, activation='linear'))

    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

    model.summary()

    return model

tf.random.set_seed(128) # 값이 달라지는걸 막기 위한 시드 고정

((training_set_x, training_set_y), (test_set_x, test_set_y), test_set_dates, multiplier) = load_dataset()
model = define_model()
model.fit(x=training_set_x, y=training_set_y, validation_split=0.1, batch_size=100, epochs=50, )

result = model.predict(test_set_x)

model.save('./btc_lstm_model')

print(test_set_dates.shape)
print(test_set_y.shape)
print(multiplier)

fig = plt.figure()
subplot = fig.add_subplot(111)
subplot.plot(test_set_dates, test_set_y * multiplier, label="actual")
subplot.plot(test_set_dates, result * multiplier, label='result')
subplot.legend()
plt.show()
    

    

