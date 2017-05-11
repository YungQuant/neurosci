import pyedflib
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler


def create_binary_eeg_epilepsy_training_dataset(datasets, epilepsy):
    X, Y = [], []
    for i in range(len(datasets)):
        X.append(datasets[i])
        if epilepsy == True:
            Y.append(1)
        else:
            Y.append(0)

    return X, Y



#filepaths = ['00000003/', '00000006/', '00000008/', '00000011/', '00000013/', '00000018/', '00000032/',
#             '00000005/', '00000007/', '00000009/', '00000012/', '00000015/', '00000024/']
filepaths = ['00000003/s01_2011_11_01/a_.edf']
datasets = []; data = [];
path = "../../../../Desktop/tuh_eeg_epilepsy/"
control_filepaths = []

for i in range(len(filepaths)):
    filepaths[i] = (path + filepaths[i])
    print("Reading:", filepaths[i])
    f = pyedflib.EdfReader(filepaths[i])
    signals = f.getSignalLabels()
    print("Signals:", signals)

    for k in range(len(signals)):
        data.append(f.readSignal(k))
        print(signals[k], "Data:", data)

    datasets.append(data)
    data = []


tot_len = 0

scaler = MinMaxScaler(feature_range=(-1, 1))
X, Y = create_binary_eeg_epilepsy_training_dataset(datasets, epilepsy=True)
X, Y = create_binary_eeg_epilepsy_training_dataset(control_datasets, epilepsy=False)

X = scaler.fit_transform(X)
#X = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))

model = Sequential()
model.add(Dense(len(datasets[0][0]), input_shape=(len(datasets), len(datasets[0]), len(datasets[0][0]))))
model.add(Dense(int(np.floor(len(datasets[0][0]) / 10)), activation='tanh'))
model.add(Dropout(0.2))
model.add(Dense(int(np.floor(len(datasets[0][0]) / 100))))
model.add(Dense(1, activation='tanh'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
model.fit(X[:int(np.floor(len(X) * 0.8))], Y[:int(np.floor(len(Y) * 0.8))], nb_epoch=30, batch_size=1, verbose=0)

print(model.predict(X[int(np.floor(len(X) * 0.8)):]))