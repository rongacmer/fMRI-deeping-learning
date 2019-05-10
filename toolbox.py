import numpy as np
from keras.callbacks import Callback
import warnings
from keras import backend as K
def cut_data(data_len,test_len,shuffle=True):
    #训练集长度
    train_len = (data_len-test_len)//2
    while 1:
        order = np.arange(0, data_len)
        if shuffle:
            np.random.shuffle(order)
        i = 0
        while i < data_len:
            for j in range(test_len):
                #换位置
                order[j],order[i] = order[i],order[j]
                i+=1
                if i >= data_len:
                    break
            data_index=dict()
            # voxnent训练集
            data_index['voxnet_train'] = order[test_len:test_len + train_len]
            # fcn训练集
            data_index['fcn_train'] = order[test_len:]
            # 测试集
            data_index['test'] = order[0:test_len]
            yield data_index


class EarlyStoppingByLossVal(Callback):
    def __init__(self, monitor='val_loss', value=0.05, verbose=0,patience = 0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose
        self.patience = patience
        self.wait = 0

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

        if current <= self.value:
            self.wait += 1
            if self.wait >= self.patience:
                if self.verbose > 0:
                    print("Epoch %05d: early stopping THR" % epoch)
                self.model.stop_training = True
        else:
            self.wait = 0

class EarlyStoppingByACC(Callback):
    def __init__(self, monitor='acc', value=0.95, verbose=0,patience = 10):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose
        self.patience = patience
        self.wait = 0

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

        if current >= self.value:
            self.wait += 1
            if self.wait >= self.patience:
                if self.verbose > 0:
                    print("Epoch %05d: early stopping THR" % epoch)
                self.model.stop_training = True
        else:
            self.wait = 0

def mean_squared_error(y_true, y_pred):
    print('acc:')
    print(K.square(y_pred - y_true))
    return K.mean(K.square(y_pred - y_true), axis=-1)