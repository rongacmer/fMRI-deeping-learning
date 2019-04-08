import numpy as np

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
            data_index['fcn_train'] = order[test_len+train_len:-1]
            # 测试集
            data_index['test'] = order[0:test_len]
            yield data_index

