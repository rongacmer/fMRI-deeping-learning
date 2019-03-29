import tensorflow as tf
class evaluation:
    def __init__(self,y_true = None,y_predict = None):
        TP,TN,FP,FN = 0,0,0,0
        y_len = len(y_true) if y_true is not None else 0
        for i in range(y_len):
            if y_true[i] == 0 and y_predict[i] == 0:
                TP += 1
            if y_true[i] == 0 and y_predict[i] == 1:
                FN += 1
            if y_true[i] == 1 and y_predict[i] == 0:
                FP += 1
            if y_true[i] == 1 and y_predict[i] == 1:
                TN += 1
        self.SEN = 0
        self.SPE = 0
        self.ACC = 0
        if TP:
            self.SEN= TP / (TP + FN)
        if TN:
            self.SPE = TN/ (TN + FP)
        if y_len:
            self.ACC = (TP + TN)/y_len

    def __str__(self):
        return 'ACC:'+str(self.ACC)+' SEN:'+str(self.SEN)+' SPE:'+str(self.SPE)

    def __iadd__(self, b):
        self.ACC += b.ACC
        self.SEN += b.SEN
        self.SPE += b.SPE
        return self

    def __itruediv__(self, b):
        self.ACC /= b
        self.SEN /= b
        self.SPE /= b
        return self

    def __truediv__(self, other):
        return self.__itruediv__(other)

def test():
    # p=dict()
    # p['labels'] = tf.placeholder(tf.float32,[None,2])
    # p['predict'] = tf.placeholder(tf.float32,[None,2])
    # p['y_true'] = tf.argmax(p['labels'],1)
    # p['y_predict'] = tf.argmax(p['predict'],1)
    # one_hots=[[1,0],[1,0],[0,1],[0,1]]
    # predict =[[1,0],[0,1],[1,0],[0,1]]
    # feed_dict={p['labels']:one_hots,p['predict']:predict}
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     y_true,y_predict = sess.run([p['y_true'],p['y_predict']],feed_dict)
    #     print(y_true,y_predict)

    a = evaluation(y_true=[1,1,1,1],y_predict=[1,1,0,0])
    print(a)
    a += a
    print(a)
    a = a/2
    print(a)

# test()