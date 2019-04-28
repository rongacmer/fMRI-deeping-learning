import tensorflow as tf
class evaluation:
    def __init__(self,y_true = None,y_predict = None):
        self.TP,self.TN,self.FP,self.FN = 0,0,0,0
        y_len = len(y_true) if y_true is not None else 0
        for i in range(y_len):
            if y_true[i] == 0 and y_predict[i] == 0:
                self.TP += 1
            if y_true[i] == 0 and y_predict[i] == 1:
                self.FN += 1
            if y_true[i] == 1 and y_predict[i] == 0:
                self.FP += 1
            if y_true[i] == 1 and y_predict[i] == 1:
                self.TN += 1

    @property
    def SEN(self):#敏感性，召回率
        if self.TP:
            return self.TP/(self.TP+self.FN)
        return 0

    @property
    def Pre(self): #精确率
        if self.TP:
            return self.TP/(self.TP + self.FP)
        return 0

    @property
    def SPE(self):
        if self.TN:
            return self.TN/(self.TN+self.FP)
        return 0

    @property
    def ACC(self):
        if self.TP or self.TN:
            return (self.TP+self.TN)/(self.TP+self.TN+self.FP+self.FN)
        return 0

    @property
    def F1(self): #F1 measure
        if self.SEN + self.Pre:
            return 2 * self.SEN * self.Pre / (self.SEN + self.Pre)
        return 0

    def __str__(self):
        return 'ACC:'+str(self.ACC)+' SEN:'+str(self.SEN)+' SPE:'+str(self.SPE) + " F1:"+str(self.F1)

    def __iadd__(self, b):
        self.TP += b.TP
        self.TN += b.TN
        self.FP += b.FP
        self.FN += b.FN
        return self

    # def __itruediv__(self, b):
    #     self.ACC /= b
    #     self.SEN /= b
    #     self.SPE /= b
    #     return self
    #
    # def __truediv__(self, other):
    #     return self.__itruediv__(other)

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