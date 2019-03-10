import tensorflow as tf
import numpy as np
from functools import reduce
class NPZSaver(object):

    def __init__(self,net):
        self._net = net

    def save(self,session,f):
        #文件保存#
        np.savez_compressed(f,**dict((v.name,session.run(v)) for v in self._net.variables))

    def restore(self,session,f):
        #加载模型#
        kwds = np.load(f)
        for v in self._net.variables:
            if v.name in kwds:
                session.run(v.assign(kwds[v.name]))

class BaseNet(object):

    def __init__(self,name,x):
        self._layers = []
        self._name = name
        self.append('input',x)

    def append(self, name, x):
        #添加网络结构#
        setattr(x, 'layer_name', name)
        self._layers.append(x)
        return self

    #静态方法
    def layer(func):
        def w(self, name,x='',*args,**kwargs):
            with tf.variable_scope(self._name):
                if isinstance(x,list) or isinstance(x, tuple):
                    x = [self[i] for i in x]
                elif isinstance(x,str):
                    x = self[x]
                else:
                    #######?#########
                    x, args = self[''],(x,)+args
                x = func(self,name,x,*args,**kwargs)
                for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self._name+'/'+name):
                    setattr(x, v.name.split('/')[-1].split(':')[0],v)
                self.append(name,x)
            return self

    layer = staticmethod(layer)

    def __getitem__(self, item):
        if isinstance(item,int) and item < len(self._layers):
            return self._layers[item]
        for l in self._layers:
            if hasattr(l,'layer_name') and l.layer_name == item:
                return l
        return self.output

    def __str__(self):return '\n'.join(
        l.layer_name + '  ' + str(l.shape.as_list()) + ''.join(
            '\n    ' + v.name + '  ' + str(v.shape.as_list())
            for v in self.variables if l.layer_name in v.name.split('/'))
        for l in self._layers)

    __repr__ = __str__
    def __len__(self):return len(self._layers)
    def __iter__(self):return iter(self._layers)
    #作用：方法变为属性
    @property
    def kernels(self):return [l.kernel for l in self._layers if hasattr(l, 'kernel')]
    @property
    def biases(self):return [l.biases for l in self._layers if hasattr(l,'biases')]
    @property
    def variables(self):return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,self._name)
    @property
    def total_params(self):
        #参数数目
        return sum(reduce(lambda a, b: a * b, v.shape, 1) for v in self.variables)

    @property
    def input(self):
        #输出层
        return self._layers[0] if self._layers else None  # self[0]

    @property
    def output(self):
        #输入层
        return self._layers[-1] if self._layers else None  # self[-1]

    @property
    def saver(self):
        return tf.train.Saver(self.variables)

    @property
    def npz_saver(self):
        return NPZSaver(self)