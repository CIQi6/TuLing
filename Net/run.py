import sys, os
sys.path.append(os.pardir)  # 为了导入父目录而进行的设定
import numpy as np
from dataset.mnist import load_mnist
from convNet import convNet
from common.trainer import Trainer


(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)


x = x_test[0,:,:,:]
x = x[np.newaxis,:]

t = t_test[0]

network = convNet()
y = network.predict(x)
print("x_t(标签):"+str(t))
print("predict:"+str(y.argmax()))
