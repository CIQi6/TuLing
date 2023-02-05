import sys, os
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import numpy as np
from common.util import im2col,col2im

class ConvLayer():
    
    def __init__(self,W,b,stride,pad):
        #
        self.W = W
        self.b = b
        self.pad = pad
        self.stride = stride
        
        # 计算反向传播时要用到的一些数据
        self.x = None
        self.col = None
        self.col_w = None
        self.dw = None
        self.db = None

    def forward(self,X):

        Filter_N,Filter_C,Filter_H,Filter_W = self.W.shape
        N_prev,C_prev,H_prev,W_prev = X.shape
        H = int((H_prev-Filter_H+2*self.pad)/self.stride+1) #输出图像的高度
        W = int((W_prev-Filter_W+2*self.pad)/self.stride+1) #输出图像的宽度

        col = im2col(X,Filter_H,Filter_W,self.stride,self.pad)
        col_w = self.W.reshape(Filter_N,-1).T # 将卷积核组展开，每一列代表一个卷积核
        out = np.dot(col,col_w) + self.b #卷积运算加上偏置
        out = out.reshape(N_prev,H,W,-1).transpose(0,3,1,2) #reshape为(N,H,W,C) 再转为(N，C,H,W)

        #记录正向传播数值
        self.x = X
        self.col = col
        self.col_W = col_w

        return out


    def backward(self,dOut):
        #dOut--输出值梯度
        Filter_N,Filter_C,Filter_H,Filter_W = self.W.shape
        dOut = dOut.transpose(0,2,3,1).reshape(-1,Filter_N)

        self.db = np.sum(dOut,axis=0)
        self.dW = np.dot(self.col.T,dOut)
        self.dW = self.dW.transpose(1,0).reshape(Filter_N,Filter_C,Filter_H,Filter_W)

        dCol = np.dot(dOut,self.col_W.T)
        dX = col2im(dCol,self.x.shape,Filter_H,Filter_W,self.stride,self.pad)

        return dX
    
class PoolingLayer():
    
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad
        
        self.x = None
        self.arg_max = None

    def forward(self,x):
        N_prev,C_prev,H_prev,W_prev = x.shape
        H = int((H_prev-self.pool_h+2*self.pad)/self.stride+1)
        W = int((W_prev-self.pool_w+2*self.pad)/self.stride+1)

        # 展开输入数据
        col = im2col(x,self.pool_h,self.pool_w,self.stride,self.pad)
        col = col.reshape(-1,self.pool_h*self.pool_w)

        #最大池化
        arg_max = np.argmax(col, axis=1) # 记住位置，反向传播Pool时用到
        out = np.max(col,axis=1)

        #reshape为(N,H,W,C) 再转为(N，C,H,W)
        out = out.reshape(N_prev,H,W,-1).transpose(0,3,1,2)
        
        #记录正向传播数值
        self.x = x
        self.arg_max = arg_max

        return out
    
    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)
        
        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,)) 
        
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)
        
        return dx
    
class Relu():
    
    def __init__(self):
        #mask蒙版，用来记录<=0 的值位置
        self.mask =None
    
    def forward(self,x):
        # 利用蒙版，把值 <=0 置为0，>0不变，
        self.mask = (x<=0)
        out = x.copy()
        out[self.mask] = 0

        return out
    
    def backward(self,dOut):
        #dOut -- 输出值的梯度
        dOut[self.mask] = 0
        dX = dOut

        return dX

class FullyConnectLayer():
    # 以矩阵形式
    def __init__(self, W, b):
        self.W =W
        self.b = b
        
        self.x = None
        self.original_x_shape = None
        # 权重和偏置参数的导数
        self.dW = None
        self.db = None

    def forward(self, x):
        # 对应矩阵
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x

        out = np.dot(self.x, self.W) + self.b
        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        
        dx = dx.reshape(*self.original_x_shape)  # 还原输入数据的形状（对应矩阵）
        return dx


class SoftmaxWithLoss():
    def __init__(self):
        self.loss = None
        self.y = None # softmax的输出
        self.x_t = None # 输入数据的标签

    def softmax(self,x):
        x = x - np.max(x) # 防止溢出
        return np.exp(x) / np.sum(np.exp(x))

    def cross_entropy_error(self,y, t):
      if y.ndim == 1:
          t = t.reshape(1, t.size)
          y = y.reshape(1, y.size)
          
      # 监督数据是one-hot-vector的情况下，转换为正确解标签的索引
      if t.size == y.size:
          t = t.argmax(axis=1)
      batch_size = y.shape[0]

      return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

    def forward(self, x, x_t):
        self.x_t = x_t
        self.y = self.softmax(x)
        self.loss = self.cross_entropy_error(self.y, self.x_t) #交叉熵损失
        
        return self.loss

    def backward(self, dout=1):
        batch_size = self.x_t.shape[0]
        if self.x_t.size == self.y.size: # 监督数据是one-hot-vector的情况
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.x_t] -= 1
            dx = dx / batch_size
        
        return dx
    
class Dropout:
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flg=True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask
        
class BatchNormalization:
    
    def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None):
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.input_shape = None # Conv层的情况下为4维，全连接层的情况下为2维  

        # 测试时使用的平均值和方差
        self.running_mean = running_mean
        self.running_var = running_var  
        
        # backward时使用的中间数据
        self.batch_size = None
        self.xc = None
        self.std = None
        self.dgamma = None
        self.dbeta = None

    def forward(self, x, train_flg=True):
        self.input_shape = x.shape
        if x.ndim != 2:
            N, C, H, W = x.shape
            x = x.reshape(N, -1)

        out = self.__forward(x, train_flg)
        
        return out.reshape(*self.input_shape)
            
    def __forward(self, x, train_flg):
        if self.running_mean is None:
            N, D = x.shape
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)
                        
        if train_flg:
            mu = x.mean(axis=0)
            xc = x - mu
            var = np.mean(xc**2, axis=0)
            std = np.sqrt(var + 10e-7)
            xn = xc / std
            
            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std
            self.running_mean = self.momentum * self.running_mean + (1-self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1-self.momentum) * var            
        else:
            xc = x - self.running_mean
            xn = xc / ((np.sqrt(self.running_var + 10e-7)))
            
        out = self.gamma * xn + self.beta 
        return out

    def backward(self, dout):
        if dout.ndim != 2:
            N, C, H, W = dout.shape
            dout = dout.reshape(N, -1)

        dx = self.__backward(dout)

        dx = dx.reshape(*self.input_shape)
        return dx

    def __backward(self, dout):
        dbeta = dout.sum(axis=0)
        dgamma = np.sum(self.xn * dout, axis=0)
        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size
        
        self.dgamma = dgamma
        self.dbeta = dbeta
        
        return dx