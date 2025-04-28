from abc import abstractmethod
import numpy as np

class Layer():
    def __init__(self) -> None:
        self.optimizable = True
    
    @abstractmethod
    def forward():
        pass

    @abstractmethod
    def backward():
        passf


class Linear(Layer):
    """
    The linear layer for a neural network. You need to implement the forward function and the backward function.
    """
    def __init__(self, in_dim, out_dim, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.W = initialize_method(size=(in_dim, out_dim))
        self.b = initialize_method(size=(1, out_dim))
        self.grads = {'W' : None, 'b' : None}
        self.input = None # Record the input for backward process.

        self.params = {'W' : self.W, 'b' : self.b}

        self.weight_decay = weight_decay # whether using weight decay
        self.weight_decay_lambda = weight_decay_lambda # control the intensity of weight decay
            
    
    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        """
        input: [batch_size, in_dim]
        out: [batch_size, out_dim]
        """
        self.input = X  # 保存输入用于 backward
        return X @ self.W + self.b

    def backward(self, grad : np.ndarray):
        """
        input: [batch_size, out_dim] the grad passed by the next layer.
        output: [batch_size, in_dim] the grad to be passed to the previous layer.
        This function also calculates the grads for W and b.
        """
        # 梯度 w.r.t 输入
        grad_input = grad @ self.W.T

        # 梯度 w.r.t 参数
        grad_W = self.input.T @ grad
        grad_b = np.sum(grad, axis=0, keepdims=True)

        # 权重衰减（L2正则化）
        if self.weight_decay:
            grad_W += self.weight_decay_lambda * self.W

        # 保存梯度
        self.grads['W'] = grad_W
        self.grads['b'] = grad_b


        return grad_input
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}

    def update(self):
        # need to update the parameters after the
        # params dictionary has been updated by
        # SGD.step() method.
        self.W = self.params["W"]
        self.b = self.params["b"]


class Conv2D(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, initialize_method=np.random.normal):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # 初始化卷积核和偏置
        self.W = initialize_method(size=(out_channels, in_channels, kernel_size, kernel_size))
        self.b = initialize_method(size=(out_channels,))

        # 保存参数
        self.params = {'W': self.W, 'b': self.b}

        self.grads = {'W': None, 'b': None}
        self.input = None

        self.weight_decay = False # whether using weight decay
        self.weight_decay_lambda = 0

    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        self.input = X
        batch_size, in_channels, height, width = X.shape
        filter_height, filter_width = self.kernel_size, self.kernel_size

        # 把输入变成列
        self.col = im2col(X, filter_height, filter_width, self.stride, self.padding)

        # 把卷积核也展开
        W_col = self.W.reshape(self.out_channels, -1)

        # 卷积（矩阵乘法）
        out = self.col @ W_col.T

        # 加上偏置
        out += self.b.reshape(1, -1)

        # 整理输出形状
        out_height = (height + 2 * self.padding - filter_height) // self.stride + 1
        out_width = (width + 2 * self.padding - filter_width) // self.stride + 1
        out = out.reshape(batch_size, out_height, out_width, self.out_channels).transpose(0, 3, 1, 2)

        return out



    def backward(self, grad):
        """
        grad: [batch_size, out_channels, out_height, out_width]
        """
        batch_size, in_channels, height, width = self.input.shape
        filter_height, filter_width = self.kernel_size, self.kernel_size

        # 1. 处理 grad 形状，准备好矩阵乘法
        grad_col = grad.transpose(0, 2, 3, 1).reshape(-1, self.out_channels)  # [B*out_H*out_W, out_C]

        # 2. 计算 W 和 b 的梯度
        self.grads['W'] = grad_col.T @ self.col  # [out_C, in_C*filter_H*filter_W]
        self.grads['W'] = self.grads['W'].reshape(self.W.shape)  # 恢复成 [out_C, in_C, filter_H, filter_W]

        self.grads['b'] = np.sum(grad, axis=(0, 2, 3), keepdims=True)  # [1, out_C, 1, 1]

        # 3. 计算输入的梯度
        W_col = self.W.reshape(self.out_channels, -1)  # [out_C, in_C*filter_H*filter_W]
        grad_input_col = grad_col @ W_col  # [B*out_H*out_W, in_C*filter_H*filter_W]

        grad_input = col2im(
            grad_input_col,
            self.input.shape,
            filter_height,
            filter_width,
            self.stride,
            self.padding
        )

        return grad_input


    def clear_grad(self):
        self.grads = {'W': None, 'b': None}

    def update(self):
        # need to update the parameters after the
        # params dictionary has been updated by
        # SGD.step() method.
        self.W = self.params["W"]
        self.b = self.params["b"]


class ReLU(Layer):
    """
    An activation layer.
    """
    def __init__(self) -> None:
        super().__init__()
        self.input = None

        self.optimizable =False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        output = np.where(X<0, 0, X)
        return output
    
    def backward(self, grads):
        assert self.input.shape == grads.shape
        output = np.where(self.input < 0, 0, grads)
        return output

class MultiCrossEntropyLoss(Layer):
    """
    A multi-cross-entropy loss layer, with Softmax layer in it, which could be cancelled by method cancel_softmax
    """
    def __init__(self, model = None, max_classes = 10) -> None:
        self.model = model
        self.max_classes = max_classes

        self.predicts = None
        self.labels = None
        self.probs = None
        self.grads = None

        self.optimizable = False
        pass

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)
    
    def forward(self, predicts, labels):
        """
        predicts: [batch_size, D]
        labels : [batch_size, ]
        This function generates the loss.
        """
        self.predicts = predicts  # 保存 logits 用于 backward
        self.labels = labels  # 保存标签用于 backward

        # Step 1: 计算 softmax 概率
        probs = softmax(predicts)  # [batch_size, num_classes]
        # Step 2: 取出正确类的概率
        batch_size = labels.shape[0]
        correct_class_probs = probs[np.arange(batch_size), labels]  # shape: [batch_size]
        # Step 3: 计算交叉熵损失
        loss = -np.mean(np.log(correct_class_probs + 1e-12))  # 防止 log(0)

        self.probs = probs  # 保存 softmax 概率用于 backward
        return loss
    
    def backward(self):
        # first compute the grads from the loss to the input
        batch_size = self.labels.shape[0]
        # 计算 softmax + cross entropy 的梯度
        grad = self.probs.copy()  # [batch_size, num_classes]
        grad[np.arange(batch_size), self.labels] -= 1
        grad /= batch_size  # 平均损失

        self.grads = grad
        # Then send the grads to model for back propagation
        self.model.backward(self.grads)

    def cancel_soft_max(self):
        self.has_softmax = False
        return self
    
class L2Regularization(Layer):
    """
    L2 Reg can act as weight decay that can be implemented in class Linear.
    """
    pass
       
class MaxPool2D(Layer):
    """
    A max pooling layer for CNNs.
    It reduces the spatial dimensions (height and width) by taking the maximum value
    from each non-overlapping region of the input.
    """
    def __init__(self, pool_size, stride, padding=0):
        self.pool_size = pool_size  # The size of the pooling window (height, width)
        self.stride = stride  # The stride of the pooling operation
        self.padding = padding  # Padding for the input
        self.input = None  # To store input for backward pass
        self.output_shape = None  # To store output shape for later use
        self.optimizable = False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        """
        Perform the forward pass of max pooling.
        input: X - input tensor of shape (batch_size, in_channels, H, W)
        output: The output after max pooling.
        """
        batch_size, in_channels, H, W = X.shape

        # Apply padding if necessary
        if self.padding > 0:
            X = np.pad(X, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                       mode='constant', constant_values=0)

        # Calculate the output shape
        out_height = (H - self.pool_size[0]) // self.stride + 1
        out_width = (W - self.pool_size[1]) // self.stride + 1
        self.output_shape = (batch_size, in_channels, out_height, out_width)

        # Create an output tensor to store the pooled results
        output = np.zeros(self.output_shape)

        # Perform max pooling operation
        for i in range(0, H - self.pool_size[0] + 1, self.stride):
            for j in range(0, W - self.pool_size[1] + 1, self.stride):
                # Define the region to pool
                region = X[:, :, i:i + self.pool_size[0], j:j + self.pool_size[1]]
                # Apply max pooling on this region
                output[:, :, i // self.stride, j // self.stride] = np.max(region, axis=(2, 3))

        self.input = X  # Save the input for backward pass
        return output

    def backward(self, grad):
        """
        Backward pass of max pooling.
        Compute the gradient of the loss w.r.t input using the max pooling derivative.
        """
        batch_size, in_channels, H, W = self.input.shape
        grad_input = np.zeros_like(self.input)

        # Iterate over each batch and channel
        for i in range(0, H - self.pool_size[0] + 1, self.stride):
            for j in range(0, W - self.pool_size[1] + 1, self.stride):
                # Define the region to pool
                region = self.input[:, :, i:i + self.pool_size[0], j:j + self.pool_size[1]]
                # Find the maximum values in this region
                max_values = np.max(region, axis=(2, 3))
                # Compute the gradient for the max values
                grad_input[:, :, i:i + self.pool_size[0], j:j + self.pool_size[1]] += (
                    (region == max_values[:, :, None, None]) * grad[:, :, i // self.stride, j // self.stride][:, :, None, None]
                )

        return grad_input

class Flatten(Layer):
    def __init__(self):
        super().__init__()
        self.optimizable = False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        """
        输入 X 的形状为 (batch_size, channels, height, width)
        输出将其展平为 (batch_size, channels * height * width)
        """
        self.input_shape = X.shape
        batch_size = X.shape[0]
        # 将 X 展平为 (batch_size, -1) 形状
        return X.reshape(batch_size, -1)

    def backward(self, grad):
        """
        反向传播时，我们将梯度恢复为原始输入形状
        """
        return grad.reshape(self.input_shape)


def softmax(X):
    x_max = np.max(X, axis=1, keepdims=True)
    x_exp = np.exp(X - x_max)
    partition = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp / partition

def im2col(input_data, filter_height, filter_width, stride, padding):
    """
    将输入数据转换成 im2col 格式，用于加速卷积计算
    input_data: 输入数据，形状为 (batch_size, in_channels, height, width)
    filter_height, filter_width: 卷积核的高度和宽度
    stride: 步幅
    padding: 填充的大小
    """
    batch_size, in_channels, height, width = input_data.shape

    # 计算输出的尺寸
    out_height = (height + 2 * padding - filter_height) // stride + 1
    out_width = (width + 2 * padding - filter_width) // stride + 1

    # 为输入添加填充
    input_padded = np.pad(input_data, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')

    # 生成 im2col 输出
    col = np.zeros((batch_size, in_channels, filter_height, filter_width, out_height, out_width))

    for y in range(filter_height):
        y_max = y + stride * out_height
        for x in range(filter_width):
            x_max = x + stride * out_width
            col[:, :, y, x, :, :] = input_padded[:, :, y:y_max:stride, x:x_max:stride]

    col = col.reshape(batch_size * out_height * out_width, -1)
    return col

def col2im(col, input_shape, filter_height, filter_width, stride, padding):
    """
    将 im2col 后的矩阵还原成原始形状的 input，同时正确累加重叠部分。
    col: [batch_size * out_height * out_width, in_channels * filter_height * filter_width]
    input_shape: (batch_size, in_channels, height, width)
    """
    batch_size, in_channels, height, width = input_shape
    out_height = (height + 2 * padding - filter_height) // stride + 1
    out_width = (width + 2 * padding - filter_width) // stride + 1

    # 先初始化带 padding 的空白 input
    input_padded = np.zeros((batch_size, in_channels, height + 2 * padding, width + 2 * padding))

    # 把 col 恢复成形状 (batch_size, out_height, out_width, in_channels, filter_height, filter_width)
    col = col.reshape(batch_size, out_height, out_width, in_channels, filter_height, filter_width)
    col = col.transpose(0, 3, 4, 5, 1, 2)

    for y in range(filter_height):
        for x in range(filter_width):
            input_padded[:, :, y:y + stride*out_height:stride, x:x + stride*out_width:stride] += col[:, :, y, x, :, :]

    # 去掉 padding
    if padding == 0:
        return input_padded
    else:
        return input_padded[:, :, padding:-padding, padding:-padding]
