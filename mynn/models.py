from .op import *
import pickle

class Model_MLP(Layer):
    """
    A model with linear layers. We provied you with this example about a structure of a model.
    """
    def __init__(self, size_list=None, act_func=None, lambda_list=None):
        self.size_list = size_list
        self.act_func = act_func

        if size_list is not None and act_func is not None:
            self.layers = []
            for i in range(len(size_list) - 1):
                layer = Linear(in_dim=size_list[i], out_dim=size_list[i + 1])
                if lambda_list is not None:
                    layer.weight_decay = True
                    layer.weight_decay_lambda = lambda_list[i]
                if act_func == 'Logistic':
                    raise NotImplementedError
                elif act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(size_list) - 2:
                    self.layers.append(layer_f)

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        assert self.size_list is not None and self.act_func is not None, 'Model has not initialized yet. Use model.load_model to load a model or create a new model with size_list and act_func offered.'
        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads

    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            param_list = pickle.load(f)
        self.size_list = param_list[0]
        self.act_func = param_list[1]

        for i in range(len(self.size_list) - 1):
            self.layers = []
            for i in range(len(self.size_list) - 1):
                layer = Linear(in_dim=self.size_list[i], out_dim=self.size_list[i + 1])
                layer.W = param_list[i + 2]['W']
                layer.b = param_list[i + 2]['b']
                layer.params['W'] = layer.W
                layer.params['b'] = layer.b
                layer.weight_decay = param_list[i + 2]['weight_decay']
                layer.weight_decay_lambda = param_list[i+2]['lambda']
                if self.act_func == 'Logistic':
                    raise NotImplemented
                elif self.act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(self.size_list) - 2:
                    self.layers.append(layer_f)
        
    def save_model(self, save_path):
        param_list = [self.size_list, self.act_func]
        for layer in self.layers:
            if layer.optimizable:
                param_list.append({'W' : layer.params['W'], 'b' : layer.params['b'], 'weight_decay' : layer.weight_decay, 'lambda' : layer.weight_decay_lambda})
        
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)




class Model_CNN(Layer):
    """
    A model with conv2D layers. Implement it using the operators you have written in op.py
    """
    def __init__(self):
        pass

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        pass

    def backward(self, loss_grad):
        pass

    def load_model(self, param_list):
        pass

    def save_model(self, save_path):
        pass



class LeNet5:
    def __init__(self):
        self.layers = []
        # 第一层卷积: 输入1通道, 输出6通道, 卷积核5x5
        self.conv1 = Conv2D(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0)
        self.relu1 = ReLU()
        self.pool1 = MaxPool2D(pool_size=(2,2), stride=2)
        self.layers.append(self.conv1)
        self.layers.append(self.relu1)
        self.layers.append(self.pool1)

        # 第二层卷积: 输入6通道, 输出16通道, 卷积核5x5
        self.conv2 = Conv2D(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.relu2 = ReLU()
        self.pool2 = MaxPool2D(pool_size=(2,2), stride=2)
        self.layers.append(self.conv2)
        self.layers.append(self.relu2)
        self.layers.append(self.pool2)

        # 展平
        self.flatten = Flatten()
        self.layers.append(self.flatten)

        # 全连接层
        self.fc1 = Linear(in_dim=16*5*5, out_dim=120)
        self.relu3 = ReLU()
        self.fc2 = Linear(in_dim=120, out_dim=84)
        self.relu4 = ReLU()
        self.fc3 = Linear(in_dim=84, out_dim=10)  # 最后是10类
        self.layers.append(self.fc1)
        self.layers.append(self.relu3)
        self.layers.append(self.fc2)
        self.layers.append(self.relu4)
        self.layers.append(self.fc3)


    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        # ===> 这里加一行 reshape
        if X.ndim == 2:
            X = X.reshape(X.shape[0], 1, 32, 32)

        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def compute_loss(self, predicts, labels):
        return self.loss_layer(predicts, labels)

    def backward(self, loss_grads):
        grads = loss_grads
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads

    def update(self, learning_rate):
        self.fc3.update(learning_rate)
        self.fc2.update(learning_rate)
        self.fc1.update(learning_rate)

        self.conv2.update(learning_rate)
        self.conv1.update(learning_rate)

    def save_model(self, save_path):
        param_list = []
        for layer in self.layers:
            if layer.optimizable:
                param_list.append({'W': layer.params['W'], 'b': layer.params['b'], 'weight_decay': layer.weight_decay,
                                   'lambda': layer.weight_decay_lambda})

        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)


