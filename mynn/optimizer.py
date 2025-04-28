from abc import abstractmethod
import numpy as np


class Optimizer:
    def __init__(self, init_lr, model) -> None:
        self.init_lr = init_lr
        self.model = model

    @abstractmethod
    def step(self):
        pass


class SGD(Optimizer):
    def __init__(self, init_lr, model):
        super().__init__(init_lr, model)
    
    def step(self):
        for layer in self.model.layers:
            if layer.optimizable == True:
                for key in layer.params.keys():
                    if layer.weight_decay:
                        layer.params[key] *= (1 - self.init_lr * layer.weight_decay_lambda)
                    layer.params[key] = layer.params[key] - self.init_lr * layer.grads[key]
                layer.update()

class SGD_Momentum(Optimizer):
    def __init__(self, init_lr, model, momentum=0.0):
        super().__init__(init_lr, model)
        self.momentum = momentum
        self.velocity = {}

    def step(self):
        for layer in self.model.layers:
            if layer.optimizable == True:
                for key in layer.params.keys():
                    grad = layer.grads[key]
                    if layer.weight_decay:
                        layer.params[key] *= (1 - self.init_lr * layer.weight_decay_lambda)
                    if (layer, key) not in self.velocity:
                        self.velocity[(layer, key)] = 0
                    self.velocity[(layer, key)] = self.momentum * self.velocity[(layer, key)] - self.init_lr * grad
                    layer.params[key] += self.velocity[(layer, key)]
                layer.update()


class Adam(Optimizer):
    def __init__(self, init_lr, model, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(init_lr, model)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {}  # 一阶矩估计
        self.v = {}  # 二阶矩估计
        self.t = 0   # 时间步

    def step(self):
        self.t += 1
        for layer in self.model.layers:
            if layer.optimizable == True:
                for key in layer.params.keys():
                    grad = layer.grads[key]
                    if grad is None:
                        continue

                    if layer.weight_decay:
                        layer.params[key] *= (1 - self.init_lr * layer.weight_decay_lambda)

                    if (layer, key) not in self.m:
                        self.m[(layer, key)] = np.zeros_like(grad)
                        self.v[(layer, key)] = np.zeros_like(grad)

                    # 更新一阶和二阶矩
                    self.m[(layer, key)] = self.beta1 * self.m[(layer, key)] + (1 - self.beta1) * grad
                    self.v[(layer, key)] = self.beta2 * self.v[(layer, key)] + (1 - self.beta2) * (grad ** 2)

                    # 偏置修正
                    m_hat = self.m[(layer, key)] / (1 - self.beta1 ** self.t)
                    v_hat = self.v[(layer, key)] / (1 - self.beta2 ** self.t)

                    # 参数更新
                    adjusted = self.init_lr * m_hat / (np.sqrt(v_hat) + self.eps)
                    if key == "b":
                        adjusted = adjusted.squeeze()
                    layer.params[key] -= adjusted

                layer.update()

class MomentGD(Optimizer):
    def __init__(self, init_lr, model, mu):
        pass
    
    def step(self):
        pass