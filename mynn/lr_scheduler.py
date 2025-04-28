from abc import abstractmethod
import numpy as np

class scheduler():
    def __init__(self, optimizer) -> None:
        self.optimizer = optimizer
        self.step_count = 0
    
    @abstractmethod
    def step():
        pass


class StepLR(scheduler):
    def __init__(self, optimizer, step_size=30, gamma=0.1) -> None:
        super().__init__(optimizer)
        self.step_size = step_size
        self.gamma = gamma

    def step(self) -> None:
        self.step_count += 1
        if self.step_count >= self.step_size:
            self.optimizer.init_lr *= self.gamma
            self.step_count = 0

class MultiStepLR(scheduler):
    def __init__(self, optimizer, milestones, gamma=0.1):
        """
        MultiStepLR scheduler.

        :param optimizer: 优化器
        :param milestones: 一个列表，包含需要调整学习率的迭代次数
        :param gamma: 学习率调整的乘数，默认值为 0.1
        """
        super().__init__(optimizer)
        self.milestones = sorted(milestones)  # 确保里程碑是按顺序排列的
        self.gamma = gamma
        self.current_step = 0  # 当前迭代次数

    def step(self):
        """
        在每次迭代时调用，根据当前迭代次数调整学习率。
        """
        self.current_step += 1
        if self.current_step in self.milestones:
            self.optimizer.init_lr *= self.gamma  # 调整学习率

class ExponentialLR(scheduler):
    pass