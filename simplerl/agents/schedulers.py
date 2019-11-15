import abc

class Scheduler(abc.ABC):
    def __init__(self, value):
        self.value = value
        self.step = 0

    @abc.abstractmethod
    def __call__(self):
        pass
        
    @abc.abstractmethod
    def reset(self):
        pass
        
class ConstantScheduler(Scheduler):
    def __init__(self, value):
        super().__init__(value)
        
    def __call__(self):
        self.step += 1
        value = self.value
        
        return value
        
    def reset(self):
        self.step = 0
        
class LinearScheduler(Scheduler):
    def __init__(self, start_value, end_value, steps):
        self.start_value = start_value
        self.end_value = end_value
        self.value = start_value
        self.step = 0
        
        self.increment = (end_value - start_value) / steps
        
        if end_value > start_value:
            self.bound = min
        elif end_value < start_value:
            self.bound = max
        else:
            raise ValueError('Start and end value cannot be the same')

    def __call__(self):
        self.step += 1
        value = self.value
        
        self.value = self.bound(self.value + self.increment, self.end_value)
        
        return value
        
    def reset(self):
        self.step = 0
        self.value = self.start_value

class ExponentialScheduler(Scheduler):
    def __init__(self, start_value, end_value, rate):
        self.start_value = start_value
        self.end_value = end_value
        self.rate = rate
        self.value = start_value
        self.step = 0
        
        if end_value > start_value:
            assert rate > 1, 'Rate must be greater than one when the ending value is greater than the starting value'
            self.bound = min
        elif end_value < start_value:
            assert rate < 1, 'Rate must be less than one when the ending value is less than the starting value'
            self.bound = max
        else:
            raise ValueError('Start and end value cannot be the same')

    def __call__(self):
        self.step += 1
        value = self.value
        
        self.value = self.bound(self.value * self.rate, self.end_value)
        
        return value
        
    def reset(self):
        self.step = 0
        self.value = self.start_value