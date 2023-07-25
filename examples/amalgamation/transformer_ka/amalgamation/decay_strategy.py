
class DecayStrategy:
    def __init__(self, total_epoch: int, enable: bool = True) -> None:
        self.total_epoch = total_epoch
        self.enable = enable

    def __call__(self, epoch: int):
        decay = 1.0
        if self.enable:
            decay = 1.0 - epoch / self.total_epoch
            if decay < 0:
                decay = 0
        return decay

