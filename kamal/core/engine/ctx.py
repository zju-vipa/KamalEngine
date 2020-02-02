import contextlib

@contextlib.contextmanager
def eval_ctx(model):
    is_training = model.training
    model.eval()
    yield
    model.train(is_training)

@contextlib.contextmanager
def train_ctx(model):
    is_training = model.training
    model.train()
    yield
    model.train(is_training)

@contextlib.contextmanager
def device_ctx(model, device):
    ori_device = next(model.parameters()).device 
    model.to(device)
    yield
    model.to(ori_device)
