class FeatureHook():
    def __init__(self, module):
        self.module = module
        self.feat_in = None
        self.feat_out = None
        self.register()

    def register(self):
        self._hook = self.module.register_forward_hook(self.hook_fn_forward)

    def remove(self):
        self._hook.remove()

    def hook_fn_forward(self, module, fea_in, fea_out):
        self.feat_in = fea_in[0]
        self.feat_out = fea_out

