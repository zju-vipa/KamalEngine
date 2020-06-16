class FeatureHook():
    def __init__(self, module):
        self.module = module
        self.feat_in = None
        self.feat_out = None

    def register(self):
        self.module.register_forward_hook(self.hook_fn_forward)

    def hook_fn_forward(self, module, fea_in, fea_out):
        self.feat_in = fea_in[0]
        self.feat_out = fea_out

