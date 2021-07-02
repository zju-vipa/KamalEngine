# Copyright 2020 Zhejiang Lab. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================

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

