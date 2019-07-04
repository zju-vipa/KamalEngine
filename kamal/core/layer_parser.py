import torch.nn as nn


class LayerParser(object):
    """ Layer-Function mapping.
    It will parse layer types and call corresponding functions.
    """

    def __init__(self, *type_fn_pair, match_fn=lambda layer, layer_type: isinstance(layer, layer_type)):
        self._layer_type = []
        self._parse_fn = []
        for layer_type, parse_fn in type_fn_pair:
            self.add_rules(layer_type, parse_fn)

        self.match_fn = match_fn

    def clear_rules(self):
        self._layer_type = []
        self._parse_fn = []

    def add_rules(self, layer_type, parse_fn):
        self._layer_type.insert(0, layer_type)
        self._parse_fn.insert(0, parse_fn)

    def __call__(self, layer):
        return self.parse(layer)

    def parse(self, layer):
        for layer_type, parse_fn in zip(self._layer_type, self._parse_fn):
            if self.match_fn(layer, layer_type):
                return parse_fn(layer)
        return None
