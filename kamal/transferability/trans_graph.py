import networkx as nx
from . import depara
import os, abc
from typing import Callable
from kamal import hub

class Node(object):
    def __init__(self, hub_root, entry_name, spec_name):
        self.hub_root = hub_root
        self.entry_name = entry_name
        self.spec_name = spec_name

    @property
    def model(self):
        return hub.load( self.hub_root, entry_name=self.entry_name, spec_name=self.spec_name ).eval()

    @property
    def tag(self):
        return hub.load_tags(self.hub_root, entry_name=self.entry_name, spec_name=self.spec_name)

    @property
    def metadata(self):
        return hub.load_metadata(self.hub_root, entry_name=self.entry_name, spec_name=self.spec_name)

class TransferabilityGraph(object):
    def __init__(self, model_zoo):
        self.model_zoo = os.path.abspath( os.path.expanduser( model_zoo ) )
        self._graphs = dict()
        self._models = dict()
        self._register_models()

    def _register_models(self):
        cnt = 0
        for hub_root in self._list_modelzoo(self.model_zoo):
            for entry_name, spec_name in hub.list_spec(hub_root):
                node = Node( hub_root, entry_name, spec_name )
                name = node.metadata['name']
                self._models[name] = node
                cnt += 1
        print("%d models has been registered!"%cnt)
    
    def _list_modelzoo(self, zoo_dir):
        zoo_list = []
        def _traverse(path):
            for item in os.listdir(path):
                item_path = os.path.join(path, item)
                if os.path.isdir(item_path):
                    if os.path.exists(os.path.join( item_path, 'code/hubconf.py' )):
                        zoo_list.append(item_path)
                    else:
                        _traverse( item_path )
        _traverse(zoo_dir)
        return zoo_list

    def add_metric(self, metric_name, metric):
        self._graphs[metric_name] = g = nx.DiGraph()
        g.add_nodes_from( self._models.values() )
        
        for n1 in self._models.values():
            for n2 in self._models.values():
                g.add_edge(n1, n2, weight=metric( n1, n2 ))
        