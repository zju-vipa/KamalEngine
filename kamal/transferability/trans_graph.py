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
import torch
import networkx as nx
from . import depara
import os, abc
from typing import Callable
from kamal import hub
import json, numbers

from tqdm import tqdm

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
            for n2 in tqdm(self._models.values()):
                if n1!=n2 and not g.has_edge(n1, n2):
                    try:
                        g.add_edge(n1, n2, dist=metric( n1, n2 ))
                    except:
                        ori_device = metric.device
                        metric.device = torch.device('cpu')
                        g.add_edge(n1, n2, dist=metric( n1, n2 ))
                        metric.device = ori_device
    
    def export_to_json(self, metric_name, output_filename, topk=None, normalize=False):
        graph = self._graphs.get( metric_name, None )
        assert graph is not None
        graph_data={
            'nodes': [],
            'edges': [],
        }
        node_to_idx = {}
        for i, node in enumerate(self._models.values()):
            tags = node.tag
            metadata = node.metadata
            node_data = { k:v for (k, v) in tags.items() if isinstance(v, (numbers.Number, str) ) }
            node_data['name'] = metadata['name']
            node_data['task'] = metadata['task']
            node_data['dataset'] = metadata['dataset']
            node_data['url'] = metadata['url']
            node_data['id'] = i
            graph_data['nodes'].append({'tags': node_data})
            node_to_idx[node] = i

        # record Edges
        edge_list = graph_data['edges']
        topk_dist = { idx: [] for idx in range(len( self._models )) }
        for i, edge in enumerate(graph.edges.data('dist')):
            s, t, d = int( node_to_idx[edge[0]] ), int( node_to_idx[edge[1]] ), float(edge[2])
            topk_dist[s].append(d)
            edge_list.append([
                s, t, d # source, target, distance
            ])

        if isinstance(topk, int):
            for i, dist in topk_dist.items():
                dist.sort()
                topk_dist[i] = dist[topk]
            graph_data['edges']  = [ edge for edge in edge_list if edge[2] < topk_dist[edge[0]] ]

        if normalize:
            edge_dist = [e[2] for e in graph_data['edges']]
            min_dist, max_dist = min(edge_dist), max(edge_dist)
            for e in graph_data['edges']:
                e[2] = (e[2] - min_dist+1e-8) / (max_dist - min_dist+1e-8)
    
        with open(output_filename, 'w') as fp:
            json.dump(graph_data, fp)