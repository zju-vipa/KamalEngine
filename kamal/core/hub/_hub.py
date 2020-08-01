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

from pip._internal import main as pipmain
from typing import Sequence, Optional

import torch
import os, sys, re
import shutil, inspect
import importlib.util
from glob import glob

from ruamel.yaml import YAML
from ruamel.yaml import comments
from ._module_mapping import PACKAGE_NAME_TO_IMPORT_NAME
import hashlib

import inspect

_DEFAULT_PROTOCOL = 2
_DEPENDENCY_FILE = "requirements.txt"
_CODE_DIR = "code"
_WEIGHTS_DIR = "weight"
_TAGS_DIR = "tag"

yaml = YAML()

def _replace_invalid_char(name):
    return name.replace('-', '_').replace(' ', '_')

def save(
    model:          torch.nn.Module,
    save_path:     str,
    
    entry_name:     str,
    spec_name:      str,
    code_path:      str, # path to SOURCE_CODE_DIR or hubconf.py

    metadata:       dict, 
    tags:           dict = None,
    ignore_files:   Sequence = None,
    save_arch:      bool = False,
    overwrite:      bool = False,
):
    entry_name = _replace_invalid_char(entry_name)
    if spec_name is not None:
        spec_name = _replace_invalid_char(spec_name)

    if not os.path.isdir(save_path):
        overwrite = True

    save_path = os.path.abspath(save_path)
    export_code_path = os.path.join(save_path, _CODE_DIR)
    export_weights_path = os.path.join(save_path, _WEIGHTS_DIR)
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(export_code_path, exist_ok=True)
    os.makedirs(export_weights_path, exist_ok=True)

    code_path = os.path.abspath(code_path)
    if os.path.isdir( code_path ):
        if overwrite:
            shutil.rmtree(export_code_path)
            _copy_file_or_tree(src=code_path, dst=export_code_path) # overwrite old files
    elif code_path.endswith('.py'):
        if overwrite:
            shutil.copy2(src=code_path, dst=os.path.join( export_code_path, 'hubconf.py' )) # overwrite old files

    if hasattr(model, 'SETUP_INFO'):
        del model.SETUP_INFO
    if hasattr(model, 'METADATA'):
        del model.METADATA

    model_and_metadata = {
        'model': model if save_arch else model.state_dict(),
        'metadata': metadata,
    }
    temp_pth = os.path.join(export_weights_path, 'temp.pth')
    torch.save(model_and_metadata, temp_pth)
    if spec_name is None:
        _md5 = md5( temp_pth )
        spec_name = _md5
    shutil.move( temp_pth, os.path.join(export_weights_path, '%s-%s.pth'%(entry_name, spec_name)) )

    if tags is not None:
        save_tags( tags, save_path, entry_name, spec_name )

def list_entry(path):
    path = os.path.abspath( os.path.expanduser( path ) )
    if path.endswith('.py'):
        code_dir = os.path.dirname( path )
        entry_file = path
    elif os.path.isdir(path):
        code_dir = os.path.join( path, _CODE_DIR )
        entry_file = os.path.join(path, _CODE_DIR, 'hubconf.py')
    else:
        raise NotImplementedError
    sys.path.insert(0, code_dir)
    spec = importlib.util.spec_from_file_location('hubconf', entry_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    entry_list = [ ( f, getattr(module, f) ) for f in dir(module) if callable(getattr(module, f)) and not f.startswith('_') ]
    sys.path.remove(code_dir)
    return entry_list

def list_spec(path, entry_name=None):
    path = os.path.abspath( os.path.expanduser( path ) )
    weight_dir = os.path.join( path, _WEIGHTS_DIR )
    spec_list = [ f.split('-') for f in os.listdir( weight_dir ) if f.endswith('.pth')]
    spec_list = [ (f[0], f[1][:-4]) for f in spec_list ]
    if entry_name is not None:
        spec_list = [ s for s in spec_list if s[0]==entry_name ]
    return spec_list

def load(path: str, entry_name:str=None, spec_name: str=None, pretrained=True, **kwargs):
    """
    check dependencies and load pytorch models.
    Args:
        path: path to the model package
        pretrained: load the pretrained model
    """
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError

    code_dir = os.path.join(path, _CODE_DIR)
    if os.path.isdir(path):
        cwd = os.getcwd()
        os.chdir(code_dir)
        sys.path.insert(0, code_dir)
        spec = importlib.util.spec_from_file_location(
            'hubconf', os.path.join(code_dir, 'hubconf.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if hasattr( module, 'dependencies' ):
            deps = getattr( module, 'dependencies')
            for dep in deps:
                _import_with_auto_install(dep)

        if entry_name is None: # auto select
            pth_file = [pth for pth in os.listdir( os.path.join(path, _WEIGHTS_DIR) )]
            assert len(pth_file)<=1, "Loading models with more than one weight files (.pth) is ambiguous"
            pth_file = pth_file[0]
            entry_name = pth_file.split('-')[0]
            entry_fn = getattr( module, entry_name )
        else:
            entry_fn = getattr( module, entry_name )
            if spec_name is None:
                pth_file = [pth for pth in os.listdir( os.path.join(path, _WEIGHTS_DIR) ) if pth.startswith(entry_name) ]
                assert len(pth_file)<=1, "Loading models with more than one weight files (.pth) is ambiguous"
                pth_file = pth_file[0]
            else:
                pth_file = '%s-%s.pth'%(entry_name, spec_name)
        
        try:
            model_and_metadata = torch.load(os.path.join(path, _WEIGHTS_DIR, pth_file), map_location='cpu' )
        except: raise FileNotFoundError

        if isinstance( model_and_metadata['model'], torch.nn.Module ):
            model = model_and_metadata['model']
        else:
            entry_args = model_and_metadata['metadata']['entry_args']
            if entry_args is None:
                entry_args = dict()
            model = entry_fn( **entry_args )
            if pretrained:
                model.load_state_dict( model_and_metadata['model'] )
        
        # setup metadata and atlas info
        model.METADATA = model_and_metadata['metadata']
        model.SETUP_INFO = {"path": path, "entry_name": entry_name}
        sys.path.pop(0)
        os.chdir(cwd)
        return model
    raise NotImplementedError

def load_metadata(path, entry_name=None, spec_name=None):
    path = os.path.abspath(path)
    if os.path.isdir(path):
        if entry_name is None: # auto select
            pth_file = [pth for pth in os.listdir( os.path.join(path, _WEIGHTS_DIR) )]
            assert len(pth_file)<=1, "Loading models with more than one weight files (.pth) is ambiguous"
            pth_file = pth_file[0]
        else:
            if spec_name is None:
                pth_file = [pth for pth in os.listdir( os.path.join(path, _WEIGHTS_DIR) ) if pth.startswith(entry_name) ]
                assert len(pth_file)<=1, "Loading models with more than one weight files (.pth) is ambiguous"
                pth_file = pth_file[0]
            else:
                pth_file = '%s-%s.pth'%(entry_name, spec_name)
        try:
            metadata = torch.load( os.path.join( path, _WEIGHTS_DIR, pth_file ), map_location='cpu' )['metadata']
        except:
            FileNotFoundError
        return metadata
    
def load_tags(path, entry_name, spec_name):
    path = os.path.abspath(path)
    tags_path = os.path.join(path, _TAGS_DIR, '%s-%s.yml'%( entry_name, spec_name ))
    if os.path.isfile(tags_path):
        return _to_python_type(_yaml_load(tags_path))
    return dict()

def save_tags(tags, path, entry_name, spec_name):
    path = os.path.abspath(path)
    if tags is None:
        tags = {}
    tags_path = os.path.join(path, _TAGS_DIR, '%s-%s.yml'%( entry_name, spec_name ))
    os.makedirs( os.path.join( path, _TAGS_DIR ), exist_ok=True )
    _yaml_dump(tags_path, tags)

def _yaml_dump(f, obj):
    with open(f, 'w') as f:
        yaml.dump(obj, f)
        
def _yaml_load(f):
    with open(f, 'r') as f:
        return yaml.load(f)

def _to_python_type(data):
    if isinstance(data, dict):
        for k, v in data.items():
            data[k] = _to_python_type(v)
        return dict(data)
    elif isinstance(data, comments.CommentedSeq ):
        for idx, v in enumerate(data):
            data[idx] = _to_python_type(v)
        return list(data)
    return data

def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def _get_package_name_and_version(package):
    _version_sym_list = ('==', '>=', '<=')
    for sym in _version_sym_list:
        if sym in package:
            return package.split(sym)
    return package, None

def _import_with_auto_install(package):
    package_name, version = _get_package_name_and_version(package)
    package_name = package_name.strip()
    import_name = PACKAGE_NAME_TO_IMPORT_NAME.get(
        package_name, package_name).replace('-', '_')
    try:
        return __import__(import_name)
    except ImportError:
        try:
            pipmain.main(['install', package])
        except:
            pipmain(['install', package])
    return __import__(import_name)

from distutils.dir_util import copy_tree
def _copy_file_or_tree(src, dst):
    if os.path.isfile(src):
        shutil.copy2(src=src, dst=dst)
    else:
        copy_tree(src=src, dst=dst)

def _glob_list(path_list, recursive=False):
    results = []
    for path in path_list:
        if '*' in path:
            path = list(glob(path, recursive=recursive))
            results.extend(path)
        else:
            results.append(path)
    return results
