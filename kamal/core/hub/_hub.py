from pip._internal import main as pipmain
from typing import Sequence, Optional

import torch
import os, sys, re
import shutil, inspect
import importlib.util
from glob import glob

from . import pickle_module
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

def save(
    model:          torch.nn.Module,
    export_dir:     str,
    
    entry_name:     str,
    spec_name:      str,
    code_path:      str, # path to SOURCE_CODE_DIR or hubconfig.py

    metadata:       dict, 
    tags:           dict = None,
    ignore_files:   Sequence = None,
    save_arch:      bool = False,
    overwrite:      bool = False,
):
    entry_name = entry_name.replace('-', '_')
    if spec_name is not None:
        spec_name = spec_name.replace('-', '_')

    export_dir = os.path.abspath(export_dir)
    export_code_path = os.path.join(export_dir, _CODE_DIR)
    export_weights_path = os.path.join(export_dir, _WEIGHTS_DIR)
    
    if not os.path.isdir(export_dir):
        overwrite = True

    os.makedirs(export_dir, exist_ok=True)
    os.makedirs(export_code_path, exist_ok=True)
    os.makedirs(export_weights_path, exist_ok=True)
    
    code_path = os.path.abspath(code_path)
    if os.path.isdir( code_path ):
        code_files = [os.path.join(code_path, f)for f in os.listdir(code_path)]
        if ignore_files is not None:
            ignore_files = [os.path.abspath(f) for f in _glob_list( ignore_files, recursive=True )]
            for f in ignore_files:
                if f in code_files:
                    code_files.remove(f)
        if overwrite:
            for f in code_files:
                _copy_file_or_tree(src=f, dst=export_code_path) # overwrite old files

    elif code_path.endswith('.py'):
        if overwrite:
            shutil.copy2(src=code_path, dst=os.path.join( export_code_path, 'atlas_entry.py' )) # overwrite old files

    #if isinstance(metadata, dict):
    #    metadata.pop( 'entry_name', None )
    
    model_and_metadata = {
        'model': model if save_arch else model.state_dict(),
        'metadata': metadata,
    }
    temp_pth = os.path.join(export_weights_path, 'temp.pth')
    torch.save(model_and_metadata, temp_pth)
    _md5 = md5( temp_pth )
    if spec_name is None:
        spec_name = _md5
    shutil.move( temp_pth, os.path.join(export_weights_path, '%s-%s.pth'%(entry_name, spec_name)) )

    if tags is not None:
        save_tags( tags, export_dir, entry_name, spec_name )

def list_entry(path):
    path = os.path.abspath( os.path.expanduser( path ) )
    if path.endswith('.py'):
        code_dir = os.path.dirname( path )
        entry_file = path
    elif os.path.isdir(path):
        code_dir = os.path.join( path, _CODE_DIR )
        entry_file = os.path.join(path, _CODE_DIR, 'atlas_entry.py')
    else:
        raise NotImplementedError
    sys.path.insert(0, code_dir)
    spec = importlib.util.spec_from_file_location('atlas_entry', entry_file)
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
        sys.path.insert(0, code_dir)

        spec = importlib.util.spec_from_file_location(
            'atlas_entry', os.path.join(code_dir, 'atlas_entry.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if hasattr( module, 'dependencies' ):
            deps = getattr( module, 'dependencies')
            for dep in deps:
                _import_with_auto_install(dep)

        if entry_name is None and spec_name is None: # auto select
            pth_file = [pth for pth in os.listdir( os.path.join(path, _WEIGHTS_DIR) )]
            assert len(pth_file)<=1, "There should be only one model if entry_name==None and spec_name==None"
            pth_file = pth_file[0]
            entry_name = pth_file.split('-')[0]
            entry_fn = getattr( module, entry_name )
        elif entry_name is not None and spec_name is not None:
            entry_fn = getattr( module, entry_name )
            pth_file = '%s-%s.pth'%(entry_name, spec_name)
        else:
            raise NotImplementedError

        model_and_metadata = torch.load(os.path.join(path, _WEIGHTS_DIR, pth_file), map_location='cpu' )
        if isinstance( model_and_metadata['model'], torch.nn.Module ):
            model = model_and_metadata['model']
        else:
            entry_args = model_and_metadata['metadata']['entry_args']
            if entry_args is None:
                entry_args = dict()
            model = entry_fn( **entry_args )
            if pretrained:
                model.load_state_dict( model_and_metadata['model'] )
        model_and_metadata['metadata']['entry_name'] = entry_name
        model.METADATA = model_and_metadata['metadata']
        model.ATLAS_INFO = {"path": path, "entry_name": entry_name}
        return model
    raise NotImplementedError

def load_metadata(path, entry_name=None, spec_name=None):
    path = os.path.abspath(path)
    if os.path.isdir(path):
        if entry_name is None and spec_name is None: # auto select
            pth_file = [pth for pth in os.listdir( os.path.join(path, _WEIGHTS_DIR) )]
            assert len(pth_file)<=1, "There should be only one model if entry_name==None and spec_name==None"
            pth_file = pth_file[0]
        else:
            pth_file = '%s-%s.pth'%( entry_name, spec_name )
        metadata = torch.load( os.path.join( path, _WEIGHTS_DIR, pth_file ), map_location='cpu' )['metadata']
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

def _extract_module_source(net):
    dep = set()
    code = set()
    mod = inspect.getmodule(net)
    src_path = inspect.getsourcefile(mod)
    code.add(src_path)
    for name, obj in inspect.getmembers(mod):
        if inspect.ismodule(obj) or inspect.isfunction(obj) or inspect.isclass(obj):
            module_name = inspect.getmodule(obj).__name__
            if '.' in module_name:
                module_name = module_name.split('.')[0]
            try:
                module_path = inspect.getfile(obj)
            except:
                print('{!r} is a built-in module'.format(obj))
                pass
            if re.search('python\d.\d', module_path):
                if 'site-packages' in module_path:
                    dep.add(module_name)
            else:
                code.add(module_path)
    return code, dep


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
    if os.path.isdir(src):
        return copy_tree( src, dst )
    else:
        return shutil.copy2(src=src, dst=dst)

    #base_path = os.getcwd()
    #dst_subpath = src[len(base_path)+1:]
    #dst_path = os.path.join(dst, dst_subpath)
    #if os.path.isfile(src):
    #    dst_dirpath = os.path.dirname(dst_path)
    #    if not os.path.exists(dst_dirpath):
    #         os.makedirs(dst_dirpath)
    #    shutil.copy(src=src, dst=dst_path)
    #else:
    #    shutil.copytree(src=src, dst=dst_path)
    #return dst_subpath

def _glob_list(path_list, recursive=False):
    results = []
    for path in path_list:
        if '*' in path:
            path = list(glob(path, recursive=recursive))
            results.extend(path)
        else:
            results.append(path)
    return results
