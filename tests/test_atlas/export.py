from atlas import meta, serialize
import torch
import importlib

if __name__=='__main__':
    entry_file = "atlas_entry.py"
    module = importlib.import_module(entry_file.split('.')[0])
    model = module.AtlasEntry.init()

    model.load_state_dict( torch.load( 'checkpoints/mnist-lenet-best-00007973-acc-0.992.pth') )
    
    metadata = meta.MetaData(
                    name='lenet5_mnist',
                    dataset='mnist',
                    task=meta.TASK.CLASSIFICATION,
                    url='https://vipazoo.com',
                    input=meta.ImageInput(
                        size=[32, 32],
                        range=[0, 1],
                        space='gray',
                    ),
                    other_metadata=dict(num_classes=10),
            )
    
    serialize.export(model,
                    entry_file=entry_file,  # AtlasEntry实现文件
                    export_path='../lenet5_mnist_classification', 
                    metadata=metadata,
                    ignore=['checkpoints', 'export.py'] )
    


