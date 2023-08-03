import os.path as osp
import sys
cur_dir = osp.dirname( __file__ )
main_path = osp.join( cur_dir, '..', '..', '..', '..')
sys.path.insert( 0, main_path )
import argparse
import lmdb
import torch
import os
from PIL import Image
from tqdm import tqdm
from transforms import VisualPrior
import torchvision.transforms.functional as TF
import ipdb
import pickle


parser = argparse.ArgumentParser(description='Viz Single Task')

parser.add_argument('--tasks', dest='tasks')
parser.set_defaults(task='NONE')

parser.add_argument('--img', dest='im_name')
parser.set_defaults(im_name='NONE')

parser.add_argument('--save_dir', dest='save_dir')
parser.set_defaults(save_dir='/KamalEngine/examples/knowledge_amalgamation/soka')

parser.add_argument('--img_dir', dest='img_dir')
parser.set_defaults(img_dir='/KamalEngine/examples/knowledge_amalgamation/soka/wiconisco/rgb/train')

parser.add_argument('--store', dest='store_name')
parser.set_defaults(store_name='./rsa_test_ihlen_200_features_crap')

MAX_SIZE = 30e9

def get_taskonomy_features():
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    file_names = os.listdir(args.img_dir)
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    tasks = args.tasks.split('/')
    print(tasks)
    for task in tasks:
        file = lmdb.open(os.path.join(args.save_dir, task), map_size=int(MAX_SIZE))
        with file.begin() as txn:
            present_entries = [key for key, _ in txn.cursor()]

        for file_name in tqdm(file_names):
            file_path = os.path.join(args.img_dir, file_name)
            image = Image.open(file_path)
            image_name = file_name.split('_rgb')[0]
            o_t = TF.to_tensor(TF.resize(image, 256)) * 2 - 1
            o_t = o_t.unsqueeze_(0).to(device)

            readout, encoder_rep, encoder_mia = VisualPrior.to_get_features_te(o_t, [task], device=device)
            
            ipdb.set_trace()
            if image_name.encode() not in present_entries:
                value = pickle.dumps((readout[0], encoder_rep[0], encoder_mia[0]))
                # value = (readout[0], encoder_rep[0], encoder_mia[0]).tobytes()
                with file.begin(write=True) as txn:
                    txn.put(image_name.encode(), value)
        



if __name__=='__main__':
    get_taskonomy_features()
        