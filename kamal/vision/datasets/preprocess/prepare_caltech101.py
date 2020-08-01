
import os, sys
from glob import glob
import random 
import argparse
from PIL import Image

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='../101_ObjectCategories')
    parser.add_argument('--test_split', type=float, default=0.3)
    args = parser.parse_args()

    SAVE_DIR = os.path.join( os.path.dirname(args.data_root), 'caltech101_data' )
    if not os.path.exists(SAVE_DIR):
        os.mkdir(SAVE_DIR)

    # Train
    TRAIN_DIR = os.path.join( SAVE_DIR, 'train' )
    if not os.path.exists(TRAIN_DIR):
        os.mkdir(TRAIN_DIR)
    
    # Test
    TEST_DIR = os.path.join( SAVE_DIR, 'test' )
    if not os.path.exists(TEST_DIR):
        os.mkdir(TEST_DIR)
    
    img_folders = os.listdir(args.data_root)
    img_folders.sort()

    for folder in img_folders:
        if folder=='Faces':
            continue
        print('Processing %s'%(folder))
        
        img_paths = glob(os.path.join( args.data_root, folder, '*.jpg')  )
        img_name = [os.path.split(p)[-1] for p in img_paths]

        random.shuffle(img_name)

        img_n = len(img_name)
        test_n = int(args.test_split * img_n)
        
        test_set = img_name[:test_n]
        train_set = img_name[test_n:]

        # test
        dst_path = os.path.join(TEST_DIR, folder)
        if not os.path.exists(dst_path):
            os.mkdir(dst_path)
        for test_name in test_set:
            img = Image.open(os.path.join( args.data_root, folder, test_name ))
            img.save( os.path.join(dst_path, test_name ) )

        # train
        dst_path = os.path.join(TRAIN_DIR, folder)
        if not os.path.exists(dst_path):
            os.mkdir(dst_path)
        for train_name in train_set:
            img = Image.open(os.path.join( args.data_root, folder, train_name ))
            img.save( os.path.join(dst_path, train_name ) )