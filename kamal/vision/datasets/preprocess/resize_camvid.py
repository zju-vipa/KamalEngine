import argparse
import os
from PIL import Image
import numpy as np 

SIZE=(480, 320) # W, H

def is_image(path: str):
    return path.endswith( 'png' ) or path.endswith('jpg') or path.endswith( 'jpeg' )

def copy_and_resize( src_dir, dst_dir, resize_fn ):

    for file_or_dir in os.listdir( src_dir ):
        src = os.path.join( src_dir, file_or_dir )
        dst = os.path.join( dst_dir, file_or_dir )
        if os.path.isdir( src ):
            os.mkdir( dst )
            copy_and_resize( src, dst, resize_fn )
        elif is_image( src ):
            print(src, ' -> ', dst)
            image = Image.open(  src  ) 
            resized_image = resize_fn(image)
            resized_image.save( dst )

def resize_input( image: Image.Image ):
    return image.resize( SIZE, resample=Image.BICUBIC )

def resize_seg( image: Image.Image ):
    return image.resize( SIZE, resample=Image.NEAREST )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    ROOT = parser.parse_args().root
    NEW_ROOT = os.path.join( ROOT, '%d_%d'%(*SIZE) )
    os.mkdir(NEW_ROOT, )
    for split in ['train', 'val', 'test']:
        IMG_DIR = os.path.join( ROOT, split )
        GT_DIR = os.path.join( ROOT, split+'annot' )
        NEW_IMG_DIR = os.path.join( NEW_ROOT, split )
        NEW_GT_DIR = os.path.join( NEW_ROOT, split+'annot' )
        
        os.mkdir( NEW_IMG_DIR )
        os.mkdir( NEW_GT_DIR )
        
        copy_and_resize( IMG_DIR, NEW_IMG_DIR, resize_input )
        copy_and_resize( GT_DIR, NEW_GT_DIR, resize_seg )




if __name__=='__main__':
    main()
