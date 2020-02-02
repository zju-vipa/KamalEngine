import argparse
import os
from PIL import Image
import numpy as np 

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
    return image.resize( (640, 320), resample=Image.BICUBIC )

def resize_seg( image: Image.Image ):
    return image.resize( (640, 320), resample=Image.NEAREST )

def resize_depth( image: Image.Image ):
    return image.resize( (640, 320), resample=Image.NEAREST )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True)

    ROOT = parser.parse_args().root
    IMG_DIR = os.path.join( ROOT, 'leftImg8bit' )
    GT_DIR = os.path.join( ROOT, 'gtFine' )
    DEPTH_DIR = os.path.join( ROOT, 'disparity' )

    NEW_ROOT = os.path.join( ROOT, '320_640' )
    NEW_IMG_DIR = os.path.join( NEW_ROOT, 'leftImg8bit' )
    NEW_GT_DIR = os.path.join( NEW_ROOT, 'gtFine' )
    NEW_DEPTH_DIR = os.path.join( NEW_ROOT, 'disparity' )

    if os.path.exists(NEW_ROOT):
        print("Directory %s existed, please remove it before running this script"%NEW_ROOT)
        return
    
    os.mkdir(NEW_ROOT)
    os.mkdir( NEW_IMG_DIR )
    os.mkdir( NEW_GT_DIR )
    os.mkdir( NEW_DEPTH_DIR )
    
    copy_and_resize( IMG_DIR, NEW_IMG_DIR, resize_input )
    copy_and_resize( GT_DIR, NEW_GT_DIR, resize_seg )
    copy_and_resize( DEPTH_DIR, NEW_DEPTH_DIR, resize_depth )




if __name__=='__main__':
    main()
