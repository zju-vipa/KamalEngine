# coding:utf-8

import os
import sys
import shutil
import random

import numpy as np
from scipy.io import loadmat

def copy_whole_image(img_file, image2category, src, dst, dataset):
    if dataset == 'dog' or 'cub':
        for img, c in zip(img_file, image2category):
            src_file = os.path.join(src, c,img)
            dst_file = os.path.join(dst, c, img)
            shutil.copy(src_file, dst_file)
    elif dataset == 'airplane' or 'car':
        for img, c in zip(img_file, image2category):
            src_file = os.path.join(src,img)
            dst_file = os.path.join(dst, c, img)
            shutil.copy(src_file, dst_file)

def copy_part_image(img_file, image2category, src, dst, part): 
    for img, c in zip(img_file, image2category):
        if c in part:
            src_file = os.path.join(src, c, img)
            dst_file = os.path.join(dst, c, img)
            shutil.copy(src_file, dst_file)

def make_whole_categories_dirs(data_root,categories_list):
    #---------------创建images_whole/train，images_whole/test
    whole_train_root = os.path.join(data_root, 'images_whole/train')
    if not os.path.exists(whole_train_root):
        os.mkdir(whole_train_root)
    whole_test_root = os.path.join(data_root, 'images_whole/test')   
    if not os.path.exists(whole_train_root):
        os.mkdir(whole_test_root)
    # --------------- copy 类别文件夹 ---------------
    for f in categories_list:
        dir_name = os.path.join(whole_train_root, f)
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        dir_name = os.path.join(whole_test_root, f)
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
    return whole_train_root,whole_test_root

def make_part_categories_dirs(data_root,parts_categories,part_id):
    #---------------创建img_part{}/train，img_part/test
    part_root = os.path.join(data_root, 'img_part{}'.format(part_id+ 1))
    if not os.path.exists(part_root):
        os.mkdir(part_root)
    part_train_root = os.path.join(part_root,'train')
    if not os.path.exists(part_train_root):
        os.mkdir(part_train_root)
    part_test_root = os.path.join(part_root,'test')   
    if not os.path.exists(part_test_root):
        os.mkdir(part_test_root)
    # --------------- copy 类别文件夹 ---------------
    for f in parts_categories[part_id]:
        dir_name = os.path.join(part_train_root, f)
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        dir_name = os.path.join(part_test_root, f)
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
    return part_train_root,part_test_root
        
def split_whole_train_test(data_root,dataset):
    
    if dataset =='airplane':
        # --------------- 读取train/test的split---------------
        with open(data_root +'/data/images_variant_trainval.txt', 'r') as f:
            lines = f.readlines()
            train_file_list= [line.strip().split(' ')[0]+'.jpg' for line in lines]
            train_categories = [line.strip()[len(line.strip().split(' ')[0])+1:].replace('/','-').replace(' ','-') for line in lines]

        # print images
        with open(data_root +'/data/images_variant_test.txt') as f:
            lines = f.readlines()
            test_file_list = [line.strip().split(' ')[0]+'.jpg' for line in lines]
            test_categories = [line.strip()[len(line.strip().split(' ')[0])+1:].replace('/','-').replace(' ','-') for line in lines]
        train_src_root = data_root +'/data/images'
        test_src_root = data_root +'/data/images'
       
    elif dataset =='car':
         # --------------- 读取train/test的split---------------
        train_list = loadmat(data_root + '/devkit/cars_train_annos.mat')
        train_file_list = [item['fname'] for item in train_list['annotations']]
        train_file_list = train_file_list[0].tolist()
        train_file_list = [item[0] for item in train_file_list]

        train_categories = [item['class'] for item in train_list['annotations']]
        train_categories = train_categories[0].tolist()
        train_categories = [item[0][0] for item in train_categories] #1~196
        # -------------------------------------------
        test_list = loadmat(data_root +'/devkit/cars_test_annos_withlabels.mat')
        test_file_list = [item['fname'] for item in test_list['annotations']]
        test_file_list = test_file_list[0].tolist()
        test_file_list = [item[0] for item in test_file_list]

        test_categories = [item['class'] for item in test_list['annotations']]
        test_categories = test_categories[0].tolist()
        test_categories = [item[0][0] for item in test_categories] #1~196
        # print test_categories
        # -------------------------------------------
        meta_list = loadmat(data_root +'/devkit/cars_meta.mat')
        categories = meta_list['class_names'][0]
        categories = [c[0].replace(' ', '-').replace('/', '-') for c in categories]
        
        train_categories = [categories[i-1] for i in train_categories]
        test_categories = [categories[i-1] for i in test_categories]

        train_src_root = data_root + '/cars_train'
        test_src_root = data_root +'/cars_test'

        
    elif dataset =='dog':
        # --------------- 读取train/test的split---------------
        train_list = loadmat('./data/dog/train_list.mat')
        train_file_list = train_list['file_list']
        train_file_list = [i[0][0] for i in train_file_list]
        train_categories = [i[0][0][:i[0][0].index('/')] for i in train_file_list]  
        # print train_file_list

        test_list = loadmat('./data/dog/test_list.mat')
        test_file_list = test_list['file_list']
        test_file_list = [i[0][0][i[0][0].index('/'):] for i in test_file_list]
        test_categories = [i[0][0][:i[0][0].index('/')] for i in test_file_list]
        
        train_src_root = data_root + '/Images'
        test_src_root = data_root +  '/Images'
        
    elif dataset =='cub':
         # --------------- 读取train/test的split---------------
        with open(data_root +'/data/images_trainval.txt', 'r') as f:
            lines = f.readlines()
            train_file_list= [line.strip().split('/')[1] for line in lines]
            train_categories = [line.strip().split('/')[0]for line in lines]

        # print images
        with open(data_root +'/data/images_test.txt', 'r') as f:
            lines = f.readlines()
            test_file_list = [line.strip().split('/')[1] for line in lines]
            test_categories = [line.strip().split('/')[0]for line in lines]
        
        train_src_root = data_root + '/Images'
        test_src_root = data_root +  '/Images'
    
    train_dst_root, test_dst_root = make_whole_categories_dirs(data_root,train_categories)

    copy_whole_image(train_file_list, train_categories,train_src_root, train_dst_root, dataset)
    copy_whole_image(test_file_list, test_categories, test_src_root, test_dst_root, dataset)

    return train_list,train_categories,test_list,test_categories

def produce_parts_categories(categories_root,n):
    categories = os.listdir(categories_root)
    random.seed(0)
    random.shuffle(categories)

    categories_part = round(len(categories)/n)
    parts = []
    for i in range(0,n):
        parts[i]=categories[i*categories_part:(i+1)*categories_part]

        with open('shuffle_category_part{}.txt'.format(i+1), 'w') as f:
            for c in parts[i]:
                f.write(c + '\n') 
    return parts
    
def split_part_train_test(data_root,n_parts,train_file_list,train_categories,test_file_list,test_categories):
    whole_train_root = os.path.join(data_root, '/images_whole/train')
    whole_test_root = os.path.join(data_root, '/images_whole/test')
    # ------------------produce part categories-----------------------------
    parts_categories = produce_parts_categories(whole_train_root,n_parts)
    for id in range(0,n_parts):
        part_train_dst_root,part_test_dst_root =make_part_categories_dirs(data_root,parts_categories,id)
        #----------------------copy image------------------------------
        copy_part_image(train_file_list, train_categories, whole_train_root, part_train_dst_root, parts_categories[id])
        copy_part_image(test_file_list, test_categories, whole_test_root, part_test_dst_root, parts_categories[id])




if __name__ == '__main__':
    dataset = 'car'
    n_parts = 4
    data_root = './data/'+dataset
    # 将原数据划分到 'images_whole/train'和'images_whole/test',train和test文件夹下是各category的子文件夹
    train_file_list,train_categories,test_file_list,test_categories = split_whole_train_test(data_root,dataset)

    # train、test 下的文件划分为 'partxx/train'和'partxx/test',train和test文件夹下是相应part下category的子文件夹
    split_part_train_test(data_root,n_parts,train_file_list,train_categories,test_file_list,test_categories)

   