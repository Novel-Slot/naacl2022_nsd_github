"""
处理数据集，only、mask、remove三种方式（注意remove方式可能会带来额外的OOD-slot-type,因此这里包括一个"update_name"）
"""
import argparse
from itertools import count
from typing import Any, Union, Dict, Iterable, List, Optional, Tuple
from tqdm.utils import disp_len
from Logger import logger
import logging
from utils import load_data
import os
import json
import random
import numpy as np
import re
from allennlp.data import vocabulary
vocabulary.DEFAULT_OOV_TOKEN = "[UNK]"  # set for bert, for evaluation
def transform_eo_bio(old_dataset):
    """
    Change "O" to "O" , "B-X" to "B-entity" , "I-X" to "I-entity"
    Attention : There is "B-entity B-entity" situation ( Previous method used this )
    """
    train_slots = []
    for idx, concept in enumerate(old_dataset):
        slot_seq = concept[1]
        slot_bbi = []
        for item in slot_seq:
            if item == "O":
                slot_bbi.append("O")
            elif re.search("^B-",item):
                slot_bbi.append("B-entity")
            elif re.search("^I-",item):
                slot_bbi.append("I-entity")
            else:
                raise "!!!!!Wrong!!!!"
        train_slots.append(slot_bbi)
    train_texts,_,train_domains = list(zip(*old_dataset))
    new_dataset = list(zip(train_texts,train_slots,train_domains))
    return new_dataset

def transform_eo_bo(old_dataset):
    """
    Change "O" to "O" , "B-X" to "B-entity" , "I-X" to "I-entity"
    Attention : There is "B-entity B-entity" situation ( Previous method used this )
    """
    train_slots = []
    for idx, concept in enumerate(old_dataset):
        slot_seq = concept[1]
        slot_bbi = []
        for item in slot_seq:
            if item == "O":
                slot_bbi.append("O")
            elif re.search("^B-",item):
                slot_bbi.append("B-entity")
            elif re.search("^I-",item):
                slot_bbi.append("B-entity")
            else:
                raise "!!!!!Wrong!!!!"
        train_slots.append(slot_bbi)
    train_texts,_,train_domains = list(zip(*old_dataset))
    new_dataset = list(zip(train_texts,train_slots,train_domains))
    return new_dataset

def write_seq_file(dataset,file_dir=None):
    """ 将转化后的数据集写入新的文件夹内 """
    print("正在将新数据集写入文件夹中....",file_dir)
    text_list,slot_list,domain_list = zip(*dataset)
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    with open(os.path.join(file_dir,"seq.in"), "w") as f_in:
        for num_in in range(len(text_list)):
            f_in.write(text_list[num_in] + "\n")
    with open(os.path.join(file_dir,"seq.out"), "w") as f_out:
        for num_out in range(len(slot_list)):
            f_out.write(" ".join(slot_list[num_out]) + "\n")
    with open(os.path.join(file_dir,"label.out"), "w") as f_do:
        for domain_out in range(len(domain_list)):
            f_do.write(domain_list[domain_out] + "\n")

def DataProcess(target_domain = None,
                data_path = None,
                write_path = None):   # AddToPlaylist 包含五个槽位，3个seen槽位，2个unseen槽位
    """
    将原始训练集和验证集中属于target_domain的数据去掉，并将数据集转化为BBI(实体/非实体)形式；测试集只保留target_domain的数据，数据集保持不变；
    """
    # 读取原始数据集
    train_set_ori = load_data(os.path.join(data_path,"train")) # [( 'rate richard carvel',['O', 'B-x', 'I-x'], 'RateBook'),(...)...]
    valid_set_ori = load_data(os.path.join(data_path,"valid"))
    test_set_ori = load_data(os.path.join(data_path,"test"))

    # 训练集和验证集：去掉intent=target_domain的数据；
    # 测试集：只保留intent=target_domain的数据；
    train_set = []
    for i,trip in enumerate(train_set_ori):
        if trip[2]!=target_domain:
            train_set.append(train_set_ori[i])
    valid_set = []
    for i,trip in enumerate(valid_set_ori):
        if trip[2]!=target_domain:
            valid_set.append(valid_set_ori[i])
    test_set = []
    for i,trip in enumerate(test_set_ori):
        if trip[2]==target_domain:
            test_set.append(test_set_ori[i])

    # 训练集和验证集转化为BBI形式，测试集保持不变
    train_set_bbi = transform_eo_bo(train_set)
    valid_set_bbi = transform_eo_bo(valid_set)
    test_set_bbi = transform_eo_bo(test_set)

    # 写入新的数据集文件
    write_seq_file(train_set_bbi,file_dir=os.path.join(write_path,"train"))
    write_seq_file(valid_set_bbi,file_dir=os.path.join(write_path,"valid"))
    write_seq_file(test_set_bbi,file_dir=os.path.join(write_path,"test_bbi"))
    write_seq_file(test_set,file_dir=os.path.join(write_path,"test"))

    print("\n","*"*70)
    print("原始训练集示例：")
    print("训练集： ",train_set[0])
    print("新数据集示例：")
    print("训练集： ",train_set_bbi[0])
    print("验证集： ",valid_set_bbi[0])
    print("测试集： ",test_set[0])
    print("原数据集比例 train:valid:test = ",len(train_set_ori),len(valid_set_ori),len(test_set_ori))
    print("现数据集比例 train:valid:test = ",len(train_set),len(valid_set),len(test_set))
    print("\n","*"*70)

    return train_set_bbi,valid_set_bbi,test_set

if __name__ == "__main__":

    # "snips_AddToPlaylist","snips_SearchCreativeWork","snips_RateBook","snips_PlayMusic","snips_SearchScreeningEvent","snips_GetWeather","snips_BookRestaurant"
    # "snips_AddToPlaylist_BO","snips_SearchCreativeWork_BO","snips_RateBook_BO","snips_PlayMusic_BO","snips_SearchScreeningEvent_BO","snips_GetWeather_BO","snips_BookRestaurant_BO"
    slots = ["GetWeather","SearchCreativeWork","RateBook","PlayMusic","SearchScreeningEvent","BookRestaurant"] # "AddToPlaylist"
    for i, dataset in enumerate(["snips_GetWeather_BO","snips_SearchCreativeWork_BO","snips_RateBook_BO","snips_PlayMusic_BO","snips_SearchScreeningEvent_BO","snips_BookRestaurant_BO"]): # "snips_AddToPlaylist_BO"
        target_domain = slots[i]
        data_process = DataProcess(target_domain = target_domain,
                                    data_path = "./data/original/snips/multi/",
                                    write_path = "./data/"+dataset) 