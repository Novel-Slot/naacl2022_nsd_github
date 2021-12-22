import random
import string
import csv
import os
import re
from math import log
from collections import Counter
import heapq
import copy
import numpy as np
from collections import  Counter
import pandas as pd
import json

def load_data(data_in = "seq.in",data_out = "seq.out",datapath=None):
    """ 
    读取seq.in 和 seq.out 文件，返回 text_list 和 label_list 两个列表 ; token_list 和true_iob_list两个列表 
    label_list = [ 'O O O O O O B-round_trip I-round_trip O O B-fromloc.city_name I-fromloc.city_name O B-toloc.city_name', 'O O O O O O O B-fromloc.city_name O B-toloc.city_name',"..."]
    iob_list = [["O","B-xx","I-xx"],[],...]
    """
    text_list = []
    label_list = []
    with open(os.path.join(datapath,data_in),"r") as file_in:
        for line in file_in.readlines():
            line = line.strip("\n")
            text_list.append(line)
    with open(os.path.join(datapath,data_out),"r") as file_out:
        for line_out in file_out.readlines():
            line_out = line_out.strip("\n")
            label_list.append(line_out)
    token_list = []
    iob_list = []
    for text in text_list:
        token_list.append(text.split())
    for truelabel in label_list:
        iob_list.append(truelabel.split())
    return text_list,label_list,token_list,iob_list

def count_token(iob_list):
    """ 统计数据集中所有token类别、类别数量、以及总数量(token级) """
    slots = []
    for seq in range(len(iob_list)):
        for j in range(len(iob_list[seq])):
            if iob_list[seq][j] == "O":
                slots.append("O")
            else:
                slots.append(iob_list[seq][j][2:])
    token_number = len(slots)
    count_tag = dict(Counter(slots))
    tag_list = count_tag.keys()
    count_dict = {
        "token total num":token_number,
        "tag list":tag_list,
        "tag and its num": count_tag
    }
    return count_dict

def count_intent_and_tag(datapath):
    """ 统计数据集中每个意图包含的槽标签及其数量 """
    text_list = []
    label_list = []
    intent_list = [] 
    iob_list = []
    with open(os.path.join(datapath,"seq.in"),"r") as file_in:
        for line in file_in.readlines():
            line = line.strip("\n")
            text_list.append(line)
    with open(os.path.join(datapath,"seq.out"),"r") as file_out:
        for line_out in file_out.readlines():
            line_out = line_out.strip("\n")
            label_list.append(line_out)
    with open(os.path.join(datapath,"label"),"r") as file_in:
        for line in file_in.readlines():
            line = line.strip("\n")
            intent_list.append(line)
    intents = list(set(intent_list)) # 意图类别
    for truelabel in label_list:
        iob_list.append(truelabel.split())
    intent_and_slot = {}
    all_slots = []
    for intent in intents:
        select_seq_idx = [i for i,x in enumerate(intent_list) if x==intent]
        slots = []
        for seq in select_seq_idx:
            for j in range(len(iob_list[seq])):
                if iob_list[seq][j][0] == "B":
                    slots.append(iob_list[seq][j][2:])
        all_slots += slots
        intent_and_slot[intent] = {}
        intent_and_slot[intent]["num_sentences"] = len(select_seq_idx)
        intent_and_slot[intent]["all_slottypes"] = list(set(slots))
        intent_and_slot[intent]["num_slottypes"] = len(list(set(slots)))
        intent_and_slot[intent]["details"] = dict(Counter(slots))
    print("数据集：",datapath)
    print("数据集中意图和槽位的对应关系：\n",intent_and_slot)
    with open(os.path.join(datapath,"metrics-intent_and_slot.json"),"w",encoding = "utf8") as fp:
            json.dump(intent_and_slot,fp,ensure_ascii = False)

    # 写入excel文件
    print("数据集中意图和槽位的对应关系：\n")
    f = open(os.path.join(datapath,"metrics-intent_and_slot.csv"),'w',encoding='utf-8')
    csv_writer = csv.writer(f)
    csv_writer.writerow(["-"]+intents)
    slot_types = list(set(all_slots))
    for slot in slot_types:
        slot_intent_md = [intent_and_slot[intent]["details"][slot] if slot in intent_and_slot[intent]["all_slottypes"] else "-" for intent in intents]
        slot_intent_md.insert(0,slot)
        print(slot_intent_md)
        csv_writer.writerow(slot_intent_md)
    f.close()
    return intent_and_slot

def exam_BB(datapath):
    """ 检查每句话中有没有存在B-tagx B-tagx的情况 """
    label_list = []
    with open(os.path.join(datapath,"seq.out"),"r") as file_out:
        for line_out in file_out.readlines():
            line_out = line_out.strip("\n")
            label_list.append(line_out.split())
    print(label_list[:2])
    import collections
    for seq in label_list:
        seq_deque = collections.deque(seq)
        tag = seq_deque.popleft()
        for i in range(1,len(seq)-1):
            next = seq_deque.popleft()
            if tag[0] == "B" and tag == next:
                print("存在："," ".join(seq))
            elif tag[0] == "I" and tag[2:] == next[2:]:
                print("存在："," ".join(seq))
            
        



# def write_seq_file(self,text_list,iob_list,file_dir,infile_name = "seq.in",outfile_name="seq.out"):
#     if not os.path.exists(file_dir):
#         os.makedirs(file_dir)
#     with open(os.path.join(file_dir,infile_name), "w") as f_in:
#         for num_in in range(len(text_list)):
#             f_in.write(text_list[num_in] + "\n")
#     with open(os.path.join(file_dir,outfile_name), "w") as f_out:
#         for num_out in range(len(iob_list)):
#             f_out.write(iob_list[num_out] + "\n")

# def transform_to_eo(seq_text_list,seq_iob_list,output_path):
#     """
#     Change "O" to "O" , "B-X" to "B-entity" , "I-X" to "I-entity"
#     Attention : There is "B-entity B-entity" situation
#     """
#     assert len(seq_text_list) == len(seq_iob_list)
#     token_text_list = [seq.split() for seq in seq_text_list]
#     token_iob_list = [seq.split() for seq in seq_iob_list]
#     eo_label_list_seq = []
#     # eo_token_list_seq = []
#     for m in range(len(token_iob_list)):
#         eo_label_list = []
#         eo_token_list = []
#         assert len(token_text_list[m]) == len(token_iob_list[m])
#         for n in range(len(token_iob_list[m])):
#             if token_iob_list[m][n] == "O":
#                 eo_label_list.append("O")
#                 # eo_token_list.append(token_text_list[m][n])
#             else:
#                 if re.search("^B-",token_iob_list[m][n]):
#                     eo_label_list.append("B-entity")
#                 elif re.search("^I-",token_iob_list[m][n]):
#                     eo_label_list.append("I-entity")
#                 # eo_token_list.append(token_text_list[m][n])
#         eo_label_list_seq.append(" ".join(eo_label_list))
#         # eo_token_list_seq.append(" ".join(eo_token_list))
#     self.write_seq_file(seq_text_list,eo_label_list_seq,output_path)
#     return seq_text_list,eo_label_list_seq
if __name__ == '__main__':
    # load test data

    # count_intent_and_tag(datapath = "./original/snips/multi/valid")
    exam_BB(datapath = "./original/snips/multi/test")