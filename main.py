from types import new_class
from allennlp.models import model
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor
from allennlp.common import Params
from allennlp.data import Vocabulary
from allennlp.data import DataIterator#
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.dataset_readers.dataset_utils.span_utils import bio_tags_to_spans
from allennlp.models import Model
from allennlp.training import Trainer,checkpointer
from allennlp.training.util import evaluate
from allennlp.common.util import prepare_global_logging, cleanup_global_logging, prepare_environment
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer, PretrainedBertIndexer
from allennlp.data import vocabulary
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cluster import KMeans#abc
from sklearn.decomposition import PCA
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.metrics import accuracy_score,confusion_matrix

from models import NSDSlotTaggingModel
from predictors import SlotFillingPredictor
from dataset_readers import MultiFileDatasetReader
from metrics import NSDSpanBasedF1Measure
from utils import *

from collections import Counter
from typing import Any, Union, Dict, Iterable, List, Optional, Tuple
from time import *
import numpy as np
import pandas as pd
import argparse
import os
import logging
import torch

vocabulary.DEFAULT_OOV_TOKEN = "[UNK]"  # set for bert

def parse_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--mode",type=str,choices=["train", "test", "both"], default="test",
                        help="Specify running mode: only train, only test or both.")
    arg_parser.add_argument("--dataset",type=str,choices=["snips_AddToPlaylist","snips_SearchCreativeWork","snips_RateBook","snips_PlayMusic","snips_SearchScreeningEvent","snips_AddToPlaylist","snips_GetWeather","snips_BookRestaurant"], default=None,
                        help="The dataset to use.")
    arg_parser.add_argument("--config_dir",type=str, default=None,
                        help="The config path.")
    arg_parser.add_argument("--output_dir",type=str, default="./output/snips_BIO",
                        help="The output path.")
    arg_parser.add_argument("--cuda",type=int, default=0,
                        help="")
    arg_parser.add_argument("--seed",type=int, default=1,
                        help="")
    arg_parser.add_argument("--threshold", default="auto",
                        help="")
    arg_parser.add_argument("--batch_size",type=int, default=256,
                        help="")
    args = arg_parser.parse_args()

    return args
    
args = parse_args()
setup_seed(args.seed)
# target_slots = ["playlist","music_item","artist","entity_name","playlist_owner"] 

# Train
if args.mode in ["train","both"]:

    output_dir = os.path.join(args.output_dir,args.dataset)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    params = Params.from_file(args.config_dir)
    stdout_handler = prepare_global_logging(output_dir, False)
    prepare_environment(params)

    reader = DatasetReader.from_params(params["dataset_reader"])
    train_dataset = reader.read(file_path=params.pop("train_data_path", None))
    valid_dataset = reader.read(params.pop("valid_data_path", None))
    test_dataset = None
    vocab = Vocabulary.from_instances(train_dataset + valid_dataset)

    # test_data_path = params.pop("test_data_path_bio", None)
    # if test_data_path:
    #     test_dataset = reader.read(test_data_path)
    #     vocab = Vocabulary.from_instances(train_dataset + valid_dataset + test_dataset)
    # else:
    #     print("test_dataset = None")
    #     test_dataset = None
    #     vocab = Vocabulary.from_instances(train_dataset + valid_dataset)
    
    model_params = params.pop("model", None)
    model = Model.from_params(model_params.duplicate(), vocab=vocab)   # 预训练语言模型
    vocab.save_to_files(os.path.join(output_dir, "vocabulary"))
    # copy config file
    with open(args.config_dir, "r", encoding="utf-8") as f_in:
        with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f_out:
            f_out.write(f_in.read())

    iterator = DataIterator.from_params(params.pop("iterator", None))
    iterator.index_with(vocab)
    
    trainer_params = params.pop("trainer", None)
    # checkpointer.from_params()
    trainer = Trainer.from_params(model=model,
                                serialization_dir=output_dir,
                                iterator=iterator,
                                train_data=train_dataset,
                                validation_data=valid_dataset,
                                params=trainer_params.duplicate())
    trainer.train()

    # evaluate on the test set
    if test_dataset:
        logging.info("Evaluating on the test set")
        import torch  # import here to ensure the republication of the experiment
        model.load_state_dict(torch.load(os.path.join(output_dir, "best.th")))
        test_metrics = evaluate(model, test_dataset, iterator,
                                cuda_device=trainer_params.pop("cuda_device", 0),
                                batch_weight_key=None)
        logging.info(f"Metrics on the test set: {test_metrics}")
        with open(os.path.join(output_dir, "test_metrics.txt"), "w", encoding="utf-8") as f_out:
            f_out.write(f"Metrics on the test set: {test_metrics}")
    cleanup_global_logging(stdout_handler)


# Test
if args.mode in ["test","both"]:
    time_begin = time()
    if args.mode == "both":
        model_dir = output_dir
    else:
        model_dir = os.path.join(args.output_dir,args.dataset)

    params = Params.from_file(os.path.join(model_dir,"config.json"))
    vocab = Vocabulary.from_files(os.path.join(model_dir,"vocabulary"))
    test_bio_data_path = params.pop("test_data_path_bio", None)
    test_data_path = params.pop("test_data_path", None)
    train_data_path = params.pop("train_data_path", None)
    print("test_bio_data_path = ",test_bio_data_path)

    # predict
    archive = load_archive(model_dir,cuda_device=args.cuda)
    predictor = Predictor.from_archive(archive=archive, predictor_name="slot_filling_predictor")
    # train_outputs = predictor.predict_multi(file_path = train_data_path ,batch_size = batch_size)
    test_outputs = predictor.predict_multi(file_path = test_data_path ,batch_size = args.batch_size)
    test_bbi_outputs = predictor.predict_multi(file_path = test_bio_data_path ,batch_size = args.batch_size)  # bert输出

    # input_text = "add sabrina salerno salerno to the grime instrumentals playlist" # TODO 用于Debug的样例
    # input_label = "O B-artist I-artist B-artist O O B-playlist I-playlist O".split() # TODO 用于Debug的样例
    # test_bbi_outputs = predictor.predict({"tokens": input_text.split(),"true_labels":input_label}) # TODO 用于Debug的样例

    # 为计算后边指标，扩展vocab
    vocab_labels = list(vocab.get_index_to_token_vocabulary("labels").values())
    pred_vocab_labels = list(set(test_outputs["true_labels"]+test_outputs["predict_labels"]))
    for pred_label in pred_vocab_labels:
        if pred_label not in vocab_labels: vocab.add_token_to_namespace(pred_label,namespace = "labels") 

    print("————————————————————————（span识别）计算在目标领域测试集上实体span的识别指标——————————————————————————")
    print("- 测试集embedding.shape = ",test_bbi_outputs["encoder_outs"].shape)
    print("- 测试集true_labels.shape = ",len(test_bbi_outputs["true_labels"]))
    print("- 测试集predict_labels.shape = ",len(test_bbi_outputs["predict_labels"]))
    print("- 测试集示例: ")
    print("- Ground Truth： ",test_outputs["true_labels"][:10])
    print("- Predict Labels: ",test_outputs["predict_labels"][:10])
    # print(test_outputs["encoder_outs"][:4])
    # Span识别的SpanF1指标
    entity_true_idx_tens = torch.Tensor([[vocab.get_token_index(label, namespace="labels") for label in test_bbi_outputs["true_labels"]]])
    pred_idx = [vocab.get_token_index(label, namespace="labels") for label in test_bbi_outputs["predict_labels"]]  
    pred_idx_tens = np.zeros((1,len(test_bbi_outputs["predict_labels"]), vocab.get_vocab_size("labels")),np.int)
    for token_ in range(pred_idx_tens.shape[1]):
        pred_idx_tens[0,token_,pred_idx[token_]] = 1  
    pred_idx_tens = torch.Tensor(pred_idx_tens)
    spanf1 = NSDSpanBasedF1Measure(
                                    vocabulary = vocab,
                                    tag_namespace="labels",
                                    ignore_classes=[],
                                    label_encoding="BIO",
                                    nsd_slots=["ns"]
                                    )
    spanf1(pred_idx_tens,entity_true_idx_tens) # pred_idx_tens(1,n_samples,n_classes)，true_idx_tens(1,n_samples)
    metric = spanf1.get_metric(reset=True)
    f,r,p = round(metric["f1-entity"]*100,2),round(metric["recall-entity"]*100,2),round(metric["precision-entity"]*100,2)
    print(f"\n在测试集上的eneity指标(span-metrics):  f:{f}, p:{p}, r:{r}")
    #  Span识别的TokenF1指标
    true_labels_bo = [_label if _label=="O" else "B-"+_label[2:] for _label in test_bbi_outputs["true_labels"]]
    pred_labels_bo = [_label if _label=="O" else "B-"+_label[2:] for _label in test_bbi_outputs["predict_labels"]]
    entity_true_idx_tens = torch.Tensor([[vocab.get_token_index(label, namespace="labels") for label in true_labels_bo]])
    pred_idx = [vocab.get_token_index(label, namespace="labels") for label in pred_labels_bo]
    pred_idx_tens = np.zeros((1,len(pred_labels_bo), vocab.get_vocab_size("labels")),np.int)
    for token_ in range(pred_idx_tens.shape[1]):
        pred_idx_tens[0,token_,pred_idx[token_]] = 1
    pred_idx_tens = torch.Tensor(pred_idx_tens)
    spanf1(pred_idx_tens,entity_true_idx_tens)
    metric = spanf1.get_metric(reset=True)
    f,r,p = round(metric["f1-entity"]*100,2),round(metric["recall-entity"]*100,2),round(metric["precision-entity"]*100,2)
    print(f"\n在测试集上的eneity指标(token-metrics):  f:{f}, p:{p}, r:{r}")
    print("-"*90)

    

    # 将同实体token embedding求平均，得到span embedding；
    gold_spans = bio_tags_to_spans(test_outputs["true_labels"], classes_to_ignore  = []) # [('artist', (1, 3)), ('playlist', (5, 6))]
    pred_spans = bio_tags_to_spans(test_outputs["predict_labels"], classes_to_ignore = [])

    # print("encoder_outs.shape = ",test_outputs["encoder_outs"].shape)
    gold_spans_pd = pd.DataFrame(gold_spans,columns=["labels","span_idx_tuple"])
    gold_spans_labels = gold_spans_pd["labels"]
    gold_spans_idxs = gold_spans_pd["span_idx_tuple"]
    # print("gold_span_pd = \n",gold_spans_pd)
    span_embeddings = []
    span_gold_labels = []
    clusters = list(set(gold_spans_labels.tolist()))
    for i,item in enumerate(pred_spans):
        span_start = item[1][0]
        span_end = item[1][1]
        span_embedding = [np.mean(test_outputs["encoder_outs"][span_start:span_end+1,:],0)] # 对 span embedding 求 mean embedding #(256,)
        span_embeddings = np.concatenate((span_embeddings,span_embedding),axis=0) if i > 0 else span_embedding
        if item[1] in list(gold_spans_idxs): # 若span刚好提取正确
            span_index = gold_spans_idxs[gold_spans_idxs.isin([item[1]])].index
            span_gold_labels += gold_spans_labels[span_index].tolist()
        else: # 若span提取错误，则取span中出现频次最高的实体标签 # TODO
            _span_gold_label = Counter(test_outputs["true_labels"][span_start:span_end+1]).most_common(2)
            if _span_gold_label[0][0] != "O":
                span_gold_label = _span_gold_label[0][0]
            elif len(_span_gold_label) == 1 and _span_gold_label[0][0] == "O":
                # print("!!!!!! O embedding 加入了聚类中 !!!!!!!!!")
                span_gold_label = "B-O"
                clusters.append("O")
            else:
                span_gold_label = _span_gold_label[1][0]
            span_gold_labels.append(span_gold_label[2:])
        
    print("\nspan_embeddings = ",span_embeddings.shape)
    # print("span_gold_labels = ",span_gold_labels)

    clusters = list(set(clusters))
    # clusters = ["playlist","music_item","artist","entity_name","playlist_owner","O"] 
    vob_clusters_label2id = {}
    vob_clusters_id2label = {}
    for i, item in enumerate(clusters):
        vob_clusters_label2id[item] = i
        vob_clusters_id2label[i] = item
    print("vob_clusters_label2id = ",vob_clusters_label2id)
    gold_cluster_labels = np.array([vob_clusters_label2id[label] for label in span_gold_labels])
    # print("gold_cluster_labels = ",gold_cluster_labels)

    """
    将span embedding 输入给kmeans，判断类别；类别与原类别对齐，计算指标.
    """
    if "O" in clusters:
        n_clusters = len(clusters)-1
        clusters.remove("O")
    kmeans = KMeans(n_clusters=n_clusters).fit(span_embeddings) #init="k-means++"，, n_init=10, random_state=0
    # pred_cluster_labels = kmeans.fit_predict(span_embeddings)
    # pred_cluster_labels = kmeans.predict(span_embeddings)
    pred_cluster_labels = kmeans.labels_
    # print(pred_cluster_labels[:30])
    # print("kmeans_center = ", kmeans.cluster_centers_)
    cluster_to_true,pred_cluster_labels = hungray_aligment(y_true=gold_cluster_labels,y_pred=pred_cluster_labels) 
    kmeans_metrics(gold_cluster_labels,pred_cluster_labels) # gold_cluster_labels(n_samples),span_embeddings(n_samples, n_features)

    vob_clusters_id2label = {v:k for k,v in vob_clusters_label2id.items()}
    span_pred_labels = [vob_clusters_id2label[id] for id in pred_cluster_labels]
    print("混淆矩阵的标签 = ",clusters+["O"])
    cm = confusion_matrix(span_gold_labels,span_pred_labels,labels=clusters+["O"])
    report = classification_report(span_gold_labels,span_pred_labels,labels=clusters+["O"])
    print("混淆矩阵：\n",cm)
    print(report)

    print("————————--——————————————————————— （最终的spanf1指标） ——————————————————————————————————————")
    # span_pred_labels:list，聚类预测的span级别标签(已与真实标签文本对齐)
    # span_to_bio_tags
    import collections
    pred_span_deque = collections.deque(span_pred_labels) # 存放span队列
    final_pred_labels = []
    for pred in test_outputs["predict_labels"]:
        if pred == "O":
            final_pred_labels.append(pred)
        elif pred[0] == "B":
            item = pred_span_deque.popleft()
            if item=="O":
                final_pred_labels.append("O")
            else:
                final_pred_labels.append("B-"+item)
        elif pred[0] == "I":
            if item=="O":
                final_pred_labels.append("O")
            else:
                final_pred_labels.append("I-"+item)
    print("聚类后得到的BIOtags标注示例：",len(final_pred_labels),final_pred_labels[:10])
    # 调整vocab：原始vob不存在"I-X",但聚类预测中出现了"I-X"的情况
    vocab_labels = list(vocab.get_index_to_token_vocabulary("labels").values())
    pred_vocab_labels = list(set(final_pred_labels))
    for pred_label in pred_vocab_labels:
        if pred_label not in vocab_labels: vocab.add_token_to_namespace(pred_label,namespace = "labels") 
    pred_idx = [vocab.get_token_index(label, namespace="labels") for label in final_pred_labels]  
    pred_idx_tens = np.zeros((1,len(final_pred_labels), vocab.get_vocab_size("labels")),np.int)
    for token_ in range(pred_idx_tens.shape[1]):
        pred_idx_tens[0,token_,pred_idx[token_]] = 1  
    pred_idx_tens = torch.Tensor(pred_idx_tens)
    entity_true_idx_tens = torch.Tensor([[vocab.get_token_index(label, namespace="labels") for label in test_outputs["true_labels"]]])
    spanf1(pred_idx_tens,entity_true_idx_tens) # pred_idx_tens(1,n_samples,n_classes)，true_idx_tens(1,n_samples)
    metric = spanf1.get_metric(reset=True)
    slots = list(set(gold_spans_labels.tolist()))
    for slot in slots:
        f,r,p = round(metric["f1-"+slot]*100,2),round(metric["recall-"+slot]*100,2),round(metric["precision-"+slot]*100,2)
        print(f"{slot}对应的指标(span-metrics):  f:{f}, p:{p}, r:{r}")
    f,r,p = round(metric["f1-overall"]*100,2),round(metric["recall-overall"]*100,2),round(metric["precision-overall"]*100,2)
    print(f"整体对应的指标(span-metrics):  f:{f}, p:{p}, r:{r}")
    print("-"*90)
    print("————————--——————————————————————— （最终的tokenf1指标） ——————————————————————————————————————")
    # 把预测/真实序列中的I-全部换成B-
    true_BO = ["B"+label[1:] if label[0] == "I" else label for label in test_outputs["true_labels"]]
    pred_BO = ["B"+label[1:] if label[0] == "I" else label for label in final_pred_labels]
    # 计算SpanF1指标
    true_idx_tens = torch.Tensor([[vocab.get_token_index(label, namespace="labels") for label in true_BO]])
    pred_idx = [vocab.get_token_index(label, namespace="labels") for label in pred_BO]  
    pred_idx_tens = np.zeros((1,len(pred_BO), vocab.get_vocab_size("labels")),np.int)
    for token_ in range(pred_idx_tens.shape[1]):
        pred_idx_tens[0,token_,pred_idx[token_]] = 1  
    pred_idx_tens = torch.Tensor(pred_idx_tens)
    spanf1(pred_idx_tens,true_idx_tens) # pred_idx_tens(1,n_samples,n_classes)，true_idx_tens(1,n_samples)
    metric = spanf1.get_metric(reset=True)
    for slot in slots:
        f,r,p = round(metric["f1-"+slot]*100,2),round(metric["recall-"+slot]*100,2),round(metric["precision-"+slot]*100,2)
        print(f"- {slot}对应的指标(token-metrics):  f:{f}, p:{p}, r:{r}")
    f,r,p = round(metric["f1-overall"]*100,2),round(metric["recall-overall"]*100,2),round(metric["precision-overall"]*100,2)
    print(f"- 整体对应的指标(token-metrics):  f:{f}, p:{p}, r:{r}")
    print("-"*90)


    print("———————————————————————（span识别）计算目标域上每个槽位Span识别的指标:span_acc ——————————————————————————")
    target_slots = list(set(gold_spans_labels.tolist()))
    pred_spans_pd = pd.DataFrame(pred_spans,columns=["labels","span_idx_tuple"])
    pred_spans_idx = pred_spans_pd["span_idx_tuple"].tolist()
    tp = 0
    for slot in target_slots:
        slot_index = gold_spans_labels[gold_spans_labels.isin([slot])].index
        slot_target_idx = gold_spans_idxs[slot_index].tolist()
        num_slot_target = len(slot_target_idx) 
        slot_tp = len(set(slot_target_idx).intersection(set(pred_spans_idx)))
        slot_acc = round(100 * slot_tp/num_slot_target,2)
        tp += slot_tp
        print(f"- 目标域'{slot}'共{num_slot_target}个实体，识别完全准确的有{slot_tp}个，Span_ACC = {slot_tp}/{num_slot_target} = {slot_acc}")
    acc = round(100 * tp/len(gold_spans_idxs),2)
    print(f"- 目标域整体实际存在{len(gold_spans_idxs)}个实体，检测出了{len(pred_spans_idx)}个实体，其中识别完全准确的有{tp}个，Span_ACC = {tp}/{len(gold_spans_idxs)} = {acc}")
    print("-"*90)

    print("————————————————————————（span识别）计算目标域上每个槽位Span识别的指标:token_acc ——————————————————————————")
    gold_labels = ["O" if token=="O" else token[2:] for token in test_outputs["true_labels"]]
    gold_labels_detail = test_outputs["true_labels"]
    exam_slot = copy.deepcopy(target_slots) + ["O"]
    exam_slot_detail = list(set(test_outputs["true_labels"]))
    tp,num_gold,tp_detail,num_gold_detail = 0, 0,0,0
    num_pred_entitytoken = len(test_outputs["predict_labels"])-test_outputs["predict_labels"].count("O")
    for i,slot in enumerate(target_slots):
        pred_labels = ["O" if token=="O" else slot for token in test_outputs["predict_labels"]]
        exam_slot[0],exam_slot[i] = exam_slot[i],exam_slot[0]
        cm = confusion_matrix(gold_labels,pred_labels,labels = exam_slot)
        slot_tp = cm[0][0]
        num_slot_gold = int(np.sum(cm[0]))
        slot_acc = round(100 * slot_tp/num_slot_gold,2)
        tp += slot_tp
        num_gold += num_slot_gold
        # 
        pred_labels_detail = ["O" if token=="O" else token[:2]+slot for token in test_outputs["predict_labels"]]
        [exam_slot_detail.remove(i) if i in exam_slot_detail else i for i in ["B-"+slot,"I-"+slot]]
        exam_slot_detail =  ["B-"+slot,"I-"+slot] + exam_slot_detail
        cm_detail = confusion_matrix(gold_labels_detail,pred_labels_detail,labels = exam_slot_detail)
        slot_tp_detail = cm_detail[0][0]+cm_detail[1][1]
        num_slot_gold_detail = int(np.sum(cm_detail[0:2])) 
        slot_acc_detail = round(100 * slot_tp_detail/num_slot_gold_detail,2)
        tp_detail += slot_tp_detail
        num_gold_detail += num_slot_gold_detail

        print(f"- 目标域'{slot}'共{num_slot_gold}个实体token，其中识别准确的有{slot_tp}个(BI之间混淆不惩罚)，'B-{slot}'识别准确的有{cm_detail[0][0]}个(共有{int(np.sum(cm_detail[0]))}个), 'I-{slot}'识别准确的有{cm_detail[1][1]}个(共有{int(np.sum(cm_detail[1]))}个), Token_ACC = {slot_tp_detail}/{num_slot_gold_detail} = {slot_acc_detail}")
    acc = round(100 * tp/num_gold,2)
    acc_detail = round(100 * tp_detail/num_gold_detail,2)
    print(f"- 目标域整体实际存在{num_gold}个实体token，检测出了{num_pred_entitytoken}个实体token，其中识别准确的有{tp}个(BI之间混淆不惩罚)，Token_ACC = {tp_detail}/{num_gold_detail} = {acc_detail}")
    print("-"*90)


    print("————————--———————————————————————————— (Error Cases) —————————————--———————————————————————")
    # 把包含错误样本的句子GroundTruth和Predict相继打印到./xxx/error.csv
    tokens = load_tokens(data_dir = test_data_path)
    seq_lens = load_seq_len(data_dir = test_data_path) # list，数据集每个sentence的长度
    error_study(seq_lens,tokens,final_pred_labels,test_outputs["true_labels"],args.dataset)


    # target_slots = list(set(gold_spans_labels.tolist()))
    # true_idx_tens = torch.Tensor([[vocab.get_token_index(label, namespace="labels") for label in test_outputs["true_labels"]]])
    # slot_vocab = list(vocab.get_index_to_token_vocabulary("labels").values())
    # for slot in target_slots: # 对目标域每个槽位单独分析
    #     if "I-"+slot not in slot_vocab:
    #         vocab.add_token_to_namespace("I-"+slot,namespace = "labels")
    #     pred_idx = []
    #     for label in test_outputs["predict_labels"]:
    #         if label[:2]=="B-":
    #             pred_idx.append(vocab.get_token_index("B-"+slot, namespace="labels"))
    #         elif label[:2]=="I-":

    #             pred_idx.append(vocab.get_token_index("I-"+slot, namespace="labels"))
    #         else:
    #             pred_idx.append(vocab.get_token_index(label, namespace="labels"))
    #     pred_idx_tens = np.zeros((1,len(test_outputs["predict_labels"]), vocab.get_vocab_size("labels")),np.int)
    #     for token_ in range(pred_idx_tens.shape[1]):
    #         pred_idx_tens[0,token_,pred_idx[token_]] = 1  
    #     pred_idx_tens = torch.Tensor(pred_idx_tens)
    #     spanf1 = NSDSpanBasedF1Measure(
    #                                     vocabulary = vocab,
    #                                     tag_namespace="labels",
    #                                     ignore_classes=[],
    #                                     label_encoding="BIO",
    #                                     nsd_slots=["ns"]
    #                                     )
    #     spanf1(pred_idx_tens,true_idx_tens) # pred_idx_tens(1,n_samples,n_classes)，true_idx_tens(1,n_samples)
    #     metric = spanf1.get_metric(reset=True)
    #     f,r,p = round(metric["f1-"+slot]*100,2),round(metric["recall-"+slot]*100,2),round(metric["precision-"+slot]*100,2)
    #     print(f"在测试集上的{slot}指标(span-metrics):  f:{f}, p:{p}, r:{r}")
    # print("-"*82)



    exit()
    # GDA
    gda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage=None, store_covariance=True)
    gda.fit(np.array(train_outputs["encoder_outs"]), train_outputs["true_labels"])
    gda_means = gda.means_ 

    test_gda_result = confidence(np.array(test_outputs["encoder_outs"]), gda.means_, "euclidean", gda.covariance_)
    # test_y = pd.Series(gda.predict(np.array(test_outputs["encoder_outs"])))
    test_score = pd.Series(test_gda_result.min(axis=1))
    test_ns_idx = [idx_vo for idx_vo , _vo in enumerate(test_outputs["true_labels"]) if _vo in ns_labels]
    test_ind_idx = [idx_vi for idx_vi , _vi in enumerate(test_outputs["true_labels"]) if _vi not in ns_labels]
    test_ns_score = test_score[test_ns_idx]
    test_ind_score = test_score[test_ind_idx]

    # threshold
    if args.threshold == "auto":
        valid_outputs = predictor.predict_multi(file_path = os.path.join("data",args.dataset,"valid") ,batch_size = args.batch_size)
        valid_gda_result = confidence(np.array(valid_outputs["encoder_outs"]), gda.means_, "euclidean", gda.covariance_)
        valid_score = pd.Series(valid_gda_result.min(axis=1))
        valid_ns_idx = [idx_vo for idx_vo , _vo in enumerate(valid_outputs["true_labels"]) if _vo in ns_labels]
        valid_ind_idx = [idx_vi for idx_vi , _vi in enumerate(valid_outputs["true_labels"]) if _vi not in ns_labels]
        valid_ns_score = valid_score[valid_ns_idx]
        valid_ind_score = valid_score[valid_ind_idx]
        threshold,_ = estimate_best_threshold(np.array(valid_ind_score),np.array(valid_ns_score)) # the reasonable threshold
    elif args.threshold == "best":
        threshold,_ = estimate_best_threshold(np.array(test_ind_score),np.array(test_ns_score))    # the best threshold
    else:
        threshold = float(args.threshold)
    print("threshold = ",threshold)
    
    # override
    test_y_ns = pd.Series(test_outputs["predict_labels"])
    test_y_ns[test_score[test_score> threshold].index] = "ns"
    test_y_ns = list(test_y_ns)

    # Metrics
    start_idx = 0
    end_idx = 0
    test_pred_lines = []
    test_true_lines = []
    seq_lines = pd.DataFrame(test_outputs["tokens"])
    for i,seq in enumerate(seq_lines["tokens"]):
        start_idx = end_idx
        end_idx = start_idx + len(seq)
        adju_pred_line = parse_line(test_y_ns[start_idx:end_idx])
        test_true_line = test_outputs["true_labels"][start_idx:end_idx]
        test_pred_lines.append(adju_pred_line)
        test_true_lines.append(test_true_line)

    spanf1 = NSDSpanBasedF1Measure(tag_namespace="labels",
                        ignore_classes=[],
                        label_encoding="BIO",
                        nsd_slots=["ns"]
                        )
    spanf1(pd.Series(test_true_lines),pd.Series(test_pred_lines),False)
    metrics = spanf1.get_metric(reset=True)
    print("*"*70,"     Spanf1     ","*"*70)
    print("[result-ind-f] = ", metrics["f1-ind"])
    print("[result-ind-r] = ", metrics["recall-ind"])
    print("[result-ind-p] = ", metrics["precision-ind"])
    print("[result-nsd-f1] = ", metrics["f1-nsd"])
    print("[result-nsd-p] = ", metrics["recall-nsd"])
    print("[result-nsd-r] = ", metrics["precision-nsd"])
    
    print("*"*70,"     ROSE     ","*"*70)
    print("\np = 0.25")
    spanf1(pd.Series(test_true_lines),pd.Series(test_pred_lines),True,0.25)
    nsd_metrics = spanf1.get_metric(reset=True)
    print("[result-nsd-f1] = ", nsd_metrics["f1-nsd"])
    print("[result-nsd-p] = ", nsd_metrics["recall-nsd"])
    print("[result-nsd-r] = ", nsd_metrics["precision-nsd"])

    print("\np = 0.5")
    spanf1(pd.Series(test_true_lines),pd.Series(test_pred_lines),True,0.5)
    nsd_metrics = spanf1.get_metric(reset=True)
    print("[result-nsd-f1] = ", nsd_metrics["f1-nsd"])
    print("[result-nsd-p] = ", nsd_metrics["recall-nsd"])
    print("[result-nsd-r] = ", nsd_metrics["precision-nsd"])

    print("\np = 0.75")
    spanf1(pd.Series(test_true_lines),pd.Series(test_pred_lines),True,0.75)
    nsd_metrics = spanf1.get_metric(reset=True)
    print("[result-nsd-f1] = ", nsd_metrics["f1-nsd"])
    print("[result-nsd-p] = ", nsd_metrics["recall-nsd"])
    print("[result-nsd-r] = ", nsd_metrics["precision-nsd"])

    print("\np = 1.00")
    spanf1(pd.Series(test_true_lines),pd.Series(test_pred_lines),True,1)
    nsd_metrics = spanf1.get_metric(reset=True)
    print("[result-nsd-f1] = ", nsd_metrics["f1-nsd"])
    print("[result-nsd-p] = ", nsd_metrics["recall-nsd"])
    print("[result-nsd-r] = ", nsd_metrics["precision-nsd"])


    print("*"*70,"   token f1(mode1)   ","*"*70)
    test_y_tokens = parse_token(test_y_ns)
    test_true_tokens = parse_token(test_outputs["true_labels"])
    print("test_true_tokens = ",test_true_tokens[:10])
    print("test_y_tokens = ",test_y_tokens[:10])
    spanf1 = NSDSpanBasedF1Measure(tag_namespace="labels",
                        ignore_classes=[],
                        label_encoding="BIO",
                        nsd_slots=["ns"]
                        )

    spanf1(pd.Series([test_true_tokens]),pd.Series([test_y_tokens]),False)
    token_metrics = spanf1.get_metric(reset=True)
    print("[result-ind-f] = ", token_metrics["f1-ind"])
    print("[result-ind-r] = ", token_metrics["recall-ind"])
    print("[result-ind-p] = ", token_metrics["precision-ind"])
    print("[result-nsd-f1] = ", token_metrics["f1-nsd"])
    print("[result-nsd-p] = ", token_metrics["recall-nsd"])
    print("[result-nsd-r] = ", token_metrics["precision-nsd"])
    
    print("*"*70,"   token f1(mode2)   ","*"*70)
    labels = list(set(test_true_tokens))
    labels.remove("B-ns")
    labels.append("B-ns")
    get_score(test_true_tokens,test_y_tokens,labels)

    time_end = time()
    loop = time_end-time_begin
    print("time = ",round(loop/60,2))


    
