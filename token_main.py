"""
这版训练BO分类器，不加CRF，利用预测得到的实体token embedding去做聚类，再返回，规则式打BIO标注.
## 模型二 TokenEmbedding
## 训练阶段：
- 利用源域数据训练BO（B-entity, O）粗标签分类器；
## 测试阶段：
- 第一步：对目标域数据，利用先前的分类器，预测得到BO粗标签，提取实体Token embedding；【计算实体Token识别的指标】
- 第二步：然后Kmeans对Token embedding进行聚类（k = 5/6）【观察一下O数量】；
- 第三步：将聚类得到的标注反馈回第一步BO，【计算token指标】，利用规则修改为BIO格式标注，【计算token指标，Span指标】；
"""
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
from sklearn.cluster import KMeans
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
    arg_parser.add_argument("--dataset",type=str,choices=["snips_AddToPlaylist_BO","snips_SearchCreativeWork_BO","snips_RateBook_BO","snips_PlayMusic_BO","snips_SearchScreeningEvent_BO","snips_GetWeather_BO","snips_BookRestaurant_BO"], default=None,
                        help="The dataset to use.")
    arg_parser.add_argument("--config_dir",type=str, default=None,
                        help="The config path.")
    arg_parser.add_argument("--output_dir",type=str, default="./output/snips_BO",
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

    # test_data_path = params.pop("test_data_path", None)
    # test_dataset_bio = reader.read(params.pop("test_data_path_bio", None))
    # if test_data_path:
        # test_dataset = reader.read(params.pop("test_data_path", None))
        # vocab = Vocabulary.from_instances(train_dataset + valid_dataset + test_dataset)
    # else:
        # print("test_dataset = None")
        # test_dataset = None
        # vocab = Vocabulary.from_instances(train_dataset + valid_dataset)
    
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
    # input_bi_label = "O B-entity B-entity B-entity O O B-entity B-entity O".split()
    # test_bbi_outputs = predictor.predict({"tokens": input_text.split(),"true_labels":input_bi_label}) # TODO 用于Debug的样例
    # test_outputs = predictor.predict({"tokens": input_text.split(),"true_labels":input_label}) # TODO 用于Debug的样例

    # 为计算后边指标，扩展vocab
    vocab_labels = list(vocab.get_index_to_token_vocabulary("labels").values())
    pred_vocab_labels = list(set(test_outputs["true_labels"]+test_outputs["predict_labels"]))
    for pred_label in pred_vocab_labels:
        if pred_label not in vocab_labels: vocab.add_token_to_namespace(pred_label,namespace = "labels") 

    print("————————————————————————第一步：（实体识别）计算在目标领域测试集上实体token的识别指标——————————————————————————")
    print("- 测试集示例: ")
    print("- Ground Truth： ",test_outputs["true_labels"][:10])
    print("- Predict Labels: ",test_bbi_outputs["predict_labels"][:10])
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
    print(f"\n- 在测试集上的eneity指标(token-metrics):  f:{f}, p:{p}, r:{r}")
    print("-"*90)
    # TODO 可以看一下BIO形式更好还是BO形式更好

    print("———————————————————————— 第二步：（聚类）计算聚类指标 ——————————————————————————")
    # 提取实体token embedding，以及对应的标注(为了对齐聚类指标以及对聚类效果进行评估)；
    true_labels_nobi = [_label if _label=="O" else _label[2:] for _label in test_outputs["true_labels"]]
    pred_labels_nobi = [_label if _label=="O" else _label[2:] for _label in test_outputs["predict_labels"]]
    true_labels_pd = pd.Series(true_labels_nobi)
    pred_labels_pd = pd.Series(pred_labels_nobi)
    pred_entity_idx = pred_labels_pd[pred_labels_pd.isin(["entity"])].index
    # print("pred_entity_idx = ",pred_entity_idx)
    # print("true_labels_pd = \n",true_labels_pd)
    # print("pred_labels_pd = \n",pred_labels_pd)
    cluster_true_labels = true_labels_pd[pred_entity_idx]
    cluster_true_labels = cluster_true_labels.tolist()
    token_embeddings = test_outputs["encoder_outs"][pred_entity_idx]
    # slots
    slots = list(set(true_labels_nobi))
    slots.remove("O")
    slots_o = list(set(true_labels_nobi))
    # 构建 slot_o 与 id 之间的映射关系
    vob_clusters_label2id = {}
    vob_clusters_id2label = {}
    for i, item in enumerate(slots_o):
        vob_clusters_label2id[item] = i
        vob_clusters_id2label[i] = item
    gold_cluster_labels = np.array([vob_clusters_label2id[label] for label in cluster_true_labels])
    # 将span embedding 输入给kmeans，判断类别；类别与原类别对齐，计算指标.
    cluster_contain_o = False
    clusters = slots_o
    if cluster_contain_o: 
        n_cluster = len(slots_o)
    else:
        n_cluster = len(slots)
        # clusters = slots
    kmeans = KMeans(init="k-means++", n_clusters=n_cluster, n_init=10, random_state=0) # TODO : 算"O"，则n_cluster = len(slots)+1; 否则 n_cluster = len(slots)+1
    cluster_pred_labels = kmeans.fit_predict(token_embeddings)
    cluster_to_true,cluster_pred_labels = hungray_aligment(y_true=gold_cluster_labels,y_pred=cluster_pred_labels) 
    kmeans_metrics(gold_cluster_labels,cluster_pred_labels) # gold_cluster_labels(n_samples),span_embeddings(n_samples, n_features)
    cluster_pred_labels = [vob_clusters_id2label[id] for id in cluster_pred_labels]
    print("- 混淆矩阵的标签 = ",clusters)
    cm = confusion_matrix(cluster_true_labels,cluster_pred_labels,labels=clusters)
    report = classification_report(cluster_true_labels,cluster_pred_labels,labels=clusters)
    print("- 混淆矩阵：\n",cm)
    print(report)

    print("————————--——————————————————————— 第三步：（规则聚类反馈BIO） ——————————————————————————————————————")
    # 先将聚类结果反馈回第一步预测的实体序列
    n = 0 
    for entity_idx in pred_entity_idx:
        pred_labels_pd[entity_idx] = cluster_pred_labels[n]
        n += 1
    # 再利用规则改成BIO形式 # TODO 还有一些潜规则之后再说
    seq_lens = load_seq_len(data_dir = test_data_path) # list，数据集每个sentence的长度
    final_pred_labels = []
    start_idx,end_idx = 0,0
    for seq_len in seq_lens: # 注意一下分句的情况(上句结尾为某tag，下句开头同为某tag)
        start_idx = end_idx # 第x条句子的起始索引
        end_idx = start_idx + seq_len # 第x条句子的终止索引
        seq_pred_labels = pred_labels_pd[start_idx:end_idx].tolist()
        seq_deque = collections.deque(seq_pred_labels)
        last_item = seq_deque.popleft()
        final_pred_labels.append("O" if last_item=="O" else "B-"+last_item)
        for i in range(len(seq_deque)):
            item = seq_deque.popleft()
            if item == last_item:
                final_pred_labels.append("O" if item=="O" else "I-"+item)
            else:
                final_pred_labels.append("O" if item=="O" else "B-"+item)
            last_item = item
    # print("- 最终预测示例：",len(final_pred_labels),final_pred_labels[:50])

    print("————————--——————————————————————— （最终的SpanF1指标） ——————————————————————————————————————")
    # 调整vocab：原始vob不存在"I-X",但聚类预测中出现了"I-X"的情况
    vocab_labels = list(vocab.get_index_to_token_vocabulary("labels").values())
    pred_vocab_labels = list(set(final_pred_labels))
    for pred_label in pred_vocab_labels:
        if pred_label not in vocab_labels: vocab.add_token_to_namespace(pred_label,namespace = "labels") 
    # 计算SpanF1指标
    true_idx_tens = torch.Tensor([[vocab.get_token_index(label, namespace="labels") for label in test_outputs["true_labels"]]])
    pred_idx = [vocab.get_token_index(label, namespace="labels") for label in final_pred_labels]  
    pred_idx_tens = np.zeros((1,len(final_pred_labels), vocab.get_vocab_size("labels")),np.int)
    for token_ in range(pred_idx_tens.shape[1]):
        pred_idx_tens[0,token_,pred_idx[token_]] = 1  
    pred_idx_tens = torch.Tensor(pred_idx_tens)
    spanf1(pred_idx_tens,true_idx_tens) # pred_idx_tens(1,n_samples,n_classes)，true_idx_tens(1,n_samples)
    metric = spanf1.get_metric(reset=True)
    for slot in slots:
        f,r,p = round(metric["f1-"+slot]*100,2),round(metric["recall-"+slot]*100,2),round(metric["precision-"+slot]*100,2)
        print(f"- {slot}对应的指标(span-metrics):  f:{f}, p:{p}, r:{r}")
    f,r,p = round(metric["f1-overall"]*100,2),round(metric["recall-overall"]*100,2),round(metric["precision-overall"]*100,2)
    print(f"- 整体对应的指标(span-metrics):  f:{f}, p:{p}, r:{r}")
    print("-"*90)

    print("————————--——————————————————————— （最终的TokenF1指标） ——————————————————————————————————————")
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

    print("————————--———————————————————————————— (Error Cases) —————————————--———————————————————————")
    # 把包含错误样本的句子GroundTruth和Predict相继打印到./xxx/error.csv
    tokens = load_tokens(data_dir = test_data_path)
    error_study(seq_lens,tokens,final_pred_labels,test_outputs["true_labels"],args.dataset)


    print("————————--———————————————————————————— (Plot PCA) —————————————--———————————————————————")
    clusters = ["playlist","music_item","artist","entity_name","playlist_owner","O"]  # TODO
    true_slot_o = pd.Series([label if label == "O" else label[2:] for label in test_outputs["true_labels"]])
    pred_slot_o = pd.Series([label if label == "O" else label[2:] for label in final_pred_labels])
    pca_2D_visualization(test_outputs["encoder_outs"],true_slot_o,clusters,os.path.join("./error_study",args.dataset+"_2D_true.png"))
    pca_3D_visualization(test_outputs["encoder_outs"],true_slot_o,clusters,os.path.join("./error_study",args.dataset+"_3D_true.png"))
    pca_2D_visualization(test_outputs["encoder_outs"],pred_slot_o,clusters,os.path.join("./error_study",args.dataset+"_2D_pred.png"))
    pca_3D_visualization(test_outputs["encoder_outs"],pred_slot_o,clusters,os.path.join("./error_study",args.dataset+"_3D_pred.png"))
    print("pred_slot_o.len = ",len(pred_slot_o))

    exit()

    print("————————————————————————计算目标域上每个槽位Span识别的指标:span_acc ——————————————————————————")
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

    print("————————————————————————计算目标域上每个槽位Span识别的指标:token_acc ——————————————————————————")
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


    
