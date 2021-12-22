import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import cluster
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix,classification_report
import numpy as np
import pandas as pd
import itertools
import matplotlib
from typing import List
import os
import json
import torch
import collections
import copy
import re
import time
import random as rn
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.metrics.cluster import mutual_info_score,rand_score,adjusted_mutual_info_score,adjusted_rand_score,homogeneity_score,completeness_score,v_measure_score,contingency_matrix
from sklearn.cluster import KMeans# abc
from sklearn import utils
from scipy.optimize import linear_sum_assignment

# from allennlp.data.vocabulary import Vocabulary

def setup_seed(SEED):
    np.random.seed(SEED)
    rn.seed(SEED)
    # tf.set_random_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.cuda.manual_seed(SEED)
    torch.manual_seed(SEED)

# def token_metrics():
def load_tokens(data_dir:str):
    """ 返回tokens: list,(n_sentences) """
    token_file = os.path.join(data_dir, "seq.in")
    with open(token_file, "r") as f:
        out_lines = f.readlines()
    tokens = [out_line.strip()
                    for out_line in out_lines if out_line.strip()]
    return tokens

def load_seq_len(data_dir:str):
    slot_label_file = os.path.join(data_dir, "seq.out")
    with open(slot_label_file, "r") as f:
        out_lines = f.readlines()
    seq_len = [len(out_line.strip().split(" "))
                    for out_line in out_lines if out_line.strip()]
    return seq_len

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import Axes3D
def pca_3D_visualization(X: np.ndarray,
                      y: pd.Series,
                      classes: List[str],
                      save_path: str):
    """
    Apply PCA visualization for features.
    """
    print("-----------------------")
    print(X.shape)
    red_features = PCA(n_components=3, svd_solver="full").fit_transform(X)
#     print(red_features[:2])

    plt.style.use("seaborn-darkgrid")
    fig = plt.figure(figsize=(10,10))
    ax = Axes3D(fig)
    plt.xlabel('X')
    plt.ylabel('Y', rotation=38)  # y 轴名称旋转 38 度
    ax.set_zlabel('Z')  # 因为 plt 不能设置 z 轴坐标轴名称，所以这里只能用 ax 轴来设置（当然，x 轴和 y 轴的坐标轴名称也可以用 ax 设置）

    for _class in classes:
        if _class == "O":
            ax.scatter(red_features[y == _class, 0], red_features[y == _class, 1], red_features[y == _class, 2],
                    label=_class, alpha=0.5, s=10, edgecolors='none', color="gray")
        else:
            ax.scatter(red_features[y == _class, 0], red_features[y == _class, 1], red_features[y == _class, 2],
                    label=_class, alpha=0.5, s=10, edgecolors='none', zorder=15)
    ax.legend(loc=2)
    ax.grid(True)
#     plt.axis('off')
#     plt.show()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, format="png")
#     plt.savefig(save_path, format="png")
def pca_2D_visualization(X: np.ndarray,
                      y: pd.Series,
                      classes: List[str],
                      save_path: str):
    """
    Apply PCA visualization for features.
    """
    print("-----------------------")
    print(X.shape)
    red_features = PCA(n_components=2, svd_solver="full").fit_transform(X)

    plt.style.use("seaborn-darkgrid")
#     fig,ax = plt.figure(figsize=(10,10))
    fig, ax = plt.subplots()
    for _class in classes:
        if _class == "O":
            ax.scatter(red_features[y == _class, 0], red_features[y == _class, 1], 
                    label=_class, alpha=0.5, s=10, edgecolors='none', color="gray")
        else:
            ax.scatter(red_features[y == _class, 0], red_features[y == _class, 1],
                    label=_class, alpha=0.5, s=10, edgecolors='none', zorder=15)
    ax.legend(loc=2)
    ax.grid(True)
#     plt.axis('off')
#     plt.show()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, format="png")
#     plt.savefig(save_path, format="png")

def TSNE_visualization(X: np.ndarray,
                      y: pd.Series,
                      classes: List[str],
                      save_path: str):
    X_embedded = TSNE(n_components=2).fit_transform(X)

    color_list=["red","green","blue","yellow","purple","black","brown","cyan","gray","pink","orange","blueviolet","greenyellow","sandybrown","deeppink"]

    plt.style.use("seaborn-darkgrid")
    fig, ax = plt.subplots()
    for _class in classes:
        if _class == "unseen":
            ax.scatter(X_embedded[y == _class, 0], X_embedded[y == _class, 1],
                       label=_class, alpha=0.5, s=20, edgecolors='none', color="gray")
        else:
            ax.scatter(X_embedded[y == _class, 0], X_embedded[y == _class, 1],
                       label=_class, alpha=0.5, s=4, edgecolors='none', zorder=15, color=color_list[_class])

    ax.grid(True)
    #plt.savefig(save_path, bbox_inches='tight', pad_inches=0, format="pdf")
    plt.savefig(save_path, format="png")



def error_study(seq_lens,tokens,pred_labels,true_labels,dataset):
    """
    @ In:
        - seq_len: array, (n_sentences)
        - tokens: list, (n_sentences)
        - pred_labels: list, (n_tokens)
        - true_labels: list, (n_tokens)
    @ Out:
        - 把包含错误样本的句子对应的Tokens & Ground & Predict 相继打印到./error_study/dataset_error.csv
    """
    error_token = []
    error_pred = []
    error_true = []
    true_token = []
    true_true = []
    start_idx,end_idx = 0,0
    for n,seq_len in enumerate(seq_lens):
        start_idx = end_idx # 第x条句子的起始索引
        end_idx = start_idx + seq_len # 第x条句子的终止索引
        seq_token = tokens[n]
        seq_pred = " ".join(pred_labels[start_idx:end_idx])
        seq_true = " ".join(true_labels[start_idx:end_idx])
        # print(f"seq_true = {seq_true}\nseq_pred = {seq_pred}")
        if seq_pred == seq_true:
            true_token.append(seq_token)
            true_true.append(seq_true)
        else:
            error_token.append(seq_token)
            error_true.append(seq_true)
            error_pred.append(seq_pred)
        with open(os.path.join("./error_study/",dataset+"_error.csv"), "w")as f_out:
            f_out.write("以下是预测存在错误的句子：\n")
            for num_out in range(len(error_pred)):     
                f_out.write(error_token[num_out] + "\n")
                f_out.write(error_true[num_out] + "\n")
                f_out.write(error_pred[num_out] + "\n")
            f_out.write("以下是预测完全正确的句子：\n")
            for num_out2 in range(len(true_true)):
                f_out.write(true_token[num_out2] + "\n")
                f_out.write(true_true[num_out2] + "\n")
    print(f"- 预测存在错误的句子有{len(error_pred)}条，预测完全正确的句子有{len(true_token)}条。")
    print(f"- 输入文本：{error_token[0]}\n- 正确示例：{error_true[0]}\n- 错误示例：{error_pred[0]}")
    print(f"- 输入文本：{error_token[1]}\n- 正确示例：{error_true[1]}\n- 错误示例：{error_pred[1]}")
    print("-"*90)

def kmeans_metrics(gold_cluster_labels,pred_cluster_labels):
    """ 计算聚类指标 """
    acc = accuracy_score(gold_cluster_labels,pred_cluster_labels)
    mi = mutual_info_score(gold_cluster_labels,pred_cluster_labels)
    ami = adjusted_mutual_info_score(gold_cluster_labels,pred_cluster_labels)
    ri = rand_score(gold_cluster_labels,pred_cluster_labels)
    ari = adjusted_rand_score(gold_cluster_labels,pred_cluster_labels)
    homo = homogeneity_score(gold_cluster_labels,pred_cluster_labels)
    comp = completeness_score(gold_cluster_labels,pred_cluster_labels)
    vmeas = v_measure_score(gold_cluster_labels,pred_cluster_labels)

    # contcm = contingency_matrix(gold_cluster_labels,pred_cluster_labels)
    print(82 * "_")
    print("model\t\tACC\tMI\tRI\tAMI\tARI\thomo\tcomp\tvmeas")
    results = ["kmenas",acc,mi,ri,ami,ari,homo,comp,vmeas]
    formatter_result = (
        "{:9s}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}"
    )
    print(formatter_result.format(*results))
    print(82 * "_")
    # print("\ncm = \n",contcm)


def load_data(data_dir:str):
    """
    In :    datadir = "./data/train/"
    Return: 三元数组列表[(BIOs, text, intent)]  eg.[('rate richard carvel','RateBook'), ['O', 'B-x', 'I-x'], (...)...]
    """
    text_in_file = os.path.join(data_dir, "seq.in")
    slot_label_file = os.path.join(data_dir, "seq.out")
    intent_label_file = os.path.join(data_dir, "label")

    with open(text_in_file, "r") as f:
        in_lines = f.readlines()
    with open(slot_label_file, "r") as f:
        out_lines = f.readlines()
    with open(intent_label_file, "r") as f:
        intent_lines = f.readlines()
    utterances = [in_line.strip() for in_line in in_lines if in_line.strip()]
    slot_labels = [out_line.strip().split(" ")
                    for out_line in out_lines if out_line.strip()]
    intent_labels = [intent_line.strip() for intent_line in intent_lines if intent_line.strip()]
    data_set = list(zip(utterances,slot_labels, intent_labels))
    return data_set

def kmeans_bench(kmeans, name, data, labels):
    """Benchmark to evaluate the KMeans initialization methods.

    Parameters
    ----------
    kmeans : KMeans instance
        A :class:`~sklearn.cluster.KMeans` instance with the initialization
        already set.
    name : str
        Name given to the strategy. It will be used to show the results in a
        table.
    data : ndarray of shape (n_samples, n_features)
        The data to cluster.
    labels : ndarray of shape (n_samples,)
        The labels used to compute the clustering metrics which requires some
        supervision.
    """
    t0 = time.time()
    estimator = make_pipeline(StandardScaler(), kmeans).fit(data)
    fit_time = time.time() - t0
    results = [name, fit_time, estimator[-1].inertia_]

    # Define the metrics which require only the true labels and estimator
    # labelsimport allennlp
    print("init\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI\tAMI\tsilhouette")
    clustering_metrics = [
        metrics.homogeneity_score,
        metrics.completeness_score,
        metrics.v_measure_score,
        metrics.adjusted_rand_score,
        metrics.adjusted_mutual_info_score,
    ]
    results += [m(labels, estimator[-1].labels_) for m in clustering_metrics]

    # The silhouette score requires the full dataset
    results += [
        metrics.silhouette_score(
            data,
            estimator[-1].labels_,
            metric="euclidean",
            sample_size=300,
        )
    ]

    # Show the results
    formatter_result = (
        "{:9s}\t{:.3f}s\t{:.0f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}"
    )
    print(formatter_result.format(*results))
    return results

def hungray_aligment(y_true, y_pred):
    """ 
    匈牙利算法对齐聚类标签和原GroundTruth标签 
    @Input:
        y_true(n_samples)
        y_pred(n_samples)
    @Return：
        cluster_to_true: dict, {cluster_pred_id, ground_true_id} 聚类标签到真实标签的映射字典
        y_pred_aligned: np.array, (n_samples) 与原GroundTruth标签对齐后的y_pred
    """
    D = max(y_pred.max(), y_true.max()) + 1 # 类别数量对齐
    w = np.zeros((D, D))
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1 
    map = np.transpose(np.asarray(linear_sum_assignment(w.max() - w))) # 匈牙利算法
    cluster_to_true = {i[0]:i[1] for i in map}
    y_pred_aligned = np.array([cluster_to_true[idx] for idx in y_pred]) # 映射
    return cluster_to_true,y_pred_aligned

def estimate_best_threshold(seen_m_dist: np.ndarray,
                            unseen_m_dist: np.ndarray) -> float:
    """
    ( multi classifier )
    Given mahalanobis distance for seen and unseen instances in valid set, estimate
    a best threshold (i.e. achieving best f1 in valid set) for test set.
    """
    lst = []
    for item in seen_m_dist:
        lst.append((item, "seen"))
    for item in unseen_m_dist:
        lst.append((item, "unseen"))
    # sort by m_dist: [(5.65, 'seen'), (8.33, 'seen'), ..., (854.3, 'unseen')]
    lst = sorted(lst, key=lambda item: item[0])
    threshold = 0.
    tp, fp, fn = len(unseen_m_dist), len(seen_m_dist), 0

    def compute_f1(tp, fp, fn):
        p = tp / (tp + fp + 1e-10)
        r = tp / (tp + fn + 1e-10)
        
        # logger.info("p= {p}".format(p = p))
        # logger.info("r= {r}".format(r = r))
        return (2 * p * r) / (p + r + 1e-10)

    f1 = compute_f1(tp, fp, fn)

    for m_dist, label in lst:
        if label == "seen":  # fp -> tn
            fp -= 1
        else:  # tp -> fn
            tp -= 1
            fn += 1
        if compute_f1(tp, fp, fn) > f1:
            f1 = compute_f1(tp, fp, fn)
            # logger.info("f1={f1} ".format(f1=f1))
            
            threshold = m_dist + 1e-10
    # print("estimated threshold:", threshold)
    return threshold,f1

if __name__ == "__main__":
    y_true = np.array([0,0,0,1,1,1,2,2,2])
    y_pred = np.array([2,2,1,2,1,1,3,3,3])
    cluster_to_true,y_pred_aligned = hungray_aligment(y_true=y_true,y_pred=y_pred)
    print("y_true: ",y_true)
    print("y_pred: ",y_pred)
    print("对齐后的聚类预测：",y_pred_aligned)
    

# from sklearn.cluster import KMeans
# from sklearn.decomposition import PCA
# from sklearn.metrics.cluster import adjusted_mutual_info_score
# span_embeddings = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
# gold_cluster_labels = np.array([0,0,0,1,1,1])

# print(82 * "_")
# print(time.time())
# kmeans_metrics(gold_cluster_labels,span_embeddings)

# kmeans = KMeans(init="k-means++", n_clusters=np.unique(gold_cluster_labels).size, n_init=4, random_state=0)
# kmeans_bench(kmeans=kmeans, name="k-means++", data=span_embeddings, labels=gold_cluster_labels) #data(n_samples, n_features), labels(n_samples)
# print(82 * "_")