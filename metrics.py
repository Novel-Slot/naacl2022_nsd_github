from allennlp.data.vocabulary import Vocabulary
from allennlp.training.metrics.span_based_f1_measure import SpanBasedF1Measure,TAGS_TO_SPANS_FUNCTION_TYPE
from allennlp.data.dataset_readers.dataset_utils.span_utils import bio_tags_to_spans

from overrides import overrides
from typing import Dict, List, Optional, Set, Callable
from collections import defaultdict,Counter
import numpy as np
import pandas as pd
import copy
import math

class NSDSpanBasedF1Measure(SpanBasedF1Measure):
    """
    SpanBasedF1Measure
    - IND, for f1-ind|precision-ind|recall-ind
    - NSD, for f1-nsd|precision-nsd|recall-nsd
    Attention:
    - 调用时的维度，pred_idx_tens(1,n_samples,n_classes)，true_idx_tens(1,n_samples)
    """
    def __init__(self,
                 vocabulary: Vocabulary,
                 tag_namespace: str = "tags",
                 ignore_classes: List[str] = None,
                 label_encoding: Optional[str] = "BIO",
                 tags_to_spans_function: Optional[TAGS_TO_SPANS_FUNCTION_TYPE] = None,
                 nsd_slots: List[str] = None) -> None:
        super(NSDSpanBasedF1Measure, self).__init__(vocabulary=vocabulary,
                                                       tag_namespace=tag_namespace,
                                                       ignore_classes=ignore_classes,
                                                       label_encoding=label_encoding,
                                                       tags_to_spans_function=tags_to_spans_function)
        self._ignore_classes: List[str] = ignore_classes or []
        self._nsd_slots = nsd_slots or ["ns"]


    @overrides
    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        A Dict per label containing following the span based metrics:
        precision : float
        recall : float
        f1-measure : float

        Additionally, an ``overall`` key is included, which provides the precision,
        recall and f1-measure for all spans.

        (*) Additionally, for novel slots and normal (not novel)
        slots, an ``ns`` key and a ``normal`` key are included respectively, which
        provide the precision, recall and f1-measure for all novel spans
        and all normal spans, respectively.
        """
        all_tags: Set[str] = set()
        all_tags.update(self._true_positives.keys())
        all_tags.update(self._false_positives.keys())
        all_tags.update(self._false_negatives.keys())
        all_metrics = {}
        for tag in all_tags:
            precision, recall, f1_measure = self._compute_metrics(self._true_positives[tag],
                                            self._false_positives[tag],
                                            self._false_negatives[tag])
            precision_key = "precision" + "-" + tag
            recall_key = "recall" + "-" + tag
            f1_key = "f1" + "-" + tag
            all_metrics[precision_key] = precision
            all_metrics[recall_key] = recall
            all_metrics[f1_key] = f1_measure

        # Compute the precision, recall and f1 for all spans jointly.
        precision, recall, f1_measure = self._compute_metrics(sum(self._true_positives.values()),
                                                              sum(self._false_positives.values()),
                                                              sum(self._false_negatives.values()))
        all_metrics["precision-overall"] = precision
        all_metrics["recall-overall"] = recall
        all_metrics["f1-overall"] = f1_measure
        # all_metrics["acc-overall"] = sum(self._true_positives.values()/sum()
        


        # # (*) Compute the precision, recall and f1 for all nsd spans jointly.
        # precision, recall, f1_measure = self._compute_metrics(
        #     sum(map(lambda x: x[1], filter(lambda x: x[0] in self._nsd_slots, self._true_positives.items()))),
        #     sum(map(lambda x: x[1], filter(lambda x: x[0] in self._nsd_slots, self._false_positives.items()))),
        #     sum(map(lambda x: x[1], filter(lambda x: x[0] in self._nsd_slots, self._false_negatives.items()))))
        # all_metrics["precision-nsd"] = precision
        # all_metrics["recall-nsd"] = recall
        # all_metrics["f1-nsd"] = f1_measure

        # (*) Compute the precision, recall and f1 for all ind spans jointly.
        precision, recall, f1_measure = self._compute_metrics(
            sum(map(lambda x: x[1], filter(lambda x: x[0] not in self._nsd_slots, self._true_positives.items()))),
            sum(map(lambda x: x[1], filter(lambda x: x[0] not in self._nsd_slots, self._false_positives.items()))),
            sum(map(lambda x: x[1], filter(lambda x: x[0] not in self._nsd_slots, self._false_negatives.items()))))
        all_metrics["precision-ind"] = precision
        all_metrics["recall-ind"] = recall
        all_metrics["f1-ind"] = f1_measure

        if reset:
            self.reset()
        return all_metrics



if __name__ == "__main__":

    gold_label  = [["B-timeRange","I-timeRange","I-timeRange","B-ns","I-ns","I-ns","I-ns","I-ns","I-ns","B-timeRange","I-timeRange","I-timeRange","O"],["B-timeRange","B-ns","I-ns", "I-ns"  ,"I-ns",  "O", "B-artist","B-ns","B-poi"],["B-ns","I-ns", "B-ns"],["O","B-ns","O"]]
    predict_label = [["B-timeRange","I-timeRange","ns",   "ns","O", "ns", "O","ns","ns",    "ns","B-timeRange","I-timeRange","I-timeRange"],["ns","ns","B-timeRange","O","O","O","B-artist","I-artist","B-poi"],["ns","ns", "B-timeRange"],["O","ns","ns"]]
    

    def parse_line(line:list):
        modify_line = []
        for i,label in enumerate(line):
            if label in ["ns","B-ns","I-ns"]:
                if i == 0:
                    modify_line.append("B-ns")
                elif modify_line[i-1] in ["ns","B-ns","I-ns"]:
                    modify_line.append("I-ns")
                else:
                    modify_line.append("B-ns")
            elif label == "O":
                modify_line.append("O")
            else:
                if i == 0:
                    modify_line.append(label)
                elif modify_line[i-1][-3:] == "-ns" and label[:2]=="I-":
                    modify_line.append("B-"+label[-2:])
                else:
                    modify_line.append(label)
        print("[before]",line)
        print("[after]",modify_line)
        return modify_line
    gold_labels=[]
    predict_labels = []
    for i,line in enumerate(gold_label):
        gold_labels.append(parse_line(line))
    for i,line in enumerate(predict_label):
        predict_labels.append(parse_line(line))

    # print("gold  = ",gold_label)
    print("gold = ",gold_labels)
    # print("pred = ",predict_label)
    print("pred = ",gold_labels)
    gold_labels = pd.Series(gold_labels)
    predict_labels = pd.Series(predict_labels)
    print("*"*70)
    spanf1 = NSDSpanBasedF1Measure(tag_namespace="labels",
                        ignore_classes=[],
                        label_encoding="BIO",
                        nsd_slots=["ns"]
                        )

    spanf1(gold_labels,predict_labels,False)
    metrics = spanf1.get_metric(reset = True)
    print("*"*70)
    # print("IND")
    print("[result-ind-f] = ", metrics["f1-ind"])
    print("[result-ind-r] = ", metrics["recall-ind"])
    print("[result-ind-p] = ", metrics["precision-ind"])
    print("[result-nsd-f1] = ", metrics["f1-nsd"])
    print("[result-nsd-p] = ", metrics["recall-nsd"])
    print("[result-nsd-r] = ", metrics["precision-nsd"])
    
    print("*"*70)
    spanf1(gold_labels,predict_labels,True,0.25)
    nsd_metrics = spanf1.get_metric(reset = True)
    print("p=0.25")
    print("[result-nsd-f1] = ", nsd_metrics["f1-nsd"])
    print("[result-nsd-p] = ", nsd_metrics["recall-nsd"])
    print("[result-nsd-r] = ", nsd_metrics["precision-nsd"])
    print("*"*70)
    spanf1(gold_labels,predict_labels,True,0.5)
    nsd_metrics = spanf1.get_metric(reset = True)
    print("p=0.5")
    print("[result-nsd-f1] = ", nsd_metrics["f1-nsd"])
    print("[result-nsd-p] = ", nsd_metrics["recall-nsd"])
    print("[result-nsd-r] = ", nsd_metrics["precision-nsd"])
    print("*"*70)
    spanf1(gold_labels,predict_labels,True,0.75)
    nsd_metrics = spanf1.get_metric(reset = True)
    print("p=0.75")
    print("[result-nsd-f1] = ", nsd_metrics["f1-nsd"])
    print("[result-nsd-p] = ", nsd_metrics["recall-nsd"])
    print("[result-nsd-r] = ", nsd_metrics["precision-nsd"])
    print("*"*70)
    spanf1(gold_labels,predict_labels,True,1)
    nsd_metrics = spanf1.get_metric(reset = True)
    print("p=0.1")
    print("[result-nsd-f1] = ", nsd_metrics["f1-nsd"])
    print("[result-nsd-p] = ", nsd_metrics["recall-nsd"])
    print("[result-nsd-r] = ", nsd_metrics["precision-nsd"])


    print("*"*70,"   token f1(mode1)   ","*"*70)
    def parse_token(line:List):
        modify_line = []
        for label in line:
            if label in ["ns","B-ns","I-ns"]:
                modify_line.append("B-ns")
            elif label == "O":
                modify_line.append("O")
            else:
                if i == 0:
                    modify_line.append(label)
                elif line[i-1][-2:] == "ns" and label[:2]=="I-":
                    modify_line.append("B-"+label[-2:])
                else:
                    modify_line.append(label)
        return modify_line
    test_y_tokens = []
    test_true_tokens = []
    for i,line in enumerate(gold_label):
        test_y_tokens.append(parse_token(line))
    for i,line in enumerate(predict_label):
        test_true_tokens.append(parse_token(line))
    print("test_true_tokens = ",test_true_tokens[:10])
    print("test_y_tokens = ",test_y_tokens[:10])
    
    spanf1 = NSDSpanBasedF1Measure(tag_namespace="labels",
                        ignore_classes=[],
                        label_encoding="BIO",
                        nsd_slots=["ns"]
                        )

    spanf1(pd.Series(test_true_tokens),pd.Series(test_y_tokens),False)
    token_metrics = spanf1.get_metric(reset=True)
    print("[result-ind-f] = ", token_metrics["f1-ind"])
    print("[result-ind-r] = ", token_metrics["recall-ind"])
    print("[result-ind-p] = ", token_metrics["precision-ind"])
    print("[result-nsd-f1] = ", token_metrics["f1-nsd"])
    print("[result-nsd-p] = ", token_metrics["recall-nsd"])
    print("[result-nsd-r] = ", token_metrics["precision-nsd"])