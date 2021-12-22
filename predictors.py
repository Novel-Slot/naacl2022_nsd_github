from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors import Predictor
from overrides import overrides

import numpy as np
import pandas as pd
from time import *
import os

@Predictor.register("slot_filling_predictor")
class SlotFillingPredictor(Predictor):

    def predict(self, inputs: JsonDict) -> JsonDict:
        """ 为了Debug，少样本测试 """
        instance = self._json_to_instance(inputs)
        output = self.predict_instance(instance)

        # mask = np.array(output["mask"].values.tolist())
        # masks = np.ma.masked_where(mask==1,mask)
        # encoder_out = np.array(output["encoder_out"].values.tolist())
        # encoder_out = encoder_out[masks.mask,:] # (batch_size,seq_dim,256) -> (token_nums,256)
        outputs = {
            "tokens": inputs["tokens"],
            "predict_labels": [self._model.vocab.get_token_from_index(index, namespace="labels")
                               for index in output["predicted_tags"]],
            "encoder_outs":np.array(output["encoder_out"]),
        }
        if "true_labels" in inputs:
            outputs["true_labels"] = inputs["true_labels"]
        return outputs

    def predict_multi(self, file_path: str, batch_size = 64):
        """ 读取数据集文件进行大规模测试 """
        tokens,labels = self.load_line(file_path)
        predict_labels = []
        for batch in range(0,len(tokens),batch_size):
            instance = self._batch_json_to_instances(tokens[batch:batch+batch_size])
            output = self.predict_batch_instance(instance) #List[JsonDict]
            output = pd.DataFrame(output)
    
            encoder_out = np.array(output["encoder_out"].values.tolist())
            mask = np.array(output["mask"].values.tolist())
            masks = np.ma.masked_where(mask==1,mask)
            encoder_out = encoder_out[masks.mask,:] # (batch_size,seq_dim,256) -> (token_nums,256)
            encoder_outs = np.concatenate((encoder_outs,encoder_out),axis=0) if batch > 0 else encoder_out 
            predicted_tag = np.array(sum(output["predicted_tags"].values.tolist(),[]))
            predict_labels = predict_labels + [self._model.vocab.get_token_from_index(index, namespace="labels")
                                    for index in predicted_tag]
        true_labels = sum(pd.DataFrame(labels)["true_labels"].values.tolist(),[])
        results = {
            "predict_labels":predict_labels,    # List, (token_nums)
            "encoder_outs":encoder_outs,    # Array, (token_nums,256)
            "tokens":tokens,    # List[JsonDict], (seq_dim,token_dim)
            "true_labels": true_labels # List (token_num)
        }
        print(f"output: predict_labels = {len(predict_labels)}, true_labels = {len(true_labels)}, encoder_outs = {encoder_out.shape}, tokens = {len(tokens)}")
        return results
    @overrides
    def load_line(self, file_path: str ) -> JsonDict:  # pylint: disable=no-self-use
        token_list_dict = []
        label_list_dict = []
        token_file_path = os.path.join(file_path, "seq.in")
        label_file_path = os.path.join(file_path, "seq.out")
        with open(token_file_path, "r", encoding="utf-8") as f_token:
            token_lines = f_token.readlines()
        with open(label_file_path, "r", encoding="utf-8") as f_label:
            label_lines = f_label.readlines()
        assert len(token_lines) == len(label_lines)
        for i in range(len(token_lines)):
            tokens = token_lines[i].strip().split(" ")
            labels = label_lines[i].strip().split(" ")
            token_list_dict.append({"tokens":[tokeni for tokeni in tokens if tokeni != ""]})
            label_list_dict.append({"true_labels":[labeli for labeli in labels if labeli != ""]})

        return token_list_dict,label_list_dict

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        tokens = json_dict["tokens"]
        instance = self._dataset_reader.text_to_instance(tokens=tokens)
        return instance

