import os
import json

datasets = ["snips_SearchCreativeWork","snips_RateBook","snips_PlayMusic","snips_SearchScreeningEvent","snips_GetWeather","snips_BookRestaurant"]
ori_config_dir = "snips_AddToPlaylist.json"
for dataset in datasets:
    with open(os.path.join(ori_config_dir), "r", encoding="utf-8") as f_in:
        config_dic = json.load(f_in)
    print(type(config_dic))
    config_dic["train_data_path"] = os.path.join("data",dataset,"train")
    config_dic["valid_data_path"] = os.path.join("data",dataset,"valid")
    config_dic["test_data_path"] = os.path.join("data",dataset,"test")
    config_dic["test_data_path_bio"] = os.path.join("data",dataset,"test_bbi")

    with open(os.path.join( dataset+".json"), "w", encoding="utf-8") as f_out:
        json.dump(config_dic,f_out)

