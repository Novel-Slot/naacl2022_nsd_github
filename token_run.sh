
mode=$1  # 选项：0  1  both  train  test
dataset=$2 # 选项：snips_AddToPlaylist_BO  snips_SearchCreativeWork  snips_RateBook  snips_PlayMusic  snips_SearchScreeningEvent  snips_GetWeather  snips_BookRestaurant
config_dir="./config/$dataset.json"
if [ $mode = 0 ];then
    echo "*************************************************** 以snips_AddToPlaylist_BO为例，进行测试 ********************************************************************"
    echo "python token_main.py  --dataset snips_AddToPlaylist_BO --batch_size 256 --cuda 0 --mode test --threshold auto --config_dir ./config/snips_AddToPlaylist_BO.json"
    python token_main.py  --dataset snips_AddToPlaylist_BO --batch_size 256 --cuda 0 --mode test --threshold auto --config_dir ./config/snips_AddToPlaylist_BO.json
elif [ $mode = 1 ];then
    echo "********************************************************* 遍历训练&测试 ************************************************************************************"
    for dataset in "snips_AddToPlaylist_BO" "snips_GetWeather_BO" "snips_SearchCreativeWork_BO" "snips_RateBook_BO" "snips_PlayMusic_BO" "snips_SearchScreeningEvent_BO" "snips_BookRestaurant_BO";do #"snips_AddToPlaylist_BO" 
        config_dir="./config/$dataset.json"
        mode="test"
        echo "python token_main.py  --dataset $dataset --batch_size 256 --cuda 0 --mode $mode --threshold auto --config_dir $config_dir"
        echo "## $dataset"
        python token_main.py  --dataset $dataset --batch_size 256 --cuda 0 --mode $mode --threshold auto --config_dir $config_dir
    done
else
    echo "********************************************************* 单独训练or测试某数据集 **********************************************************************************"
    echo "python token_main.py  --dataset $dataset --batch_size 256 --cuda 0 --mode $mode --threshold auto --config_dir $config_dir"
    python token_main.py  --dataset $dataset --batch_size 256 --cuda 0 --mode $mode --threshold auto --config_dir $config_dir
fi


