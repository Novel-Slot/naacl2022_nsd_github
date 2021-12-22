# var=0
# while [ $var -eq 0 ]
# do
#     count=0
#     for i in $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
#     do
#         if [ $i -lt 1000 ]
#         then
#             echo 'GPU'$count' is avaiable'
#             python main.py  --dataset SnipsNSD15% --batch_size 200 --cuda 0 --mode both
#             python main.py  --dataset SnipsNSD30% --batch_size 200 --cuda 0 --mode both 
#             # 此处加上train 模型的语句
#             var=1
#             break
#         fi
#         count=$(($count+1))    
#     done    
# done


mode=$1  # 选项：0, 1, both, train, test
dataset=$2 # 选项：snips_AddToPlaylist, snips_SearchCreativeWork, snips_RateBook, snips_PlayMusic, snips_SearchScreeningEvent, snips_GetWeather, snips_BookRestaurant
config_dir="./config/$dataset.json"
if [ $mode = 0 ];then
    echo "*************************************************** 以snips_AddToPlaylist为例，进行测试 ********************************************************************"
    echo "python main.py  --dataset snips_AddToPlaylist --batch_size 256 --cuda 0 --mode test --threshold auto --config_dir ./config/snips_AddToPlaylist.json"
    python main.py  --dataset snips_AddToPlaylist --batch_size 256 --cuda 0 --mode test --threshold auto --config_dir ./config/snips_AddToPlaylist.json
elif [ $mode = 1 ];then
    echo "********************************************************* 遍历训练&测试 ************************************************************************************"
    for dataset in "snips_PlayMusic" "snips_SearchCreativeWork" "snips_RateBook" "snips_SearchScreeningEvent" "snips_GetWeather" "snips_BookRestaurant";do
        config_dir="./config/$dataset.json"
        mode="both"
        echo "python main.py  --dataset $dataset --batch_size 256 --cuda 0 --mode $mode --threshold auto --config_dir $config_dir"
        python main.py  --dataset $dataset --batch_size 256 --cuda 0 --mode $mode --threshold auto --config_dir $config_dir
    done
else
    echo "********************************************************* 单独训练or测试某数据集 **********************************************************************************"
    echo "python main.py  --dataset $dataset --batch_size 256 --cuda 0 --mode $mode --threshold auto --config_dir $config_dir"
    python main.py  --dataset $dataset --batch_size 256 --cuda 0 --mode $mode --threshold auto --config_dir $config_dir
fi


