# 任务：
- 基于源域的有标注数据，自动获取目标域的slot集合并打标，最终目标是得到一个目标域的slot filling模型。
# 数据集：

- SNIPS，“AddToPlaylist”作为目标领域，其他6个类作为源域。# ["snips_AddToPlaylist","snips_SearchCreativeWork","snips_RateBook","snips_PlayMusic","snips_SearchScreeningEvent","snips_GetWeather","snips_BookRestaurant"]
- 目标领域包含五个槽位，其中三个槽位与域内槽位共享（playlist,music_item,artist），其他两个槽位独有(entity_name,playlist_owner)。

# 模型一 SpanEmbedding
## 训练阶段：
- 利用源域数据训练BIO（B-entity, I-entity, O）粗标签分类器；
## 测试阶段：
- 第一步：对目标域数据，利用先前的分类器，预测得到BIO粗标签；提取Span，得到 Span embedding（这里直接求的mean embedding）；
- 第二步：然后Kmeans对 Span embedding进行聚类；
## 结果：
- 详细实验结果：https://picturesque-clematis-c2b.notion.site/63ae517d380943b3b26028d6c95feede

- 第一步Span提取的指标：f:72.85, p:82.05, r:65.5
    ```python
         —————————计算目标域上每个槽位Span识别的指标:span_acc ————————
        目标域'entity_name'共33个实体，识别完全准确的有26个，Span_ACC = 26/33 = 78.79
        目标域'playlist'共124个实体，识别完全准确的有95个，Span_ACC = 95/124 = 76.61
        目标域'playlist_owner'共70个实体，识别完全准确的有0个，Span_ACC = 0/70 = 0.0
        目标域'music_item'共62个实体，识别完全准确的有53个，Span_ACC = 53/62 = 85.48
        目标域'artist'共53个实体，识别完全准确的有50个，Span_ACC = 50/53 = 94.34
        目标域整体实际存在342个实体，检测出了273个实体，其中识别完全准确的有224个，Span_ACC = 224/342 = 65.5
        ----------------------------------------------------------------------------------------------
         ————————— 计算目标域上每个槽位Span识别的指标:token_acc —————————
        目标域'entity_name'共87个实体token，其中识别准确的有82个(BI之间混淆不惩罚)，'B-entity_name'识别准确的有29个(共有33个), 'I-entity_name'识别准确的有49个(共有54个), Token_ACC = 78/87 = 89.66
        目标域'playlist'共348个实体token，其中识别准确的有327个(BI之间混淆不惩罚)，'B-playlist'识别准确的有103个(共有124个), 'I-playlist'识别准确的有198个(共有224个), Token_ACC = 301/348 = 86.49
        目标域'playlist_owner'共77个实体token，其中识别准确的有19个(BI之间混淆不惩罚)，'B-playlist_owner'识别准确的有12个(共有70个), 'I-playlist_owner'识别准确的有4个(共有7个), Token_ACC = 16/77 = 20.78
        目标域'music_item'共62个实体token，其中识别准确的有54个(BI之间混淆不惩罚)，'B-music_item'识别准确的有53个(共有62个), 'I-music_item'识别准确的有0个(共有0个), Token_ACC = 53/62 = 85.48
        目标域'artist'共110个实体token，其中识别准确的有110个(BI之间混淆不惩罚)，'B-artist'识别准确的有51个(共有53个), 'I-artist'识别准确的有57个(共有57个), Token_ACC = 108/110 = 98.18
        目标域整体实际存在684个实体token，检测出了609个实体token，其中识别准确的有592个(BI之间混淆不惩罚)，Token_ACC = 556/684 = 81.29
        ----------------------------------------------------------------------------------------------
    ```

- 第二步聚类的指标：

    ```python   
     __________________________________________________________________________________
      model            ACC         MI      RI         AMI     ARI     homo    comp    vmeas
      kmenas          0.641   0.910   0.812   0.556   0.475   0.630   0.521   0.570
    __________________________________________________________________________________ 
    ```
    对应的混淆矩阵(标签是对应的)：(pred,true)
    
    ```python
    混淆矩阵的标签 =  ['playlist', 'music_item', 'artist', 'entity_name', 'playlist_owner', 'O']
    混淆矩阵：
     [[50  0  1 20 44  5]
     [ 0 53  0  0  0  0]
     [ 1  0 43  0  3  5]
     [ 3  0  3 18  1  8]
     [ 1  0  1  2  4  2]
     [ 2  0  0  0  0  3]]
                    precision    recall  f1-score   support
    
          playlist       0.88      0.42      0.56       120
        music_item       1.00      1.00      1.00        53
            artist       0.90      0.83      0.86        52
       entity_name       0.45      0.55      0.49        33
    playlist_owner       0.08      0.40      0.13        10
                 O       0.13      0.60      0.21         5
    
          accuracy                           0.63       273
         macro avg       0.57      0.63      0.54       273
      weighted avg       0.81      0.63      0.67       273
    ```

### 指标注意：

- 由于第一步得到的Span与真实Span标签无法直接对齐（看下边的示例），这里对聚类的Ground cluster进行了调整——统计每个Span中出现次数最高频的实体标签作为Ground cluster，若无实体标签则Ground cluster=“O”
    - 无法对齐示例:
        - Ground Lables：['O', 'B-artist', 'I-artist', 'O', 'O', 'B-playlist', 'I-playlist', 'B-artist', 'O', 'O', "O"]
        - Predict labels： ['O', 'B-entity', 'I-entity', 'O', 'O', 'B-entity', 'I-entity', 'I-entity', 'O', 'B-entity',"I-entity"]
    - 对应标签调整：
        - Ground clusters：("artist",(1,2)), ("playlist",(5,7)),("O",(9,10))
        - Predict clusters：("xx",(1,2)), ("xx",(5,7)),("x",(9,10))


## 模型二 TokenEmbedding
### 训练阶段：
- 利用源域数据训练BO（B-entity, O）粗标签分类器；
### 测试阶段：
- 第一步：对目标域数据，利用先前的分类器，预测得到BO粗标签，提取实体Token embedding；【计算实体Token识别的指标】
- 第二步：然后Kmeans对Token embedding进行聚类（k = 5/6）【观察一下O数量】【计算token指标】；
- 第三步：将聚类得到的标注反馈回第一步BO，【计算token指标】，利用规则修改为BIO格式标注，【计算token指标，Span指标】；
 

### Error Study
- add the song don t drink the water to my playlist
- O O B-music_item B-playlist I-playlist I-playlist I-playlist I-playlist O B-playlist_owner O
- O O B-music_item B-entity_name B-playlist_owner I-playlist_owner I-playlist_owner I-playlist_owner O O O