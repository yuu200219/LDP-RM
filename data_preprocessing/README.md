# Final Project- Relation Mining Under Local Differential Privacy   
[https://github.com/yuu200219/LDP-RM/tree/main](https://github.com/yuu200219/LDP-RM/tree/main)    
## Original Results   
### Movies (10 iterations)   
```
average NCR: 0.5756
average F1: 0.5563
average consume time: 956.8


```
## How to improve   
- 針對 2-item relation movie dataset   
    - **分群！**   
   
## 方法：分群 (clustering using HDBSCAN)   
### Step1: 轉換為向量   
```
[[1, 2, 3, ...],
[4, 5, 6, ...],
...
]


```
### Step2: 將序列向量化   
**N-gram + TF-IDF 向量化**   
- **2-gram (建議)**   
    ex: [1, 2], [3, 4]   
- 3-gram   
    ex: [1, 2, 3], [4, 5, 6]   
- TF-IDF   
    - TF: term frequency
某個詞語在一份文件中出現的次數
   
    
$$
TF_{t,d} = \frac{\text{t in document d count }}{\text{total document d count}}
$$
    - IDF: inverse document frequency
用來衡量一個詞語是否為「稀有」或「常見」。如果一個詞在越多文件都出現過，它的區辨力就越低
   
    
$$
IDF_{t} = log\frac{\text{total document count}}{\text{t in document count}}
$$
   
### Step 3: 將這個向量降為，並輸入到 HDBSCAN   
### Step 4: 根據 HDBSCAN 結果去切割原始 dataset   
### Step 5: 將切割的 dataset 分別跑 LDP-RM   
- 目前分數不佳，嘗試了以下的 modification   
   
```
vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(2, 2))
vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(2, 2), min_df=5)  # 使用 bi-gram, 忽略少於 5 個 bi-gram的序列
vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(2, 3), min_df=5)
vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=5)
```
- cluster\_labels\_min\_df\_5
max NCR: 0.3314
max F1:  0.25   
- cluster\_label\_min\_1\_2\_min\_df\_5
max NCR: 0
max F1:  0   
- cluster\_label\_min\_2\_3\_min\_df\_5
max NCR: 0.3068
max F1:  0.25   
- cluster\_label\_min\_df\_5 (ngram\_range = (2, 2))   
    - cluster 24
12966 5001 12966
epsilon: 6
ldp\_rm NCR 0.4299
ldp\_rm F1 0.3438   
    - cluster\_36
131619 5001 131619
epsilon: 6
ldp\_rm NCR 0.6364
ldp\_rm F1 0.5938   
   
    max NCR: 0.6364
max F1:  0.5938   
       
- **cluster\_label\_min\_df\_5, 加入 cosine similarity**
average NCR: 0.043
average F1: 0.0385
average consume time: 9.0385
max NCR: 0.6345
max F1:  0.625

average NCR: 0.0512
average F1: 0.0382
average consume time: 10.2778
max NCR: 0.4394
max F1:  0.4062
   
   
## 改善方向：   
- [x] 合併相似的 cluster（針對資料量太少的 cluster）   
  - 使用你已經建好的 TF-IDF `X\_tfidf` 矩陣（或 cluster centroid 向量）   
  - 對所有 cluster 計算彼此的 **cosine similarity**   
  - 找出相似度高且**樣本數都偏少**的 cluster pair   
  - 合併資料、重新建構 movie\_list 與 sequence   
   
  **Implementation example:**   
  ```
from sklearn.metrics.pairwise import cosine_similarity

# 計算每個 cluster 的 TF-IDF 中心向量
cluster_centroids = []
for c in cluster_to_user_sequences:
    indices = [i for i, seq in enumerate(sequences) if any(mid in movie_id_to_cluster and movie_id_to_cluster[mid] == c for mid in seq)]
    centroid = X_tfidf[indices].mean(axis=0)
    cluster_centroids.append(centroid)

# 計算 cluster 之間的 cosine similarity
sim_matrix = cosine_similarity(cluster_centroids)

# 合併相似度高（如 > 0.8） 且資料量都 < threshold 的 cluster pair

```
- [ ] 保留有序資訊，挖掘有順序的 itemset（sequence-based）   
  - 感覺比較難做，先不做   
- [x] 將所有 cluster 的 top\_64 資料合併並丟到 metrics 做計算，分數應該會更高? （最終步驟）   
# final result   
- cluster\_label\_min\_df\_5,  ngram\_range=(2,2)
   
   
```
max NCR: 0.7576
file:  cluster_36.txt
max F1:  0.7188
file:  cluster_36.txt

```
- 相較於原始資料集，分數更高   
   
```
average NCR: 0.5756
average F1: 0.5563


```
- 原本想要去合併所有 cluster 結果，取 top 32，但是分數不理想
**原因：**
在這種 user sequence、movie id 非均勻分布的大型資料下，在 cluster 後的結果跟原始資料並沒有重疊，所以分數分長低，也就是：
   
    1. **群內偏好、分布高度異質**   
        - 某 cluster 的 users 可能只偏好科幻片（他們的 top-64 規則都與 sci-fi 有關），但全體 top-64 規則可能被動作片佔據。   
        - 或 cluster 是一個冷門電影小圈圈，全域根本不常見。   
    2. **資料太 sparse，群內 top 32 頻率都不高**   
        - 群內資料小，擾動比例高，容易挖出“只在該群有”的規則，這些規則在全體根本不成氣候。   
    3. **合併 cluster 頂多能補齊一些，但只要 cluster segmentation 跨領域（如 genre、地域），全域和局部規則永遠很難重疊**   
   
   
