import re
import numpy as np
from tqdm import tqdm
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

# 步驟 1：讀取 movie.txt
with open('../dataset/movie_new2.txt', 'r') as f:
    lines = f.readlines()

# 步驟 2：每筆觀看序列轉為電影 ID 的 list（例如：[910, 905, 2600, ...]）
def extract_sequence(line):
    pairs = re.findall(r'\((\d+),\s*(\d+)\)', line)
    sequence = [int(pairs[0][0])] + [int(p[1]) for p in pairs]
    return sequence

sequences = [extract_sequence(line) for line in tqdm(lines, desc="Extracting sequences...")]

# 步驟 3：轉為字串形式供 TF-IDF 使用（如："910 905 2600 1535 ..."）
corpus = [' '.join(map(str, seq)) for seq in tqdm(sequences, desc="Building corpus...")]

# 步驟 4：使用 TF-IDF 向量化（含 N-gram）
print("Building TF-IDF matrix...")
vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(2, 2))  # 使用 bi-gram
X_tfidf = vectorizer.fit_transform(corpus)  # X_tfidf 是稀疏矩陣

print("finished building TF-IDF matrix.")
print("TF-IDF matrix shape:", X_tfidf.shape)
sparse.save_npz("X_tfidf_sparse.npz", X_tfidf)
# X_tfidf = sparse.load_npz("X_tfidf_sparse.npz") # load t he X_tfidf matrix if needed

user_index = 0  # 第幾位使用者
row = X_tfidf.getrow(user_index)  # 取出一行稀疏向量
nonzero_indices = row.nonzero()[1]  # 取得非零欄位的索引
feature_names = vectorizer.get_feature_names_out()

# 顯示所有非零特徵及其 TF-IDF 值
for idx in nonzero_indices:
    print(f"{feature_names[idx]}: {row[0, idx]:.4f}")
