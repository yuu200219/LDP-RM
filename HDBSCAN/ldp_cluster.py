import sys
import os
import re
from tqdm import tqdm

sys.path.append("../LDP-RM")
from ldp_rm import LDP_RM
from data_rm import Data
from svsm_rm import SVSM
from metrics import Metrics

import ast


def analyze_cluster_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    limit = len(lines)
    domain_ids = set()

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if "(" in line:  # 格式是 like: (1, 2) # (3, 4)
            parts = line.split("#")
            for item in parts:
                try:
                    pair = ast.literal_eval(item.strip())
                    domain_ids.update(pair)
                except:
                    continue
        else:  # 格式是 like: 1 2 3 4 5
            ids = map(int, line.strip().split())
            domain_ids.update(ids)

    domain_size = max(domain_ids) + 1  # 假設 ID 從 0 開始

    return limit, domain_size, limit  # user_total = limit


def extract_sequence(line):
    pairs = re.findall(r"\((\d+),\s*(\d+)\)", line)
    sequence = [int(pairs[0][0])] + [int(p[1]) for p in pairs]
    return sequence


# ======================
# Main
# ======================

cluster_folder = "../dataset/cluster_movie/"
cluster_files = [
    f for f in os.listdir(cluster_folder) if re.match(r"cluster_\d+\.txt$", f)
]

# ===== 可調參數 =====
PRESET_TOP_K = 64  # 預設全域 top_k，可依需求自訂
TOP_KS = 1600  # 預設 top_ks
EPSILON = 4.0  # 預設 epsilon
SUBMAT = 2  # 預設 submat
# ====================

# filename = 'cluster_27.txt'
max_ncr = 0
max_f1 = 0
min_var = 1
ncr_sum = 0
f1_sum = 0
var_sum = 0
ct_sum = 0
total = 0
for filename in sorted(cluster_files):
    total += 1
    print(filename)
    cluster_path = os.path.join(cluster_folder, filename)
    cluster_id = filename.replace(".txt", "")

    # analyze cluster and it corresponding parameter for Data
    limit, domain_size, user_total = analyze_cluster_txt(cluster_path)

    print(limit, domain_size, user_total)
    # get the movie list in each cluster file
    with open(cluster_path, "r") as f:
        lines = f.readlines()
    sequences = [
        extract_sequence(line) for line in tqdm(lines, desc="Extracting sequences...")
    ]
    movie_ids = sorted(set(movie_id for seq in sequences for movie_id in seq))
    if len(movie_ids) < 15:
        continue
    # 根據每個 cluster 動態決定 top_k
    top_k = min(PRESET_TOP_K, int(len(movie_ids) * 0.5))

    # 決定 top_ks
    top_ks = min(TOP_KS, int(top_k * (top_k - 1) / 2))  # 不超過可組合的 pair 數

    # 決定 top_kc
    top_kc = min(32, top_ks / 2)

    # Build Data, Metrics, LDP_RM parameters
    data = Data(
        dataname=cluster_id,
        limit=limit,
        domain_size=domain_size,
        user_total=user_total,
    )  # Movie dataset
    # metrics = Metrics(data, top_k=64, top_ks=1600, top_kc=32)
    metrics = Metrics(data, top_k=top_k, top_ks=top_ks, top_kc=top_kc)
    ldp_rm = LDP_RM(
        data, epsilon=EPSILON, top_k=top_k, top_ks=top_ks, top_kc=top_kc, submat=SUBMAT
    )
    import time

    # 10 rounds
    for t in range(10):
        t1 = time.time()
        result_fre_dict_svd, result_conf_dict, hitrate_rm = ldp_rm.find_itemset_svd(
            task="RM",
            method="AMN",
            singnum=0.5,
            use_group=True,
            group_num=5,
            test="test_constant",
        )
        t2 = time.time()
        consume_time = int(t2 - t1)
        print("Final mining topks relations:", result_conf_dict)
        print("ldp_rm NCR", ncr := metrics.NCR(result_conf_dict))
        print("ldp_rm F1", f1 := metrics.F1(result_conf_dict))
        print("ldp_rm VAR", var := metrics.VARt(result_conf_dict))
        print("time:", ct := consume_time)
        ncr_sum += ncr
        f1_sum += f1
        var_sum += var
        ct_sum += ct
        max_ncr = max(max_ncr, ncr)
        max_f1 = max(max_f1, f1)
print("average NCR:", round(ncr_sum / total, 4))
print("average F1:", round(f1_sum / total, 4))
print("average consume time:", round(ct_sum / total, 4))
print("max NCR:", max_ncr)
print("max F1: ", max_f1)
