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
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    limit = len(lines)
    domain_ids = set()

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if '(' in line:  # 格式是 like: (1, 2) # (3, 4)
            parts = line.split('#')
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


cluster_folder = '../dataset/cluster_movie/'
cluster_files = [f for f in os.listdir(cluster_folder) if re.match(r'cluster_\d+\.txt$', f)]
filename = 'cluster_576.txt'
# for filename in sorted(cluster_files):
print(cluster_files)
cluster_path = os.path.join(cluster_folder, filename)
cluster_id = filename.replace('.txt', '')
limit, domain_size, user_total = analyze_cluster_txt(cluster_path)
    # print(limit, domain_size, user_total)
data = Data(dataname='cluster_1657', limit=limit, domain_size=domain_size, user_total=user_total) # Movie dataset
    # metrics = Metrics(data, top_k=64, top_ks=1600, top_kc=32)
try:
    metrics = Metrics(data, epsilon=4.0, top_k=8, top_ks=400, top_kc=8, submat=2)
except Exception as e:
    print(f"[!] Metrics failed: {e}")
ldp_rm = LDP_RM(data, epsilon=4.0, top_k=8, top_ks=400, top_kc=8, submat=2)
import time
ncr_sum = 0
f1_sum = 0
var_sum = 0
ct_sum = 0
# 10 rounds
for t in range(10):
    t1 = time.time()
    result_fre_dict_svd, result_conf_dict, hitrate_rm = ldp_rm.find_itemset_svd(task='RM', method='AMN', singnum=0.5, use_group=True, group_num=5,
                                                                                        test='test_constant')
    t2 = time.time()
    consume_time = int(t2-t1)
    print('Final mining topks relations:',result_conf_dict)
    print('ldp_rm NCR', ncr:=metrics.NCR(result_conf_dict))
    print('ldp_rm F1', f1:=metrics.F1(result_conf_dict))
    print('ldp_rm VAR', var:=metrics.VARt(result_conf_dict))
    print('time:', ct:=consume_time)
    ncr_sum+= ncr
    f1_sum+=f1
    var_sum+=var
    ct_sum+=ct
print('average NCR:', round(ncr_sum/10,4))
print('average F1:', round(f1_sum/10,4))
print('average consume time:', round(ct_sum/10,4))