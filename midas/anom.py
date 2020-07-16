import math

import numpy as np
from tqdm import tqdm

from midas.edgehash import Edgehash
from midas.nodehash import Nodehash

__all__ = [
    'midas',
    'midasR',
]

def counts_to_anom(tot, cur, cur_t):
    cur_mean = tot / cur_t
    
    # Why taking the max here? Only consider error if `cur` is greater than `cur_mean`?
    sqerr = np.power(max(0, cur - cur_mean), 2)
    
    # Why not using the same formula in the paper, the same but shorter?
    # sqerr / cur_mean * (t/(t-1))
    return sqerr / cur_mean + sqerr / (cur_mean * max(1, cur_t - 1))

def getRowInfo(row, anom_score,cur_count,total_count,cur_t):
    cur_src = int(row["src"]) 
    cur_dst = int(row["dst"])
    # count is added row by row
    # so, multiple (src,dst) pairs at the same time tick have different scores because they are processed in order
    # should they have the same score given that the records are identical?
    cur_count.insert(cur_src, cur_dst, 1)
    total_count.insert(cur_src, cur_dst, 1)
    cur_mean = total_count.get_count(cur_src, cur_dst) / cur_t
    # OK, it doesn't take the max between 0 and signed error
    sqerr = np.power(cur_count.get_count(cur_src, cur_dst) - cur_mean, 2)
    cur_score = 0 if cur_t == 1 else sqerr / cur_mean + sqerr / (cur_mean * (cur_t - 1))
    cur_score = 0 if math.isnan(cur_score) else cur_score
    anom_score.append(cur_score)

def midas(df, num_rows, num_buckets):
    m = df.src.max()
    cur_count = Edgehash(num_rows, num_buckets, m)
    total_count = Edgehash(num_rows, num_buckets, m)
    anom_score = []
    
    timestamp_keys =  df.groupby(["timestamp"]).groups.keys()
    for timeframe in tqdm(timestamp_keys):
        cur_t = timeframe
        curr_df = df.groupby(["timestamp"]).get_group(timeframe)        
        curr_df.apply(lambda row: getRowInfo(row, anom_score,cur_count,total_count,cur_t), axis=1)
        cur_count.clear()
    return anom_score


def midasR(src, dst, times, num_rows, num_buckets, factor):
    m = np.max(src)
    num_entries = src.shape[0]
    cur_count = Edgehash(num_rows, num_buckets, m)
    total_count = Edgehash(num_rows, num_buckets, m)
    src_score = Nodehash(num_rows, num_buckets)
    dst_score = Nodehash(num_rows, num_buckets)
    src_total = Nodehash(num_rows, num_buckets)
    dst_total = Nodehash(num_rows, num_buckets)
#     anom_score = np.zeros(num_entries)
    anom_score = []
    cur_t = 1
    for i in tqdm(range(num_entries)):
        if i == 0 or times[i] > cur_t:
            cur_count.lower(factor)
            src_score.lower(factor)
            dst_score.lower(factor)
            cur_t = times[i]

        cur_src = src[i]
        cur_dst = dst[i]
        cur_count.insert(cur_src, cur_dst, 1)
        total_count.insert(cur_src, cur_dst, 1)
        src_score.insert(cur_src, 1)
        dst_score.insert(cur_dst, 1)
        src_total.insert(cur_src, 1)
        dst_total.insert(cur_dst, 1)

        cur_score = counts_to_anom(
            total_count.get_count(cur_src, cur_dst),
            cur_count.get_count(cur_src, cur_dst),
            cur_t,
        )
        cur_score_src = counts_to_anom(
            src_total.get_count(cur_src), src_score.get_count(cur_src), cur_t
        )
        cur_score_dst = counts_to_anom(
            dst_total.get_count(cur_dst), dst_score.get_count(cur_dst), cur_t
        )
        combined_score = max(cur_score_src, cur_score_dst, cur_score)
#         anom_score[i] = np.log(1 + combined_score)
        # It would be useful to seperate between those 3 scores
        anom_score.append((np.log(1 + cur_score_src), np.log(1 + cur_score_dst), np.log(1 + cur_score)))

    return anom_score
