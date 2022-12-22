from itertools import islice
import numpy as np

def window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield np.array(result)
    for elem in it:
        result = result[1:] + (elem,)
        yield np.array(result)


def nansafe_prod(min_nonnan, sub_seq, weight):
    num_nan = np.count_nonzero(np.isnan(sub_seq))
    if num_nan > min_nonnan:
        prod = np.array([[np.nan]])
    elif num_nan == 0:
        prod = weight.reshape(1, -1) @ sub_seq.reshape(-1, 1)
    else:
        sub_seq_masked = np.ma.array(sub_seq, mask=np.isnan(sub_seq))
        # Also need to renormalize the weight
        weight = np.ma.array(weight, mask=np.isnan(sub_seq))
        weight /= np.nansum(weight)
        prod = np.array(np.ma.dot(weight.reshape(1, -1), sub_seq_masked.reshape(-1, 1)))
        # if num_nan > 9:
        #     print("=========================================")
        #     print(sub_seq)
        #     print(weight)
        #     print(prod)
    prod = prod[0, 0]
    # if np.isnan(prod):
    #     print(weight, sub_seq)
    return prod
