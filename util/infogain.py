import random 

def e2e_gen_tests(mr, n_shuff_pos=0):
    mr = mr.split(', ')
    poss = [ ', '.join(mr) ] + [ ', '.join(random.sample(mr, k=len(mr))) for _ in range(n_shuff_pos) ]
    negs = []
    for i in range(len(mr)):
        item = mr[i]
        key, value = mr[i].split('[')
        key_only = key + '[]'
        neg = ', '.join((mr[:i] + [key_only] + mr[i+1:]))
        negs.append(neg)
    return poss, negs
