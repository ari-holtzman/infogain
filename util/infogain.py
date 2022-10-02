import random 

def e2e_gen_tests(mr, n_shuff_pos=0):
    mr = mr.split(', ')
    poss = [ ', '.join(mr) ] + [ ', '.join(random.sample(mr, k=len(mr))) for _ in n_shuff_pos ]
    negs = []
    for i in range(len(mr)):
        neg = ', '.join((mr[:i] + mr[i+1:]))
        negs.append(neg)
    return poss, negs
