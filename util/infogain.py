import random 
import re

def get_pad_lens(m, pad=0):
  if len(m) < 2:
    return [0]*len(m)
  max_len = float('-inf')
  for row in m:
    max_len = max(max_len, len(row))
  pad_lens = []
  for row in m:
    pad_len = max_len-len(row)
    pad_lens.append(pad_len)
  return pad_lens

# NLG

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

# NLU

def multirc_make_tests(question_text, answer_texts):
  tests = []
  for answer in answer_texts:
    pos = f'{question_text.strip()} {answer.strip()}'
    negs = [ f'{answer.strip()}', f'{question_text.strip()}' ]
    tests.append((pos, negs))
  return tests

def multirc_proc_passage(s):
  s = re.sub('<b>.*?</b>', '', s)
  secs = s.split('<br>')
  secs = [ sec.strip() for sec in secs ]
  secs = list(filter(lambda s: len(s) > 0, secs))
  secs = [ ' '.join(sec.split()) for sec in secs]
  return secs
