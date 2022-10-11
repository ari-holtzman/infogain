import random

import torch
from torch.nn.functional import softmax, log_softmax
eps = torch.finfo(torch.float32).eps

# Tokenization

def xglm_ids(s, tokenizer):
    # strip first token, which the tokenizer always makes </s>
    return tokenizer(s).input_ids[1:]

def xglm_tokenize(s, tokenizer):
  ids = []
  i = 0
  t = 0
  while t < len(s):
    c = s[t]
    if c == '\n':
      ids += xglm_ids(s[i:t], tokenizer) + [2]
      i = t+1
      t = t+1
    else:
      t += 1
  ids += xglm_ids(s[i:], tokenizer)
  return ids

def post_proc(s):
    return s.replace('</s>', '\n')

# Prompting

def make_demo_prompt(src_points, tgt_points, src_name, tgt_name, tokenizer):
    assert(len(src_points) == len(tgt_points))
    demos = []
    for s, t in zip(src_points, tgt_points):
        assert(s['id'] == t['id'])
        demo = f"{src_name} : {s['sentence']} {tgt_name} : {t['sentence']}"
        demos.append(demo)
    demo_prompt_ids = [2]
    for demo in demos:
      demo_prompt_ids.extend(xglm_tokenize(demo, tokenizer))
      demo_prompt_ids.append(2)
    return demo_prompt_ids

def make_uncond_demo_prompt(tgt_points, tokenizer):
    uncond_demo_prompt_ids = [2]
    for t in tgt_points:
      uncond_demo_prompt_ids.extend(xglm_tokenize(t['sentence'], tokenizer))
      uncond_demo_prompt_ids.append(2)
    return uncond_demo_prompt_ids

# Logit and Probability Operations

def top_p_mask(probs, p):
  sorted_probs, sorted_indices = torch.sort(probs, descending=True)
  cumulative_probs = sorted_probs.cumsum(dim=-1)
  sorted_indices_to_keep = cumulative_probs < p
  sorted_indices_to_keep[..., 0] = True
  if sorted_probs[sorted_indices_to_keep].sum() < p:
    sorted_indices_to_keep[..., 1:] = sorted_indices_to_keep[..., :-1].clone()
  indices_to_keep = sorted_indices_to_keep.scatter(0, sorted_indices, sorted_indices_to_keep)
  return indices_to_keep

# Decoding Algorithms

def greedy(model, prompt_ids, length):
    gen_ids = torch.LongTensor([[]]).to(model.device)
    for i in range(length):
        cur_ids = torch.cat([prompt_ids, gen_ids], dim=1)
        logits = model(cur_ids).logits
        next_token_id = logits[0][-1].argmax()
        gen_ids = torch.LongTensor([ gen_ids[0].tolist() + [next_token_id] ]).to(model.device)
    return gen_ids

def sample(model, prompt_ids, length, temp=None):
    gen_ids = torch.LongTensor([[]]).to(model.device)
    for i in range(length):
        cur_ids = torch.cat([prompt_ids, gen_ids], dim=1)
        logits = model(cur_ids).logits[0][-1]
        if temp is not None:
            logits.div_(temp)
        probs = softmax(logits, dim=0)
        next_token_id = torch.multinomial(probs, 1).item()
        gen_ids = torch.LongTensor([ gen_ids[0].tolist() + [next_token_id] ]).to(model.device)
    return gen_ids

def cfg_greedy(model, scale, c_ids, u_ids, length, **kwargs):
    gen_ids = torch.LongTensor([[]]).to(model.device)
    for i in range(length):
        cur_c_ids = torch.cat([c_ids, gen_ids], dim=1)
        cur_u_ids = torch.cat([u_ids, gen_ids], dim=1)
        c_logits = model(cur_c_ids).logits[0][-1]
        u_logits = model(cur_u_ids).logits[0][-1]
        logits = c_logits + scale*(c_logits-u_logits) 
        next_token_id = logits.argmax()
        gen_ids = torch.LongTensor([ gen_ids[0].tolist() + [next_token_id] ]).to(model.device)
    return gen_ids

def cfg_k_greedy(model, scale, c_ids, u_ids, length, k=5):
    gen_ids = torch.LongTensor([[]]).to(model.device)
    for i in range(length):
        cur_c_ids = torch.cat([c_ids, gen_ids], dim=1)
        cur_u_ids = torch.cat([u_ids, gen_ids], dim=1)
        c_logits = model(cur_c_ids).logits[0][-1]
        c_topk = torch.topk(c_logits, k).indices
        u_logits = model(cur_u_ids).logits[0][-1]
        logits = c_logits + scale*(c_logits-u_logits) 
        next_token_id = c_topk[logits[c_topk].argmax()].item()
        gen_ids = torch.LongTensor([ gen_ids[0].tolist() + [next_token_id] ]).to(model.device)
    return gen_ids

def cfg_sample(model, scale, c_ids, u_ids, length, temp=None):
    gen_ids = torch.LongTensor([[]]).to(model.device)
    for i in range(length):
        cur_c_ids = torch.cat([c_ids, gen_ids], dim=1)
        cur_u_ids = torch.cat([u_ids, gen_ids], dim=1)
        c_logits = model(cur_c_ids).logits[0][-1]
        u_logits = model(cur_u_ids).logits[0][-1]
        if temp is not None:
            c_logits.div_(temp)
            u_logits.div_(temp)
        logits = c_logits + scale*(c_logits-u_logits) 
        probs = softmax(logits, dim=0)
        next_token_id = torch.multinomial(probs, 1).item()
        gen_ids = torch.LongTensor([ gen_ids[0].tolist() + [next_token_id] ]).to(model.device)
    return gen_ids

def true_cfg_greedy(model, scale, c_ids, u_ids, length):
    gen_ids = torch.LongTensor([[]]).to(model.device)
    for i in range(length):
        cur_c_ids = torch.cat([c_ids, gen_ids], dim=1)
        cur_u_ids = torch.cat([u_ids, gen_ids], dim=1)
        c_logits = model(cur_c_ids).logits[0][-1]
        u_logits = model(cur_u_ids).logits[0][-1]
        logits = u_logits + scale*(c_logits-u_logits) 
        next_token_id = logits.argmax()
        gen_ids = torch.LongTensor([ gen_ids[0].tolist() + [next_token_id] ]).to(model.device)
    return gen_ids

def cfg_greedy_explore(model, scale, c_ids, u_ids, length, tokenizer, k=5):
    gen_ids = torch.LongTensor([[]]).to(model.device)
    for i in range(length):
        cur_c_ids = torch.cat([c_ids, gen_ids], dim=1)
        cur_u_ids = torch.cat([u_ids, gen_ids], dim=1)
        c_logits = model(cur_c_ids).logits[0][-1]
        c_topk = torch.topk(c_logits, k).indices
        c_cover = softmax(c_logits, dim=0)[c_topk].sum()
        u_logits = model(cur_u_ids).logits[0][-1]
        logits = c_logits + scale*(c_logits-u_logits) 
        cfg_topk = torch.topk(logits, k).indices
        cfg_cover = softmax(logits, dim=0)[cfg_topk].sum()
        next_token_id = logits.argmax()
        print(f"{next_token_id} | {tokenizer.decode(cfg_topk)} {cfg_cover.item()}| {tokenizer.decode(c_topk)} {c_cover.item()}")
        gen_ids = torch.LongTensor([ gen_ids[0].tolist() + [next_token_id] ]).to(model.device)
    return gen_ids

def cfg_greedy_debug(model, scale, c_ids, u_ids, length, tokenizer, k=5):
    gen_ids = torch.LongTensor([[]]).to(model.device)
    for i in range(length):
        cur_c_ids = torch.cat([c_ids, gen_ids], dim=1)
        cur_u_ids = torch.cat([u_ids, gen_ids], dim=1)
        c_logits = model(cur_c_ids).logits[0][-1]
        c_topk_res = torch.topk(c_logits, k)
        c_topk = c_topk_res.indices
        c_cover = softmax(c_logits, dim=0)[c_topk].sum()
        u_logits = model(cur_u_ids).logits[0][-1]
        logits = c_logits + scale*(c_logits-u_logits) 
        cfg_topk_res = torch.topk(logits, k)
        cfg_topk = cfg_topk_res.indices
        cfg_cover = softmax(logits, dim=0)[cfg_topk].sum()
        next_token_id = logits.argmax()
        print(f"{next_token_id} | {tokenizer.decode(cfg_topk)} {cfg_cover.item()}| {tokenizer.decode(c_topk)} {c_cover.item()}")
        gen_ids = torch.LongTensor([ gen_ids[0].tolist() + [next_token_id] ]).to(model.device)
    return gen_ids

def cfg_k_greedy_explore(model, scale, c_ids, u_ids, length, tokenizer, k=5):
    gen_ids = torch.LongTensor([[]]).to(model.device)
    for i in range(length):
        cur_c_ids = torch.cat([c_ids, gen_ids], dim=1)
        cur_u_ids = torch.cat([u_ids, gen_ids], dim=1)
        c_logits = model(cur_c_ids).logits[0][-1]
        c_topk = torch.topk(c_logits, k).indices
        c_cover = softmax(c_logits, dim=0)[c_topk].sum()
        u_logits = model(cur_u_ids).logits[0][-1]
        logits = c_logits + scale*(c_logits-u_logits) 
        cfg_topk = torch.topk(logits, k).indices
        cfg_cover = softmax(logits, dim=0)[cfg_topk].sum()
        next_token_id = c_topk[logits[c_topk].argmax()].item()
        # print(tokenizer.decode(c_topk[torch.topk(logits[c_topk], k).indices]))
        # print(torch.topk(logits[c_topk], k).values.tolist())
        print(f"{next_token_id} | {tokenizer.decode(cfg_topk)} {cfg_cover.item()}| {tokenizer.decode(c_topk)} {c_cover.item()}") # | {tokenizer.decode(c_topk[torch.topk(logits[c_topk], k).indices])}")
        gen_ids = torch.LongTensor([ gen_ids[0].tolist() + [next_token_id] ]).to(model.device)
    return gen_ids

def cfg_p_greedy(model, scale, c_ids, u_ids, length, p=0.5, **kwargs):
    gen_ids = torch.LongTensor([[]]).to(model.device)
    for i in range(length):
        cur_c_ids = torch.cat([c_ids, gen_ids], dim=1)
        cur_u_ids = torch.cat([u_ids, gen_ids], dim=1)
        c_logits = model(cur_c_ids).logits[0][-1]
        c_probs = softmax(c_logits, dim=0)
        c_p_set = top_p_mask(c_probs, p).nonzero().view(-1)
        u_logits = model(cur_u_ids).logits[0][-1]
        logits = c_logits + scale*(c_logits-u_logits) 
        next_token_id = c_p_set[logits[c_p_set].argmax()].item()
        gen_ids = torch.LongTensor([ gen_ids[0].tolist() + [next_token_id] ]).to(model.device)
    return gen_ids


def cfg_p_greedy_explore(model, scale, c_ids, u_ids, length, tokenizer, p=0.5):
    gen_ids = torch.LongTensor([[]]).to(model.device)
    for i in range(length):
        cur_c_ids = torch.cat([c_ids, gen_ids], dim=1)
        cur_u_ids = torch.cat([u_ids, gen_ids], dim=1)
        c_logits = model(cur_c_ids).logits[0][-1]
        c_probs = softmax(c_logits, dim=0)
        c_p_set = top_p_mask(c_probs, p).nonzero().view(-1)
        c_p_set = c_p_set[torch.topk(c_logits[c_p_set], c_p_set.size(0)).indices]
        c_cover = c_probs[c_p_set].sum().item()
        u_logits = model(cur_u_ids).logits[0][-1]
        logits = c_logits + scale*(c_logits-u_logits) 
        cfg_cover = softmax(logits, dim=0)[c_p_set].sum().item()
        next_token_id = c_p_set[logits[c_p_set].argmax()].item()
        cfg_p_set = c_p_set[torch.topk(logits[c_p_set], c_p_set.size(0)).indices]
        print(f"{next_token_id} | {tokenizer.decode(cfg_p_set)} {cfg_cover}| {tokenizer.decode(c_p_set)} {c_cover}")
        gen_ids = torch.LongTensor([ gen_ids[0].tolist() + [next_token_id] ]).to(model.device)
    return gen_ids

def pg_greedy(model, alpha, temp, c_ids, u_ids, length, **kwargs):
    gen_ids = torch.LongTensor([[]]).to(model.device)
    for i in range(length):
        cur_c_ids = torch.cat([c_ids, gen_ids], dim=1)
        cur_u_ids = torch.cat([u_ids, gen_ids], dim=1)
        c_logits = model(cur_c_ids).logits[0][-1].double()
        u_logits = model(cur_u_ids).logits[0][-1].double()
        c_lls = log_softmax(c_logits, dim=0)
        u_lls = log_softmax(u_logits, dim=0)
        pmi = c_lls - u_lls
        pmi_dist = pmi - pmi.min() + eps
        pmi_dist = pmi_dist / pmi_dist.sum()
        pmi_lls = pmi_dist.log()
        temp_c_lls = log_softmax(c_logits / temp, dim=0)
        lls = temp_c_lls + alpha*pmi_lls
        next_token_id = lls.argmax()
        gen_ids = torch.LongTensor([ gen_ids[0].tolist() + [next_token_id] ]).to(model.device)
    return gen_ids


