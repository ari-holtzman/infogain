#!/usr/bin/env python
# coding: utf-8

import random
import os
import inspect
import json
import argparse

import torch
from torch.nn.functional import cross_entropy, log_softmax

import datasets
from util import const, icl, lm, infogain, util

parser = argparse.ArgumentParser()
parser.add_argument('out_file', type=str)
parser.add_argument('--n_tries', type=int, default=10)
parser.add_argument('--model', type=str, default='n125m')
parser.add_argument('--n_demos', type=int, default=10)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--print_int', type=int, default=1)
args = parser.parse_args()

assert(not os.path.exists(args.out_file))

util.set_all_seeds(args.seed)

e2e = datasets.load_dataset("GEM/e2e_nlg")
model, tokenizer, cw_length = lm.load(args.model, args.device)
icl_demos = random.sample(list(e2e['train']), k=args.n_demos)
demo_prompt = icl.basic_prompt(icl_demos, input_key='meaning_representation', output_key='target')
demo_prompt_idxs = tokenizer(demo_prompt).input_ids

to_write = []
all_data = []
all_results = []
verify_results, prob_results, mean_results = [], [], []
with torch.no_grad():
  for datum_idx, inference_datum in enumerate(e2e['validation']):
    inference_prompt = icl.inference_demo(inference_datum, input_key='meaning_representation')
    prompt = demo_prompt + inference_prompt
    prompt_idxs = tokenizer(prompt).input_ids
    poss, negs = infogain.e2e_gen_tests(inference_datum['meaning_representation'])
    tests = poss + negs
    test_inputs = []
    largest_len = float('-inf')
    for test in tests:
        test = icl.inference_demo({'input' : test})
        test_idxs = tokenizer(test).input_ids
        test_inputs.append(test_idxs)
        if len(test_idxs) > largest_len:
            largest_len = len(test_idxs)
    round_results = []

    prob_baseline, mean_baseline, verify, = None, None, None
    for try_idx in range(args.n_tries):
        gen = lm.gen(model, prompt_idxs, 40, p=0.5)[0]
        gen_idxs = tokenizer.encode(tokenizer.decode(gen['idxs'].cpu().tolist()).split('\n')[0])
        gen_len = len(gen_idxs)
        pad_lens = []
        cur_test_inputs = []
        for i in range(len(test_inputs)):
            test = test_inputs[i]
            pad_len = largest_len-len(test)
            pad_lens.append(pad_len)
            cur_test_inputs.append(demo_prompt_idxs + test + gen_idxs + [0]*pad_len)
        infogain_idxs = torch.LongTensor(cur_test_inputs).to(args.device)
        
        result = model(infogain_idxs)
        vocab_size = result.logits.size(-1)
        out_logits = torch.stack([ l[-gen_len-pad_len-1:-pad_len-2] for l, pad_len in zip(result.logits, pad_lens) ])
        gen_t = torch.LongTensor(gen_idxs).repeat(len(tests), 1)[:, :-1].to(args.device)
        lls = cross_entropy(out_logits.transpose(1,2), gen_t, reduction='none').mul(-1)
        ll_sums = lls.sum(1)
        infolosses = (ll_sums[1:] - ll_sums[0])
        verification = infolosses.le(0).tolist()
        max_loss = infolosses.max().item()
        gen['idxs'] = gen_idxs
        gen['logprobs'] = gen['logprobs'].tolist()
        gen['string'] = tokenizer.decode(gen_idxs)
        gen['ll'] = ll_sums[0].item()
        gen['lls'] = lls[0].tolist()
        gen['mean_ll'] = lls[0].mean().item()
        gen['poss'] = poss
        gen['pos_lls'] = lls[:len(poss)].tolist()
        gen['negs'] = negs
        gen['neg_lls'] = lls[len(poss):].tolist()
        gen['verification'] = verification
        gen['v_prop'] = sum(verification) / len(verification)
        gen['infolosses'] = infolosses.tolist()
        gen['max_loss'] = max_loss
        round_results.append(gen)
        if verify is None or \
            verify['v_prop'] < gen['v_prop'] or \
            (verify['v_prop'] == gen['v_prop'] and verify['max_loss'] > gen['max_loss']):
            verify = gen
        if prob_baseline is None or prob_baseline['ll'] < gen['ll']:
            prob_baseline = gen
        if mean_baseline is None or mean_baseline['mean_ll'] < gen['mean_ll']:
            mean_baseline = gen
    all_results.append(round_results)
    verify_results.append(verify)
    prob_results.append(prob_baseline)
    mean_results.append(mean_baseline)

    datum = { 
                'verify' : verify,
                'prob' : prob_baseline,
                'mean' : mean_baseline,
                'round_results' : round_results,
            }
    all_data.append(datum)
    to_write.append(datum)
    if (datum_idx+1) % args.print_int == 0:
        print(datum_idx+1)
        with open(args.out_file, 'a') as out:
            for datum in to_write:
                print(datum)
                out.write(f'{json.dumps(datum)}\n')
        to_write = []
with open(args.out_file, 'a') as out:
    for datum in to_write:
        out.write(f'{json.dumps(datum)}\n')
