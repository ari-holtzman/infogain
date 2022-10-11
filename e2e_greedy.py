#!/usr/bin/env python
# coding: utf-8

import math
import random
import os
import json
import argparse

import inspect
import code

import torch
from torch.nn.functional import cross_entropy, log_softmax

import datasets
from util import const, icl, lm, infogain, util

parser = argparse.ArgumentParser()
parser.add_argument('out_file', type=str)
parser.add_argument('--gen_len', type=int, default=50)
parser.add_argument('--model', type=str, default='j')
parser.add_argument('--n_demos', type=int, default=1)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--worker_id', type=int, default=None)
parser.add_argument('--n_shards', type=int, default=None)
parser.add_argument('--print_int', type=int, default=1)
parser.add_argument('--override', action='store_true')
args = parser.parse_args()

if args.override:
    with open(args.out_file, 'w') as out:
        pass
else:
    assert(not os.path.exists(args.out_file))

util.set_all_seeds(args.seed)

e2e = datasets.load_dataset("GEM/e2e_nlg")
data = e2e['validation']
model, tokenizer, cw_length = lm.load(args.model, args.device)
icl_demos = random.sample(list(e2e['train']), k=args.n_demos)
demo_prompt = icl.basic_prompt(icl_demos, input_key='meaning_representation', output_key='target')
demo_prompt_idxs = tokenizer(demo_prompt).input_ids

if args.worker_id is None or args.n_shards is None:
    args.i = 0
    args.j = float('inf')
else:
    shard_len = math.ceil(len(data) / args.n_shards)
    args.i = args.worker_id*shard_len
    args.j = len(data) if args.worker_id == args.n_shards-1 else (args.worker_id+1)*shard_len

to_write = []
all_data = []
all_results = []
with torch.no_grad():
  for datum_idx, inference_datum in enumerate(data):
    if datum_idx < args.i:
        continue
    elif datum_idx >= args.j:
        break
    inference_prompt = icl.inference_demo(inference_datum, input_key='meaning_representation')
    prompt = demo_prompt + inference_prompt
    prompt_idxs = tokenizer(prompt).input_ids
    datum = lm.gen(model, prompt_idxs, args.gen_len, greedy=True)
    datum['idxs'] = datum['idxs'].tolist()
    all_data.append(datum)
    to_write.append(datum)
    if (datum_idx+1) % args.print_int == 0:
        print(datum_idx+1)
        with open(args.out_file, 'a') as out:
            for datum in to_write:
                out.write(f'{json.dumps(datum)}\n')
        to_write = []
with open(args.out_file, 'a') as out:
    for datum in to_write:
        out.write(f'{json.dumps(datum)}\n')
