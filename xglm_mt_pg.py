import os
import random
import argparse

import torch

from datasets import load_dataset

from transformers import AutoTokenizer, AutoModelForCausalLM

from util import mt
from util.util import set_all_seeds

XGLM_MAX_LENGTH = 2048

parser = argparse.ArgumentParser()
parser.add_argument('src_lang', type=str)
parser.add_argument('tgt_lang', type=str)
parser.add_argument('demo_split', type=str)
parser.add_argument('infr_split', type=str)
parser.add_argument('out_dir', type=str)
parser.add_argument('--s', type=float, default=1)
parser.add_argument('--n_demos', type=int, default=32)
parser.add_argument('--max_len', type=int, default=100)
parser.add_argument('--model_name', type=str, default='facebook/xglm-564M')
parser.add_argument('--worker_id', type=int, default=0)
parser.add_argument('--n_workers', type=int, default=1)
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()
set_all_seeds(args.seed)

args.out_dir = os.path.join(args.out_dir, f'{args.src_lang}-{args.tgt_lang}.{str(args.worker_id).zfill(3)}.101.txt')
src_splits = load_dataset("gsarti/flores_101", args.src_lang)
tgt_splits = load_dataset("gsarti/flores_101", args.tgt_lang)
src_points = random.sample(list(src_splits[args.demo_split]), k=args.n_demos)
tgt_points = [ tgt_splits[args.demo_split][point['id']-1] for point in src_points ]
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
model = AutoModelForCausalLM.from_pretrained(args.model_name).eval().half().cuda()
demo_prompt_ids = mt.make_demo_prompt(src_splits[args.demo_split], tgt_splits[args.demo_split], tokenizer)
uncond_prompt_ids = mt.right_crop(demo_prompt_ids + mt.xglm_ids("=", tokenizer), XGLM_MAX_LENGTH-args.max_len)
uncond_prompt_t = torch.LongTensor([uncond_prompt_ids]).to(model.device)

with torch.no_grad(), open(args.out_dir, 'w') as out:
   for i, infr_point in enumerate(src_splits[args.infr_split]):
       if (i % args.n_workers) != args.worker_id:
           continue
       prompt_ids = demo_prompt_ids + mt.make_infr_prompt(infr_point, tokenizer)
       prompt_ids = mt.right_crop(prompt_ids, XGLM_MAX_LENGTH-args.max_len)
       prompt_t = torch.LongTensor([prompt_ids]).to(model.device)
       gen_ids = mt.tp_greedy(model, args.s, prompt_t, uncond_prompt_t, args.max_len)
       gen_str = tokenizer.decode(gen_ids[0].tolist()).split('</s>')[0].strip()
       out.write(f'{gen_str}\n')
       print(i)
       print(gen_str)
