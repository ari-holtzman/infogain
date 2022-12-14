tpimport random
import argparse

import torch

from datasets import load_dataset

from transformers import AutoTokenizer, AutoModelForCausalLM

from util import mt

XGLM_MAX_LENGTH = 2048

parser = argparse.ArgumentParser()
parser.add_argument('src_lang', type=str)
parser.add_argument('tgt_lang', type=str)
parser.add_argument('src_name', type=str)
parser.add_argument('tgt_name', type=str)
parser.add_argument('demo_split', type=str)
parser.add_argument('infr_split', type=str)
parser.add_argument('out_path', type=str)
parser.add_argument('--s', type=float, default=1, help='scaling factor for PMI adjustment')
parser.add_argument('--n_demos', type=int, default=32)
parser.add_argument('--max_len', type=int, default=100)
parser.add_argument('--model_name', type=str, default='facebook/xglm-7.5B')
args = parser.parse_args()

src_splits = load_dataset("gsarti/flores_101", args.src_lang)
tgt_splits = load_dataset("gsarti/flores_101", args.tgt_lang)
src_name, tgt_name = args.src_name, args.tgt_name
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
model = AutoModelForCausalLM.from_pretrained(args.model_name).eval().half().cuda()
src_points = random.sample(list(src_splits[args.demo_split]), k=args.n_demos)
tgt_points = [ tgt_splits[args.demo_split][point['id']-1] for point in src_points ]
demo_prompt_ids = mt.make_demo_prompt(src_points, tgt_points, src_name, tgt_name, tokenizer)
uncond_demo_prompt_ids = mt.make_uncond_demo_prompt(tgt_points, tokenizer)
uncond_prompt_t = torch.LongTensor([uncond_demo_prompt_ids]).to(model.device)

with torch.no_grad(), open(args.out_path, 'w') as out:
   for i, infr_point in enumerate(src_splits[args.infr_split]):
       prompt_ids = demo_prompt_ids + mt.xglm_ids(f"{src_name} : {infr_point['sentence']} {tgt_name} :", tokenizer)
       if len(prompt_ids)+args.max_len > XGLM_MAX_LENGTH:
           prompt_ids = prompt_ids[len(promt_ids)+args.max_len-XGLM_MAX_LENGTH:]
       cond_prompt_t = torch.LongTensor([prompt_ids]).to(model.device)
       gen_ids = mt.tp_greedy(model, args.s, cond_prompt_t, uncond_prompt_t, args.max_len)
       gen_str = tokenizer.decode(gen_ids[0].tolist()).split('</s>')[0].strip()
       out.write(f'{gen_str}\n')
       print(i)
       print(gen_str)
