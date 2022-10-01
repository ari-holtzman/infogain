import pdb

import sys
import os
import json
import time

import torch
from torch.nn import functional as F

import openai

from transformers import GPT2Tokenizer

from .const import OPENAI_API

## load necessary tools and data

# load_openai key
def load_openai_key(key_path='api.key'):
    if not os.path.exists(key_path):
        raise ValueError(f'OpenAI key could not be loaded because {key_path} does not exist!')
    with open(key_path) as f:
        api_key = f.read().strip()
    openai.api_key = api_key

# load tokenizer for all GPT-{2,Neo,J}
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# # load bytes to token mapping for GPT-3
# with open('data/3toi.json') as f:
#     byte2idx = json.load(f)
# 
# 
# ## utility functions
# 
# def tok2idx(tok):
#     if tok.startswith('bytes:'):
#         res = byte2idx[tok]
#     else:
#         # TODO whitespace fix
#         # TODO make actual mapping
#         res = tokenizer.encode(tok)[0]
#     if type(res) != int:
#         pdb.set_trace()
#     return res


## wrappers around various libraries

def gpt3_helper(wait_time, **args):
    # call GPT-3 API until result is provided and then return it
    response = None
    received = False
    while not received:
        try:
            response = openai.Completion.create(**args)
            received = True
        except:
            error = sys.exc_info()[0]
            if error == openai.error.InvalidRequestError: 
                # something is wrong: e.g. prompt too long
                print(f"InvalidRequestError\nPrompt passed in:\n\n{prompt}\n\n")
                assert False
            print("API error:", error)
            time.sleep(wait_time)
    return response

def gpt3(idxs, max_tokens, model='ada', t=1, p=1, logprobs=1,
         n=1, stop=None, echo=True, stream=False, presence_penalty=0, 
         frequency_penalty=0,logit_bias={}, best_of=None, wait_time=10):
    if best_of is None:
        response = gpt3_helper(
            prompt=idxs,
            max_tokens=max_tokens,
            engine=model,
            temperature=t,
            top_p=p,
            logprobs=logprobs,
            n=n,
            stop=stop, 
            echo=echo,
            stream=stream,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            logit_bias={},
            wait_time=wait_time)
    else:
        response = gpt3_helper(
            prompt=idxs,
            max_tokens=max_tokens,
            engine=model,
            temperature=t,
            top_p=p,
            logprobs=logprobs,
            n=n,
            stop=stop, 
            echo=echo,
            stream=stream,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            logit_bias={},
            best_of=best_of,
            wait_time=wait_time)
    return response

def gpt2p():
    raise NotImplementedError

## universal interface

def gpt3_wrapper(model, idxs, **args):
    if args.get('greedy', False):
        args['t'] = 0
        del args['greedy']
    response = gpt3(model=model, idxs=idxs, **args)

    outputs = []
    for choice in response.choices:
        output = {}
        output['model'] = model
        output['text'] = choice['text']
        # get rid of the first logprob, as it is always "None"
        output['logprobs'] = choice['logprobs']['token_logprobs'][1:]
        output['idxs'] = [ tok2idx(tok) for tok in choice['logprobs']['tokens'] ]

        if output.get('include_original', False):
            for k in response.keys():
                if k == 'choices':
                    pass
                else:
                    choice[key] = response[key]
            output['orig'] = choice

        outputs.append(output)

    return outputs

# TODO support other tokenizers
def prompt2tokens(prompt):

    # different preprocessing for different ways of giving prompts!
    prompt_type = type(prompt)
    if prompt_type == torch.Tensor:
        idxs = prompt.cpu().numpy().tolist()
    elif prompt_type == list:
        if len(prompt) == 0:
            raise ValueError("Can't use zero-length prompt!")
        elif type(prompt[0]) == int:
            idxs = prompt
        elif type(prompt[0]) == str:
            idxs = tokenizer.batch_encode_plus(prompt)['input_ids']
        else:
            raise ValueError(f"Don't know how to interpret prompt of type List[{type(prompt[0])}]!")
    elif prompt_type == str:
        idxs = [tokenizer.encode(prompt)]
    else:
        raise TypeError(f"Don't know how to interpret {prompt_type} as tokens!")

    return idxs
    
def hf_logprobs_wrapper(model, idxs, cont_idxs=None, **args):
    """a simple wrapper for getting logprobs from GPT-2 like models"""
    
    idxs = torch.tensor(idxs, device=model.device, dtype=torch.int64)
    if idxs.dim() == 1:
        idxs = idxs.unsqueeze(0)

    if cont_idxs is not None:
        cont_idxs = torch.tensor(cont_idxs, device=model.device, dtype=torch.int64)
        if cont_idxs.dim() == 1:
            cont_idxs = cont_idxs.unsqueeze(0)
        prompt_idxs = idxs
        idxs = torch.cat([idxs, cont_idxs], dim=1)

    r = model(idxs)
    all_logprobs = F.log_softmax(r['logits'], dim=-1)
    logprobs = all_logprobs[:, :-1].gather(-1, idxs[:, 1:].unsqueeze(-1)).squeeze(-1)

    outputs = []
    for seq_idxs, lps in zip(idxs, logprobs):
        prompt_len = prompt_idxs.size(1)
        output = {}
        output['model'] = model.config._name_or_path
        output['text'] = tokenizer.decode(seq_idxs[prompt_len:])
        # logprobs excludes first token already
        output['logprobs'] = lps[prompt_len-1:]
        output['idxs'] = seq_idxs[prompt_len:]
        outputs.append(output)

    return outputs
  

def logprobs(model, prompt, **args):
    """get log probabilities from an arbitrary model, return in a canonical format"""

    # since this is a function for getting probabilities, not generating
    # we don't allow a "max_tokens" argument
    assert('max_tokens' not in args)
    
    if type(prompt) != list:
        prompt = [prompt]
    
    prompt_idxs = prompt2tokens(prompt) 
    # TODO fix this
    # for p, i in zip(prompt, prompt_idxs):
    #     assert(p.rstrip() == p)
    #     assert(tokenizer.decode(i) == p)
    if 'cont' in args:
        cont = args['cont']
        cont_idxs = prompt2tokens(cont)

    with torch.no_grad():
        if model not in OPENAI_API:
            assert(type(model) != str)
            if 'cont' in args:
                outputs = hf_logprobs_wrapper(model=model, idxs=prompt_idxs, cont_idxs=cont_idxs, **args)
            else:
                outputs = hf_logprobs_wrapper(model=model, idxs=prompt_idxs, **args)
        else:
            # max_tokens is hardcoded to be 0 as this function is only meant for
            # querying probabilities of fixed strings
            if 'cont' not in args:
                outputs = gpt3_wrapper(model=model, idxs=prompt_idxs, max_tokens=0, **args)
            else:
                del args['cont']
                idxs = [ p+c for p, c in zip(prompt_idxs, cont_idxs) ]
                outputs = gpt3_wrapper(model=model, idxs=idxs, max_tokens=0, **args)
                for i in range(len(outputs)):
                    output = outputs[i]
                    output['idxs'] = cont_idxs[i]
                    output['text'] = tokenizer.decode(cont_idxs[i])
                    # logprobs excludes first token already
                    output['logprobs'] = output['logprobs'][len(prompt_idxs[i])-1:]

    return outputs

def hf_greedy_wrapper(model, prompt, max_tokens, eos=50256, cache=None, use_cache=True):
    with torch.no_grad():
        assert(prompt.size(0) == 1)
        gen, lps = None, None
        for i in range(max_tokens - prompt.size(1)):
            if not use_cache or cache is None:
                logits, cache = model(prompt).to_tuple()
            else:
                logits, cache = model(prompt, past_key_values=cache).to_tuple()
            next_token_idx = logits[0][-1].argmax()
            next_token_lp = F.log_softmax(logits[0][-1], dim=-1)[next_token_idx].resize_(1, 1)
    
            if next_token_idx.item() == eos:
                break
    
            next_token_tensor = next_token_idx.resize_(1, 1)
            if i == 0:
                gen = next_token_tensor
                lps = next_token_lp
            else:
                gen = torch.cat([gen, next_token_tensor], dim=1)
                lps = next_token_lp
    
            if use_cache:
                prompt = next_token_tensor
            else:
                prompt = torch.cat([prompt, next_token_tensor], dim=1)
    return gen, lps

def hf_gen_wrapper(model, idxs, max_tokens, **args):
    """a simple wrapper for generating from GPT-2 like models"""

    if args.get('echo', False):
        raise NotImplementedError('echoing not implemented for HF based generation')

    idxs = torch.tensor(idxs, device=model.device, dtype=torch.int64)
    if idxs.dim() == 1:
        idxs = idxs.unsqueeze(0)

    greedy = args.get('greedy', False)
    cached = args.get('cache') is not None if 'cache' in args else False
    max_length = idxs.size(1) + max_tokens
    if greedy:
        gen_idxs, logprobs = hf_greedy_wrapper(model, idxs, max_length, tokenizer.eos_token_id, cache=args.get('cache', None), use_cache=args.get('use_cache', True))
        prompt_len = 0
    else:
        assert('cache' not in args)
        r = model.generate(
                    input_ids=idxs,
                    max_length=max_length,
                    temperature=args.get('t', 1),
                    top_k=args.get('k', 0),
                    top_p=args.get('p', 1),
                    do_sample=True,
                    num_return_sequences=args.get('n', 1),
                    return_dict_in_generate=True,
                    output_scores=True,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

        gen_idxs = r['sequences']
        all_scores = r['scores']

        prompt_len = idxs.size(1)

        all_logprobs = torch.stack([ torch.nn.functional.log_softmax(scores, dim=-1) for scores in all_scores ], dim=1)
        logprobs = all_logprobs[:, :-1].gather(-1, gen_idxs[:, prompt_len:-1].unsqueeze(-1)).squeeze(-1)


    outputs = []
    for i, (seq_idxs, lps) in enumerate(zip(gen_idxs, logprobs)):
        output = {}
        output['model'] = model.config._name_or_path
        output['text'] = tokenizer.decode(seq_idxs) #[prompt_len:])
        output['logprobs'] = logprobs[i]
        output['idxs'] = seq_idxs[prompt_len:]
        outputs.append(output)

    return outputs
 
def gen(model, prompt, max_tokens, **args):
    """generate from an arbitrary model, return in a canonical format"""

    prompt_idxs = prompt2tokens(prompt) 

    if model not in OPENAI_API:
        assert(type(model) != str)
        outputs = hf_gen_wrapper(model=model, idxs=prompt_idxs, max_tokens=max_tokens, **args)  
    else:
        # max_tokens is hardcoded to be 0 as this function is only meant for
        # querying probabilities of fixed strings
        outputs = gpt3_wrapper(model=model, idxs=prompt_idxs, max_tokens=max_tokens, **args)

    return outputs
