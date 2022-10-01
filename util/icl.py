import pdb

import sys
import os
import json
import csv
import random

def identity(o):
    return o

def sample(path, n, filetype='ending'):
    """samples N data points from a 1 data point per line file"""
    if filetype == 'ending':
        filetype = os.path.splitext(path)[1][1:]

    data = []
    if filetype == 'jsonl':
        with open(path) as lines:
            for line in lines:
                data.append(json.loads(line))
    elif filetype == 'csv':
        with open(path) as demo_file:
            reader = csv.DictReader(demo_file)
            for row in reader:
                data.append(row)

    return random.sample(data, n)


def iter(path, filetype='ending'):
    if filetype == 'ending':
        filetype = os.path.splitext(path)[1][1:]

    data = []
    if filetype == 'jsonl':
        with open(path) as lines:
            for line in lines:
                data.append(json.loads(line))
    elif filetype == 'json':
        with open(path) as json_file:
            data.extend(json.load(json_file))
    elif filetype == 'csv':
        with open(path) as demo_file:
            reader = csv.DictReader(demo_file)
            for row in reader:
                data.append(row)

    return data

def basic_prompt(demos, inference_point=None, input_key="input", output_key="output", input_prefix='input: ', output_prefix='output: ',
                 input_suffix='\n', output_suffix='\n\n', overall_prefix='', input_proc=identity, output_proc=identity):

    prompt = overall_prefix
    for demo in demos:
        input_value = input_proc(demo[input_key])
        output_value = output_proc(demo[output_key])
        prompt += f'{input_prefix}{input_value}{input_suffix}{output_prefix}{output_value}{output_suffix}'

    if inference_point is not None:
        input_value = input_proc(inference_point[input_key])
        prompt += f'{input_prefix}{input_value}{input_suffix}{output_prefix}'.rstrip()

    return prompt

def inference_demo(inference_point, input_key="input", input_prefix='input: ', output_prefix='output: ',
                 input_suffix='\n', input_proc=identity):
    input_value = input_proc(inference_point[input_key])
    prompt = f'{input_prefix}{input_value}{input_suffix}{output_prefix}'.rstrip()
    return prompt

