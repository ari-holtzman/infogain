import os
import glob
import json
import copy
import itertools
import random

id2cl = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

cl2id = {
    'N/A': 83, 'person': 1, 'bicycle': 2, 'car': 3, 'motorcycle': 4,
    'airplane': 5, 'bus': 6, 'train': 7, 'truck': 8, 'boat': 9,
    'traffic light': 10, 'fire hydrant': 11, 'stop sign': 13,
    'parking meter': 14, 'bench': 15, 'bird': 16, 'cat': 17, 'dog': 18,
    'horse': 19, 'sheep': 20, 'cow': 21, 'elephant': 22, 'bear': 23, 'zebra': 24,
    'giraffe': 25, 'backpack': 27, 'umbrella': 28, 'handbag': 31, 'tie': 32,
    'suitcase': 33, 'frisbee': 34, 'skis': 35, 'snowboard': 36, 'sports ball': 37,
    'kite': 38, 'baseball bat': 39, 'baseball glove': 40, 'skateboard': 41,
    'surfboard': 42, 'tennis racket': 43, 'bottle': 44, 'wine glass': 46,
    'cup': 47, 'fork': 48, 'knife': 49, 'spoon': 50, 'bowl': 51, 'banana': 52,
    'apple': 53, 'sandwich': 54, 'orange': 55, 'broccoli': 56, 'carrot': 57,
    'hot dog': 58, 'pizza': 59, 'donut': 60, 'cake': 61, 'chair': 62, 'couch': 63,
    'potted plant': 64, 'bed': 65, 'dining table': 67, 'toilet': 70, 'tv': 72, 'laptop': 73, 'mouse': 74, 'remote': 75, 'keyboard': 76, 'cell phone': 77, 'microwave': 78, 'oven': 79, 'toaster': 80, 'sink': 81, 'refrigerator': 82, 'book': 84, 'clock': 85, 'vase': 86, 'scissors': 87, 'teddy bear': 88, 'hair drier': 89, 'toothbrush': 90}

def load_lexicon(name):
    lexicon = [] 
    with open(os.path.join('data_gen/lexica/', name + '.txt')) as lines:
        for line in lines:
            lexicon.append(line.strip())
    return lexicon

def load_lexica():
    lexica = {}
    for path in glob.glob('data_gen/lexica/*.txt'):
        name = os.path.basename(path)[:-len('.txt')]
        lexica[name] = []
        with open(path) as items:
            for item in items:
                lexica[name].append(item.strip())
    return lexica

def load_templates(path):
    templates = []
    with open(path) as template_jsons:
        for i, template_json in enumerate(template_jsons):
            template = json.loads(template_json)
            templates.append(template)
    return templates

def proc_term(term):
  term = ' '.join(term.split()[1:])
  term = term[len('computer')+1:] if term.startswith('computer') else term
  term = term[len('pair of')+1:] if term.startswith('pair of') else term
  return term.strip()

def instantiate_templates(templates, lexica, each=None):
    templates = copy.deepcopy(templates)
    prompts = []
    for idx, template in enumerate(templates):
        options = [ lexica[category] for category in template['categories'] ]
        cur_prompts = []
        for instances in itertools.product(*options):
            try:
                p = {
                        'prompt' : template['template'].format(*instances),
                        'template' : template,
                        'instances' : instances,
                        'objects' : [ proc_term(instance) for instance in instances if instance in id2cl ]
                    }
            except IndexError as e:
                print(template)
                print(instances)
                raise e
            cur_prompts.append(p)
        if each is not None:
            cur_prompts = random.sample(cur_prompts, k=each)
        prompts.extend(cur_prompts)
    return prompts
