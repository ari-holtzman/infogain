import sys
import json

with open(sys.argv[1]) as lines, open(sys.argv[2], 'w') as out:
    for line in lines:
        datum = json.loads(line)
        out.write(f'{datum[sys.argv[3]]["string"]}\n')
