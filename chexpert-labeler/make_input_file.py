from pathlib import Path
import json
import csv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str)
args = parser.parse_args()

INPUT_PATH = Path(args.input_path)
GT_OUTPUT_PATH = 'GT_' + INPUT_PATH.stem + '.csv'
GEN_OUTPUT_PATH = 'GEN_' + INPUT_PATH.stem + '.csv'

with open(INPUT_PATH, 'r') as f:
    data = json.load(f)

f_gt = open(GT_OUTPUT_PATH, 'w')
wr_gt = csv.writer(f_gt)

f_gen = open(GEN_OUTPUT_PATH, 'w')
wr_gen = csv.writer(f_gen)

for study_id in data.keys():
    wr_gt.writerow(
        [data[study_id]['GT_text']]
    )
    wr_gen.writerow(
        [data[study_id]['gen_text']]
    )

f_gt.close()
f_gen.close()
