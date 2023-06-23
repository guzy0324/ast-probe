import glob
import argparse

import numpy as np
import torch

from collections import defaultdict

def print_table(table, sep="\t"):
    for lang in table:
        print(lang)
        header = sorted(set(k for model in table[lang] for k in table[lang][model].keys()))
        print("model", end=sep)
        print(sep.join(map(str, header)))
        for model in table[lang]:
            print(model, end=sep)
            print(sep.join(f"{table[lang][model][k]:.4f}" if k in table[lang][model] else "-" for k in header))
        print()

def main():
    parser = argparse.ArgumentParser(description='Script for computing the F norm')
    parser.add_argument('--run_dir', default='./runs', help='Path of the run logs')
    args = parser.parse_args()

    fro_table = defaultdict(lambda: defaultdict(lambda: defaultdict(float))) # fro_table[lang][model][layer]=fro
    inf_table = defaultdict(lambda: defaultdict(lambda: defaultdict(float))) # inf_table[lang][model][layer]=inf
    inf_normalized_table = defaultdict(lambda: defaultdict(lambda: defaultdict(float))) # inf_normalized_table[lang][model][layer]=inf_normalized
    fro_rq4_table = defaultdict(lambda: defaultdict(lambda: defaultdict(float))) # fro_rq4_table[lang][model][rank]=fro
    inf_rq4_table = defaultdict(lambda: defaultdict(lambda: defaultdict(float))) # inf_rq4_table[lang][model][rank]=inf
    inf_normalized_rq4_table = defaultdict(lambda: defaultdict(lambda: defaultdict(float))) # inf_normalized_rq4_table[lang][model][rank]=inf_normalized

    for file in sorted(glob.glob(args.run_dir + "/*/pytorch_model.bin")):
        checkpoint = torch.load(file, map_location=torch.device('cpu'))
        proj = checkpoint['proj'].cpu().detach().numpy()
        mult = np.matmul(proj.T, proj)
       # print(np.round(mult, 3))
        print(file)
        print('Fro norm', np.linalg.norm(mult - np.eye(mult.shape[0]), 'fro'))
        print('Inf norm', np.linalg.norm(mult - np.eye(mult.shape[0]), np.inf))
        print('Inf norm normalized', np.linalg.norm(mult - np.eye(mult.shape[0]), np.inf)/mult.shape[0])
        print(np.linalg.norm(mult - np.eye(mult.shape[0]), 'fro') < 0.05)
        print('vectors c', checkpoint['vectors_c'].shape)
        print('vectors u', checkpoint['vectors_u'].shape)
        name = file.rsplit("/", 2)[1]
        if name.endswith("rq4"):
            model, lang, layer, rank, rq4 = name.rsplit("_", 4)
            fro_rq4_table[lang][model][int(rank)] = np.linalg.norm(mult - np.eye(mult.shape[0]))
            inf_rq4_table[lang][model][int(rank)] = np.linalg.norm(mult - np.eye(mult.shape[0]), np.inf)
            inf_normalized_rq4_table[lang][model][int(rank)] = np.linalg.norm(mult - np.eye(mult.shape[0]), np.inf)/mult.shape[0]
        else:
            model, lang, layer, rank = name.rsplit("_", 3)
            fro_table[lang][model][int(layer)] = np.linalg.norm(mult - np.eye(mult.shape[0]))
            inf_table[lang][model][int(layer)] = np.linalg.norm(mult - np.eye(mult.shape[0]), np.inf)
            inf_normalized_table[lang][model][int(layer)] = np.linalg.norm(mult - np.eye(mult.shape[0]), np.inf)/mult.shape[0]

    print("Fro norm")
    print_table(fro_table)
    print("Inf norm")
    print_table(inf_table)
    print("Inf norm normalized")
    print_table(inf_normalized_table)
    print("Fro norm RQ4")
    print_table(fro_rq4_table)
    print("Inf norm RQ4")
    print_table(inf_rq4_table)
    print("Inf norm normalized RQ4")
    print_table(inf_normalized_rq4_table)


if __name__ == '__main__':
    main()
