from argparse import ArgumentParser
import os


def main(args):
    langs = args.langs
    models = args.models
    folders = args.folders
    model_types = args.model_types
    device = args.device
    num_devices = args.num_devices

    for lang in langs:
        for model, folder, model_type in zip(models, folders, model_types):
            layers = [get_layer_model(lang, folder)]
            for layer in layers:
                for i, rank in enumerate([8, 16, 32, 64, 128, 256, 512]):
                    if i % num_devices != device:
                        continue
                    run_name = '_'.join([folder, lang, str(layer), str(rank), 'rq4'])
                    os.system(f"CUDA_VISIBLE_DEVICES={device} python src/main.py --do_train --run_name {run_name} "
                              f"--pretrained_model_name_or_path {model} "
                              f"--model_type {model_type} --lang {lang} "
                              f"--layer {layer} --rank {rank}")


def get_layer_model(lang, folder):
    if lang == 'python':
        if folder == 'codebert':
            return 5
        if folder == 'graphcodebert':
            return 4
        if folder == 'codet5':
            return 7
        if folder == 'codeberta':
            return 4
        if folder == 'roberta':
            return 5
        if folder == "unixcoder":
            return 4
        if folder == "unixcoder-unimodal":
            return 5
        if folder == "unixcoder-nine":
            return 4
    if lang == 'go':
        if folder == 'codebert':
            return 5
        if folder == 'graphcodebert':
            return 5
        if folder == 'codet5':
            return 8
        if folder == 'codeberta':
            return 4
        if folder == 'roberta':
            return 5
    if lang == 'javascript':
        if folder == 'codebert':
            return 5
        if folder == 'graphcodebert':
            return 4
        if folder == 'codet5':
            return 6
        if folder == 'codeberta':
            return 5
        if folder == 'roberta':
            return 8


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--langs", nargs="+")
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--folders", nargs="+")
    parser.add_argument("--model_types", nargs="+")
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--num_devices", type=int, required=True)
    args = parser.parse_args()
    main(args)
