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
            if model == 'huggingface/CodeBERTa-small-v1':
                layers = list(range(1, 7))
            elif folder == 'codebert0':
                layers = [0]
            else:
                layers = list(range(1, 13))
            for layer in layers:
                if layer % num_devices != device:
                    continue
                run_name = '_'.join([folder, lang, str(layer), '128'])
                os.system(f"CUDA_VISIBLE_DEVICES={device} python src/main.py --do_train --run_name {run_name} "
                          f"--pretrained_model_name_or_path {model} "
                          f"--model_type {model_type} --lang {lang} "
                          f"--layer {layer} --rank 128")


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
