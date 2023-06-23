from argparse import ArgumentParser
import os

def main(args):
    lang = args.lang
    model=args.model
    folder=args.folder
    model_type=args.model_type
    layer = args.layer
    rank = args.rank
    device = args.device

    model_checkpoint=f"{folder}_{lang}_{layer}_{rank}"
    if not os.path.exists(f"runs/{model_checkpoint}"):
        model_checkpoint=f"{model_checkpoint}_rq4"

    os.system(f"CUDA_VISIBLE_DEVICES={device} python src/main.py --do_visualization --run_name vis_{model_checkpoint} "
              f"--pretrained_model_name_or_path {model} "
              f"--model_type {model_type} --lang {lang} "
              f"--layer {layer} --rank {rank} "
              f"--model_checkpoint runs/{model_checkpoint}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--lang", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--folder", required=True)
    parser.add_argument("--model_type", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--device", type=int, required=True)
    args = parser.parse_args()
    main(args)