import random
import numpy as np
import torch

# original lib
import common as com
from networks.models import Models

########################################################################
# load parameter.yaml (path may be overridden by --config=...)
########################################################################
config_path = com.get_config_path()
print(f"[train.py] loading config from: {config_path}")
param = com.yaml_load(path=config_path)
########################################################################

# Map --model_type values to registered model names.
MODEL_TYPE_TO_MODEL = {
    "ae": "DCASE2023T2-AE",
    "domain_cae": "DCASE2023T2-Domain-CAE",
}

def main():
    parser = com.get_argparse()
    # read parameters from yaml
    flat_param = com.param_to_args_list(params=param)
    args = parser.parse_args(args=flat_param)
    # read parameters from command line
    args = parser.parse_args(namespace=args)

    # If the user specified --model_type, route it to the registered model name
    # so the CLI flag from the thesis spec works end-to-end.
    if args.model_type in MODEL_TYPE_TO_MODEL:
        mapped = MODEL_TYPE_TO_MODEL[args.model_type]
        if args.model != mapped:
            print(f"[train.py] --model_type={args.model_type} overrides --model -> {mapped}")
            args.model = mapped

    print(args)

    if args.train_only and args.test_only:
        raise ValueError("--train_only and --test_only cannot be used together.")
    elif args.train_only:
        train = True
        test = False
    elif args.test_only:
        train = False
        test = True
    else:
        train = True
        test = True
    
    args.cuda = args.use_cuda and torch.cuda.is_available()

    # Python random
    random.seed(args.seed)
    # Numpy
    np.random.seed(args.seed)
    # Pytorch
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True

    net = Models(args.model).net(
        args=args,
        train=train,
        test=test
    )


    print(args.model)

    print("============== BEGIN TRAIN ==============")
    if train:
        for epoch in range(1, args.epochs + 2):
            net.train(epoch)
    print("============ END OF TRAIN ============")
    
    if test:
        net.test()

if __name__ == "__main__":
    main()