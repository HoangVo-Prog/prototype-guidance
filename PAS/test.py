import os.path as op

from datasets import build_dataloader
from model import build_model
from processor.processor import do_inference
from utils.checkpoint import Checkpointer
from utils.env import load_dotenv_if_present
from utils.iotools import load_train_configs
from utils.logger import setup_logger
from utils.options import get_args


def _maybe_load_saved_run_config(args):
    if args.config_file:
        return args
    if not args.output_dir:
        return args
    candidates = [
        op.join(args.output_dir, 'resolved_config.yaml'),
        op.join(args.output_dir, 'configs.yaml'),
    ]
    for candidate in candidates:
        if op.exists(candidate):
            saved_args = load_train_configs(candidate)
            for key, value in vars(saved_args).items():
                setattr(args, key, value)
            break
    return args


if __name__ == '__main__':
    load_dotenv_if_present()
    args = get_args()
    args = _maybe_load_saved_run_config(args)
    args.training = False

    logger = setup_logger('pas', save_dir=args.output_dir, if_train=args.training)
    logger.info(args)
    device = args.device

    if args.cross_domain_generalization:
        _, _, num_classes = build_dataloader(args)
        args.dataset_name = args.target_domain
        test_img_loader, test_txt_loader, _ = build_dataloader(args)
    else:
        test_img_loader, test_txt_loader, num_classes = build_dataloader(args)

    checkpoint_path = args.checkpoint or op.join(args.output_dir, 'best.pth')
    if not op.exists(checkpoint_path):
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')

    model = build_model(args, num_classes)
    checkpointer = Checkpointer(model)
    checkpointer.load(f=checkpoint_path)
    model = model.to(device)
    do_inference(model, test_img_loader, test_txt_loader, args)



