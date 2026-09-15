"""Train an MMSeg model from a config file."""
import argparse
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', help='Path to an MMEngine/MMSeg config')
    parser.add_argument('--work-dir', help='Directory for logs and checkpoints')
    parser.add_argument('--resume', nargs='?', const=True, default=False,
                        help='Resume latest checkpoint or the supplied checkpoint path')
    return parser.parse_args()


def main():
    args = parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    try:
        from mmengine.config import Config
        from mmengine.runner import Runner
    except ImportError as exc:
        raise RuntimeError('train.py requires mmengine and mmsegmentation') from exc

    config = Config.fromfile(args.config)
    if args.work_dir:
        config.work_dir = args.work_dir
    if args.resume:
        config.resume = True
        if isinstance(args.resume, str):
            config.load_from = args.resume
    Runner.from_cfg(config).train()


if __name__ == '__main__':
    main()
