"""Evaluate an MMSeg checkpoint from a config file."""
import argparse
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', help='Path to an MMEngine/MMSeg config')
    parser.add_argument('checkpoint', help='Checkpoint to evaluate')
    parser.add_argument('--work-dir', help='Directory for evaluation outputs')
    return parser.parse_args()


def main():
    args = parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    try:
        from mmengine.config import Config
        from mmengine.runner import Runner
    except ImportError as exc:
        raise RuntimeError('test.py requires mmengine and mmsegmentation') from exc

    config = Config.fromfile(args.config)
    config.load_from = args.checkpoint
    if args.work_dir:
        config.work_dir = args.work_dir
    Runner.from_cfg(config).test()


if __name__ == '__main__':
    main()
