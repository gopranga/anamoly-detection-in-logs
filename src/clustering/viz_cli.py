import argparse
import logging

from .visualize import main_visualize


def main() -> None:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
    ap = argparse.ArgumentParser(description='Clustering visualization CLI')
    ap.add_argument('--config', default='src/configs/project.yaml')
    ap.add_argument('--sample-size', type=int, default=20000)
    args = ap.parse_args()

    main_visualize(args.config, sample_size=args.sample_size)


if __name__ == '__main__':
    main()


