import argparse
import logging

from .kmeans_runner import run_kmeans_pipeline


def main() -> None:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

    ap = argparse.ArgumentParser(description='KMeans/MiniBatchKMeans clustering pipeline')
    ap.add_argument('--config', default='src/configs/project.yaml', help='Path to project config YAML')
    args = ap.parse_args()

    run_kmeans_pipeline(args.config)


if __name__ == '__main__':
    main()


