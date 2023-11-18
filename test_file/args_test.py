import argparse
parser = argparse.ArgumentParser('train.py')
parser.add_argument('config', help='configuration YAML file.')
print(parser.parse_args().config)