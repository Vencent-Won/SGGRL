import argparse
import pandas as pd
from tqdm import tqdm

from utils.dataset import split

def main():
    parser = argparse.ArgumentParser(description='Build a corpus file')

    parser.add_argument('--in_path', '-i', type=str, default='./data/clintox.csv', help='input file')
    parser.add_argument('--out_path', '-o', type=str, default='./data/clintox.txt', help='output file')
    args = parser.parse_args()

    smiles = pd.read_csv(args.in_path)['smiles'].values
    with open(args.out_path, 'w') as f:
        for sm in tqdm(smiles):
            f.write(split(sm)+'\n')
    print('Built a corpus file!')

if __name__=='__main__':
    main()



