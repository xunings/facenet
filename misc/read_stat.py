import argparse
import h5py

def main(args):
    # Ref: https://stackoverflow.com/questions/28170623/how-to-read-hdf5-files-in-python
    f = h5py.File(args.filename, 'r')
    keys = list(f.keys())
    print('keys:')
    print(keys)
    if args.key != '':
        print(list(f[args.key]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, required=True)
    parser.add_argument('--key', type=str, default='')
    args = parser.parse_args()
    main(args)