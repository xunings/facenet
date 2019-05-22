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

    # List all groups
    # print("Keys: %s" % f.keys())
    #a_group_key = list(f.keys())[0]

    # Get the data
    #data = list(f[a_group_key])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, required=True)
    parser.add_argument('--key', type=str, default='')
    args = parser.parse_args()
    main(args)