import argparse
import sys
import compare
import calc_embeddings
import numpy as np
import os
import json

def distance(emb1, emb2):
    dot = emb1 @ emb2.reshape((-1,1))
    norm = np.linalg.norm(emb1) * np.linalg.norm(emb2)
    similarity = dot/norm
    return similarity

def main(args):
    image = compare.load_and_align_data([args.img_path], 160, 32, 0.8)
    emb_test = calc_embeddings.np2embeddings(image, args.model)
    database_embeddings_path = os.path.join(args.img_database_dir, 'embeddings.npy')
    database_mapping_path = os.path.join(args.img_database_dir, 'img_path.txt')
    embeddings_database = np.load(database_embeddings_path)
    with open(database_mapping_path) as fin:
        database_img_paths = json.load(fin)
    scores = embeddings_database @ emb_test.reshape((-1,1))
    idx = np.argmax(scores)
    dist = distance(embeddings_database[idx,:], emb_test)
    print("Best fit celebrity: {}".format(database_img_paths[idx]) )
    print("Similarity: {}".format(dist) )

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str,
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('img_path', type=str, help='Image to test.')
    parser.add_argument('img_database_dir', type=str, help='Directory with embeddings and image paths.')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))