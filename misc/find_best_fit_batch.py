import argparse
import sys
import compare
import calc_embeddings
import numpy as np
import os
import json
import scipy
import facenet

def calc_similarity(emb1, emb2):
    dot = emb1 @ emb2.reshape((-1,1))
    norm = np.linalg.norm(emb1) * np.linalg.norm(emb2)
    similarity = dot/norm
    return similarity

def main(args):
    per_batch_output = False
    skip_face_align = False
    if skip_face_align:
        image = []
        img = scipy.misc.imread(os.path.expanduser(args.img_path), mode='RGB')
        img = scipy.misc.imresize(img, (160, 160), interp='bilinear')
        prewhitened = facenet.prewhiten(img)
        image.append(prewhitened)
    else:
        image = compare.load_and_align_data([args.img_path], 160, 32, 0.8)
    emb_test = calc_embeddings.np2embeddings(image, args.model)
    best_similarity = 0
    best_fit_img = ''
    n_batch_limit = 1000
    for i_batch in range(n_batch_limit):
        i_batch_str = str(i_batch).zfill(3)
        database_embeddings_path = os.path.join(args.img_database_dir, 'embeddings{}.npy'.format(i_batch_str))
        database_mapping_path = os.path.join(args.img_database_dir, 'img_path{}.txt'.format(i_batch_str))
        if not ( os.path.isfile(database_embeddings_path) \
                 and os.path.isfile(database_mapping_path) ):
            break
        embeddings_database = np.load(database_embeddings_path)
        with open(database_mapping_path) as fin:
            database_img_paths = json.load(fin)
        scores_wo_norm = embeddings_database @ emb_test.reshape((-1,1))
        best_idx_this_batch = np.argmax(scores_wo_norm)
        best_similarity_this_batch = calc_similarity(embeddings_database[best_idx_this_batch,:], emb_test)
        if per_batch_output:
            print("Processed batch {}".format(i_batch))
            print("Best fit image of this batch: {}".format(database_img_paths[best_idx_this_batch]))
            print("Similarity: {}".format(best_similarity_this_batch))
        if best_similarity_this_batch > best_similarity:
            best_similarity = best_similarity_this_batch
            best_fit_img = database_img_paths[best_idx_this_batch]

    print("Best fit image: {}".format(best_fit_img) )
    print("Similarity: {}".format(best_similarity) )


'''
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
'''

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str,
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('img_path', type=str, help='Image to test.')
    parser.add_argument('img_database_dir', type=str, help='Directory with embeddings and image paths.')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))