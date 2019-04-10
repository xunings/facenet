import argparse
import os
import numpy as np
import tensorflow as tf
import json
import facenet
import compare
import sys

def np2embeddings(img_np, model):
    print('Calculating embedding')
    with tf.Graph().as_default():
        tfconfig = tf.ConfigProto()
        tfconfig.gpu_options.allow_growth = True
        sess = tf.Session(config=tfconfig)
        with sess.as_default():
            facenet.load_model(model)
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            feed_dict = { images_placeholder: img_np, phase_train_placeholder:False }
            emb = sess.run(embeddings, feed_dict=feed_dict)
    return emb


def main(args):
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    batch_size = 1000
    img_names = os.listdir(args.input_dir)
    img_names.sort()
    n_img = len(img_names)
    img_names_this_batch = []
    for i, img_name in enumerate(img_names):
        img_names_this_batch.append(img_name)
        if i % batch_size == batch_size - 1 or i == n_img - 1:
            i_batch = int(i / batch_size)
            i_batch_str = str(i_batch).zfill(3)
            print("Processing batch {}".format(i_batch))
            img_paths_this_batch = [os.path.join(args.input_dir, img_name_this_batch) \
                                    for img_name_this_batch in img_names_this_batch ]
            # The corresponding image path is removed if face alignment fails.
            imgs_np = compare.load_and_align_data(img_paths_this_batch, 160, 32, 0.8)
            img_names_this_batch = [os.path.basename(img_path_this_batch) \
                                      for img_path_this_batch in img_paths_this_batch ]
            emb = np2embeddings(imgs_np, args.model)
            img_names_output = os.path.join(output_dir, 'img_names_{}.txt'.format(i_batch_str))
            embeddings_output = os.path.join(output_dir, 'embeddings_{}'.format(i_batch_str))
            with open(img_names_output, 'w') as fout:
                path_names = {'path': args.input_dir, 'names': img_names_this_batch}
                fout.write(json.dumps(path_names, indent=4))
            np.save(embeddings_output, emb)
            img_names_this_batch = []
    print('Done')

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str,
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('input_dir', type=str, help='Directory with raw images')
    parser.add_argument('output_dir', type=str, help='Output numpy array for embeddings')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))