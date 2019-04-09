import argparse
import os
import numpy as np
import tensorflow as tf
import json
import facenet
import compare
import sys

def np2embeddings(img_np, model):
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
    img_paths = facenet.get_image_paths(args.input_dir)
    n_img = len(img_paths)
    img_paths_this_batch = []
    for i, img_path in enumerate(img_paths):
        img_paths_this_batch.append(img_path)
        if i % batch_size == batch_size - 1 or i == n_img - 1:
            i_batch = int(i / batch_size)
            i_batch_str = str(i_batch).zfill(3)
            print("Processing batch {}".format(i_batch))
            imgs_np = compare.load_and_align_data(img_paths_this_batch, 160, 32, 0.8)
            emb = np2embeddings(imgs_np, args.model)
            img_paths_output = os.path.join(output_dir, 'img_path_{}.txt'.format(i_batch_str))
            embeddings_output = os.path.join(output_dir, 'embeddings_{}'.format(i_batch_str))
            with open(img_paths_output, 'w') as fout:
                fout.write(json.dumps(img_paths_this_batch, indent=4))
            np.save(embeddings_output, emb)
            img_paths_this_batch = []
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