import argparse
import json
import sys
import os
import facenet
import scipy
import numpy as np
import tensorflow as tf
import align.detect_face

def np2embeddings(img_np, model):
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
        gpu_options.allow_growth = True
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

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
    imgs_this_batch = []
    img_paths_this_batch = []
    for i, img_path in enumerate(img_paths):
        img = scipy.misc.imread(os.path.expanduser(img_path), mode='RGB')
        img = facenet.prewhiten(img)
        imgs_this_batch.append(img)
        img_paths_this_batch.append(img_path)
        if i%batch_size == batch_size -1 or i == n_img-1:
            i_batch = int(i/batch_size)
            i_batch_str = str(i_batch).zfill(3)
            print("Processing batch {}".format(i_batch))
            img_np = np.stack(imgs_this_batch)

            emb = np2embeddings(img_np, args.model)
            img_paths_output = os.path.join(output_dir, 'img_path{}.txt'.format(i_batch_str))
            embeddings_output = os.path.join(output_dir, 'embeddings{}'.format(i_batch_str))
            with open(img_paths_output, 'w') as fout:
                fout.write(json.dumps(img_paths_this_batch, indent=4))
            np.save(embeddings_output, emb)
            imgs_this_batch = []
            img_paths_this_batch = []
    print('Done.')

'''
    img_np = np.stack(imgs_this_batch)

    emb = np2embeddings(img_np, args.model)

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    img_paths_output = os.path.join(output_dir, 'img_path.txt')
    embeddings_output = os.path.join(output_dir, 'embeddings')

    with open(img_paths_output, 'w') as fout:
        fout.write(json.dumps(img_paths, indent=4))

    np.save(embeddings_output, emb)
'''




def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str,
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('input_dir', type=str, help='Directory with aligned face thumbnails.')
    parser.add_argument('output_dir', type=str, help='Output numpy array for embeddings')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
