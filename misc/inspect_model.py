import tensorflow as tf
import sys
import json

def usage(argv):
    print("{} <path-to-pb-model>".format(argv[0]))

if __name__ == "__main__":
    try:
        path = sys.argv[1]
    except IndexError:
        usage(sys.argv)
        sys.exit(1)

    # Ref: https://stackoverflow.com/questions/43517959/given-a-tensor-flow-model-graph-how-to-find-the-input-node-and-output-node-name
    gf = tf.GraphDef()
    with open(path, 'rb') as fin:
        gf.ParseFromString(fin.read())

    nodes = [n.name + '=>' + n.op for n in gf.node]
    nodes = json.dumps(nodes, indent=4)

    nodes_io = [n.name + '=>' + n.op for n in gf.node if n.op in ('Placeholder') or n.name in ('embeddings')]
    nodes_io = json.dumps(nodes_io, indent=4)

    print("All nodes:")
    print(nodes)

    print("IO nodes:")
    print(nodes_io)