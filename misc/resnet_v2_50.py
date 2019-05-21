import resnet_model

# Modified from tensorflow/official/resnet/imagenet_main.py
def inference(images, keep_probability=1.0, phase_train=True,
              bottleneck_layer_size=512, weight_decay=0.0, reuse=None):

    # The output of the final dense layer is
    # seen as the embedding vector here, instead of the class scores.
    model = resnet_model.Model(resnet_size=50,
            bottleneck=True,
            num_classes=bottleneck_layer_size,
            num_filters=64,
            kernel_size=7,
            conv_stride=2,
            first_pool_size=3,
            first_pool_stride=2,
            second_pool_size=7,
            second_pool_stride=1,
            block_sizes=[3, 4, 6, 3],
            block_strides=[1, 2, 2, 2],
            final_size=2048,
            version=resnet_model.DEFAULT_VERSION,
            data_format=None,
            weight_decay=weight_decay)

    prelogits = model(images, training=phase_train)
    return prelogits, None
