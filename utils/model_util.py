import os
import tensorflow as tf

def convert_ckpt_to_pb(ckpt_dir, pb_dir, epoch):

    trained_checkpoint_prefix = os.path.join(
        ckpt_dir, 'model.ckpt-' + str(epoch))

    graph = tf.Graph()
    with tf.compat.v1.Session(graph=graph) as sess:
        # Restore from checkpoint
        loader = tf.compat.v1.train.import_meta_graph(
            trained_checkpoint_prefix + '.meta')
        loader.restore(sess, trained_checkpoint_prefix)

        # Export checkpoint to SavedModel
        builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(pb_dir)
        builder.add_meta_graph_and_variables(sess,
                                             [tf.saved_model.TRAINING,
                                                 tf.saved_model.SERVING],
                                             strip_default_attrs=True)
        builder.save()

        with tf.Session(graph=tf.Graph()) as sess:
            # We import the meta graph in the current default Graph
            saver = tf.train.import_meta_graph(
                ckpt_dir + "/model.ckpt-985" + '.meta', clear_devices=True)

            # We restore the weights
            saver.restore(sess, ckpt_dir)
        graph_def = sess.graph.as_graph_def()
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,  # The session is used to retrieve the weights
            graph_def,  # The graph_def is used to retrieve the nodes
            # output_node_names # The output node names are used to select the usefull nodes
        )
        with tf.gfile.GFile(pb_dir, 'wb') as f:
            f.write(output_graph_def.SerializeToString())

def convert_h5_to_pb(h5_path,pb_dir):
    from tensorflow.keras.models import load_model
    model = load_model(h5_path)
    tf.keras.backend.set_learning_phase(0)  # Ignore dropout at inference

    with tf.keras.backend.get_session() as sess:
        tf.saved_model.simple_save(
            sess,
            pb_dir,
            inputs={'input_image': model.input},
            outputs={'y_pred': model.output})
    
    # saved_model.pb: serialized model, lưu giữ toàn bộ thông tin graph của mô hình cũng như các meta-data khác như signature, inputs, outputs của model
    # variables: lưu giữ các serialized variables của graph (learned weights)



def freeze_graph(model_dir, output_node_names,version):

    # exp: freeze_graph("/Users/brown/code/transformer_gkv/weights/nttd/drive-download-20200112T112654Z-001", "loss/Mean,loss/Mean_1,final/Graph-CNN/add_1",1100)
    """Extract the sub graph defined by the output nodes and convert 
    all its variables into constant 
    Args:
        model_dir: the root folder containing the checkpoint state file (ckpt dir)
        output_node_names: a string, containing all the output node's names, 
                            comma separated
    """
    if not tf.gfile.Exists(model_dir):
        raise AssertionError(
            "Export directory doesn't exists. Please specify an export "
            "directory: %s" % model_dir)

    if not output_node_names:
        print("You need to supply the name of a node to --output_node_names.")
        return -1

    # We retrieve our checkpoint fullpath
    checkpoint = tf.train.get_checkpoint_state(model_dir)
    if (version == 0):
        input_checkpoint = checkpoint.model_checkpoint_path
    else:
        input_checkpoint = "/".join(checkpoint.model_checkpoint_path.split('/')[:-1]) + "/model.ckpt-" + str(version)

    # We precise the file fullname of our freezed graph
    absolute_model_dir = "/".join(input_checkpoint.split('/')[:-1])
    output_graph = absolute_model_dir + "/frozen_model.pb"

    # We clear devices to allow TensorFlow to control on which device it will load operations
    clear_devices = True

    # We start a session using a temporary fresh Graph
    with tf.Session(graph=tf.Graph()) as sess:
        # We import the meta graph in the current default Graph
        saver = tf.train.import_meta_graph(
            input_checkpoint + '.meta', clear_devices=clear_devices)

        # We restore the weights
        saver.restore(sess, input_checkpoint)
        graph_def = sess.graph.as_graph_def()

        for node in graph_def.node:
            if node.op == 'RefSwitch':
                node.op = 'Switch'
                for index in range(len(node.input)):
                    if 'moving_' in node.input[index]:
                        node.input[index] = node.input[index] + '/read'
            elif node.op == 'AssignSub':
                node.op = 'Sub'
                if 'use_locking' in node.attr:
                    del node.attr['use_locking']
            elif node.op == 'AssignAdd':
                node.op = 'Add'
                if 'use_locking' in node.attr:
                    del node.attr['use_locking']

        # We use a built-in TF helper to export variables to constants
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,  # The session is used to retrieve the weights
            # tf.get_default_graph().as_graph_def(), # The graph_def is used to retrieve the nodes
            graph_def,
            # The output node names are used to select the usefull nodes
            output_node_names.split(",")
        )

        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))

    return output_graph_def


def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def

    # USE: 
    # graph = load_graph(
    #     "/Users/brown/code/transformer_gkv/weights/nttd/drive-download-20191221T114654Z-001/frozen_model.pb")
    # with tf.Session(graph=graph) as sess:
    # loss, acc, out = sess.run(["prefix/loss/Mean", "prefix/loss/Mean_1", "prefix/final/Graph-CNN/add_1"],
    #                             feed_dict={"prefix/input_vertices:0": v,
    #                                         "prefix/input_adj:0": adj,
    #                                         "prefix/Placeholder:0": lb,
    #                                         "prefix/is_training:0": False}
    #                             )


    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    for node in graph_def.node:
        if node.op == 'RefSwitch':
            node.op = 'Switch'
            for index in range(len(node.input)):
                if 'moving_' in node.input[index]:
                    node.input[index] = node.input[index] + '/read'
        elif node.op == 'AssignSub':
            node.op = 'Sub'
            if 'use_locking' in node.attr:
                del node.attr['use_locking']
        elif node.op == 'AssignAdd':
            node.op = 'Add'
            if 'use_locking' in node.attr:
                del node.attr['use_locking']

    # Then, we import the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="prefix")

    for op in graph.get_operations():
        print(op.name)

    return graph