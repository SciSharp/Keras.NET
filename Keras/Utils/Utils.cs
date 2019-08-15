namespace Keras.Utils
{
    using global::Keras.Models;
    using Numpy;
    using Python.Runtime;
    using System.Linq;

    public class Util : Base
    {
        /// <summary>
        /// Replicates a model on different GPUs.    Specifically, this function implements single-machine multi-GPU data parallelism.It works in the following way:
        ///  Divide the model's input(s) into multiple sub-batches. Apply a model copy on each sub-batch.Every model copy is executed on a dedicated GPU.
        /// Concatenate the results(on CPU) into one big batch.
        ///E.g. if your batch_size is 64 and you use gpus = 2, then we will divide the input into 2 sub-batches of 32 samples, process each sub-batch on one GPU, then return the full batch of 64 processed samples.
        ///This induces quasi-linear speedup on up to 8 GPUs. This function is only available with the TensorFlow backend for the time being.
        /// </summary>
        /// <param name="model">A Keras model instance. To avoid OOM errors, this model could have been built on CPU, for instance.</param>
        /// <param name="gpus"> Integer >= 2 or list of integers, number of GPUs or list of GPU IDs on which to create model replicas.</param>
        /// <param name="cpu_merge"> Integer >= 2 or list of integers, number of GPUs or list of GPU IDs on which to create model replicas.</param>
        /// <param name="cpu_relocation">A boolean value to identify whether to create the model's weights under the scope of the CPU. If the model is not defined under any preceding device scope, you can still rescue it by activating this option.</param>
        /// <returns>A Keras Model instance which can be used just like the initial model argument, but which distributes its workload on multiple GPUs.</returns>
        public static BaseModel MultiGPUModel(BaseModel model, int[] gpus, bool cpu_merge = true, bool cpu_relocation = false)
        {
            BaseModel result = new BaseModel();
            result.PyInstance = (PyObject)Instance.keras.utils.multi_gpu_model(model: model.PyInstance, gpus: gpus.ToList(), cpu_merge: cpu_merge, cpu_relocation: cpu_relocation);
            return result;
        }

        /// <summary>
        /// Replicates a model on different GPUs.    Specifically, this function implements single-machine multi-GPU data parallelism.It works in the following way:
        ///  Divide the model's input(s) into multiple sub-batches. Apply a model copy on each sub-batch.Every model copy is executed on a dedicated GPU.
        /// Concatenate the results(on CPU) into one big batch.
        ///E.g. if your batch_size is 64 and you use gpus = 2, then we will divide the input into 2 sub-batches of 32 samples, process each sub-batch on one GPU, then return the full batch of 64 processed samples.
        ///This induces quasi-linear speedup on up to 8 GPUs. This function is only available with the TensorFlow backend for the time being.
        /// </summary>
        /// <param name="model">A Keras model instance. To avoid OOM errors, this model could have been built on CPU, for instance.</param>
        /// <param name="gpus"> Integer >= 2 or list of integers, number of GPUs or list of GPU IDs on which to create model replicas.</param>
        /// <param name="cpu_merge"> Integer >= 2 or list of integers, number of GPUs or list of GPU IDs on which to create model replicas.</param>
        /// <param name="cpu_relocation">A boolean value to identify whether to create the model's weights under the scope of the CPU. If the model is not defined under any preceding device scope, you can still rescue it by activating this option.</param>
        /// <returns>A Keras Model instance which can be used just like the initial model argument, but which distributes its workload on multiple GPUs.</returns>
        public static BaseModel MultiGPUModel(BaseModel model, int gpus, bool cpu_merge = true, bool cpu_relocation = false)
        {
            BaseModel result = new BaseModel();
            result.PyInstance = (PyObject)Instance.keras.utils.multi_gpu_model(model: model.PyInstance, gpus: gpus, cpu_merge: cpu_merge, cpu_relocation: cpu_relocation);
            return result;
        }

        /// <summary>
        /// Converts a class vector (integers) to binary class matrix. E.g. for use with categorical_crossentropy.
        /// </summary>
        /// <param name="y">class vector to be converted into a matrix (integers from 0 to num_classes).</param>
        /// <param name="num_classes">total number of classes.</param>
        /// <param name="dtype">The data type expected by the input, as a string (float32, float64, int32...)</param>
        /// <returns>A binary matrix representation of the input. The classes axis is placed last.</returns>
        public static NDarray ToCategorical(NDarray y, int? num_classes = null, string dtype = "float32")
        {
            return new NDarray((PyObject)Instance.keras.utils.to_categorical(y: y.PyObject, num_classes: num_classes, dtype: dtype));
        }

        /// <summary>
        /// Normalizes a Numpy array.
        /// </summary>
        /// <param name="y">Numpy array to normalize.</param>
        /// <param name="axis">axis along which to normalize.</param>
        /// <param name="order">Normalization order (e.g. 2 for L2 norm).</param>
        /// <returns></returns>
        public static NDarray Normalize(NDarray y, int axis = -1, int order = 2)
        {
            return new NDarray((PyObject)Instance.keras.utils.normalize(y: y.PyObject, axis: axis, order: order));
        }

        /// <summary>
        /// Converts a Keras model to dot format and save to a file.
        /// </summary>
        /// <param name="model"> A Keras model instance</param>
        /// <param name="to_file"> File name of the plot image.</param>
        /// <param name="show_shapes"> whether to display shape information.</param>
        /// <param name="show_layer_names"> whether to display layer names.</param>
        /// <param name="rankdir: rankdir argument passed to PyDot, a string specifying the format of the plot"> 'TB' creates a vertical plot; 'LR' creates a horizontal plot.</param>
        /// <param name="expand_nested"> whether to expand nested models into clusters.</param>
        /// <param name="dpi"> dot DPI.</param>

        public static void PlotModel(BaseModel model, string to_file = "model.png", bool show_shapes = false, bool show_layer_names = true,
            string rankdir = "TB", bool expand_nested = false, int dpi = 96)
        {
            Instance.keras.utils.plot_model(model: model.PyInstance, to_file: to_file, show_shapes: show_shapes, show_layer_names: show_layer_names,
                rankdir: rankdir, expand_nested: expand_nested, dpi: dpi);
        }
        
        /// <summary>
        /// Set TensorFlow Backend Session configuration parameters
        /// </summary>
        /// <param name="intra_op_parallelism_threads">The execution of an individual op (for some op types) can be parallelized on a pool of intra_op_parallelism_threads. 0 means the system picks an appropriate number.</param>
        /// <param name="inter_op_parallelism_threads">Nodes that perform blocking operations are enqueued on a pool of inter_op_parallelism_threads available in each process. 0 means the system picks an appropriate number.</param>
        /// <param name="allow_soft_placement">Whether soft placement is allowed. If allow_soft_placement is true, an op will be placed on CPU if 1. there's no GPU implementation for the OP or 2. no GPU devices are known or registered or 3. need to co-locate with reftype input(s) which are from CPU.</param>
        /// <param name="cpu_device_count">Maximum number of CPU devices of that type to use.</param>
        /// <param name="gpu_device_count">Maximum number of GPU devices of that type to use.</param>
        public static void ConfigTensorFlowBackend( int intra_op_parallelism_threads, int inter_op_parallelism_threads, bool allow_soft_placement, int cpu_device_count, int gpu_device_count )
        {
            dynamic tf = Py.Import( "tensorflow" );
            dynamic kb = Py.Import( "keras.backend" );

            PyDict deviceCount = new PyDict();
            deviceCount["CPU"] = new PyInt( cpu_device_count );
            deviceCount["GPU"] = new PyInt( gpu_device_count );
            dynamic config = tf.ConfigProto(
                intra_op_parallelism_threads: intra_op_parallelism_threads,
                inter_op_parallelism_threads: inter_op_parallelism_threads,
                allow_soft_placement: allow_soft_placement,
                device_count: deviceCount
            );
            dynamic session = tf.Session( config: config );
            kb.set_session( session );
        }
    }
}
