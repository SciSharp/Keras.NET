using Keras.Callbacks;
using Keras.Layers;
using Keras.Utils;
using Numpy;
using Python.Runtime;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using Tensorflow.Train;
using static System.Net.WebRequestMethods;

namespace Keras.Models
{
    public class BaseModel : Base
    {
        /// <summary>
        ///Configures the model for training.
        /// </summary>
        /// <param name="optimizer"> String (name of optimizer) or optimizer instance. See optimizers.</param>
        /// <param name="loss"> List of Strings (name of objective function) or objective function. See losses. If the model has multiple outputs, you can use a different loss on each output by passing a dictionary or a list of losses. The loss value that will be minimized by the model will then be the sum of all individual losses.</param>
        /// <param name="metrics"> List of metrics to be evaluated by the model during training and testing. Typically you will use metrics=['accuracy']. To specify different metrics for different outputs of a multi-output model, you could also pass a dictionary, such as metrics={'output_a': 'accuracy'}.</param>
        /// <param name="loss_weights"> Optional list or dictionary specifying scalar coefficients (Python floats) to weight the loss contributions of different model outputs. The loss value that will be minimized by the model will then be the weighted sum of all individual losses, weighted by the loss_weightscoefficients. If a list, it is expected to have a 1:1 mapping to the model's outputs. If a tensor, it is expected to map output names (strings) to scalar coefficients.</param>
        /// <param name="weighted_metrics"> List of metrics to be evaluated and weighted by sample_weight or class_weight during training and testing.</param>
        /// <param name="run_eagerly"> Bool. Defaults to False. If True, this Model's logic will not be wrapped in a tf.function. Recommended to leave this as None unless your Model cannot be run inside a tf.function. run_eagerly=True is not supported when using.</param>
        /// <param name="steps_per_execution"> Int. Defaults to 1. The number of batches to run during each tf.function call. Running multiple batches inside a single tf.function call can greatly improve performance on TPUs or small models with a large Python overhead. At most, one full epoch will be run each execution</param>
        /// <param name="jit_compile"> If True, compile the model training step with XLA. XLA is an optimizing compiler for machine learning. jit_compile is not enabled for by default. Note that jit_compile=True may not necessarily work for all models.</param>
        public void Compile(StringOrInstance optimizer, string loss, string[] metrics = null, float[] loss_weights = null,
                       string[] weighted_metrics = null, bool run_eagerly = false, int steps_per_execution = 1, bool jit_compile = false)
        {
            var args = new Dictionary<string, object>();
            args["optimizer"] = optimizer;
            args["loss"] = loss;
            args["metrics"] = metrics;
            args["loss_weights"] = loss_weights;
            args["weighted_metrics"] = weighted_metrics;
            args["run_eagerly"] = run_eagerly;
            args["steps_per_execution"] = steps_per_execution;
            args["jit_compile"] = jit_compile;

            InvokeMethod("compile", args);
        }

        /// <summary>
        ///Configures the model for training.
        /// </summary>
        /// <param name="optimizer"> String (name of optimizer) or optimizer instance. See optimizers.</param>
        /// <param name="loss"> List of Strings (name of objective function) or objective function. See losses. If the model has multiple outputs, you can use a different loss on each output by passing a dictionary or a list of losses. The loss value that will be minimized by the model will then be the sum of all individual losses.</param>
        /// <param name="metrics"> List of metrics to be evaluated by the model during training and testing. Typically you will use metrics=['accuracy']. To specify different metrics for different outputs of a multi-output model, you could also pass a dictionary, such as metrics={'output_a': 'accuracy'}.</param>
        /// <param name="loss_weights"> Optional list or dictionary specifying scalar coefficients (Python floats) to weight the loss contributions of different model outputs. The loss value that will be minimized by the model will then be the weighted sum of all individual losses, weighted by the loss_weightscoefficients. If a list, it is expected to have a 1:1 mapping to the model's outputs. If a tensor, it is expected to map output names (strings) to scalar coefficients.</param>
        /// <param name="weighted_metrics"> List of metrics to be evaluated and weighted by sample_weight or class_weight during training and testing.</param>
        /// <param name="run_eagerly"> Bool. Defaults to False. If True, this Model's logic will not be wrapped in a tf.function. Recommended to leave this as None unless your Model cannot be run inside a tf.function. run_eagerly=True is not supported when using.</param>
        /// <param name="steps_per_execution"> Int. Defaults to 1. The number of batches to run during each tf.function call. Running multiple batches inside a single tf.function call can greatly improve performance on TPUs or small models with a large Python overhead. At most, one full epoch will be run each execution</param>
        /// <param name="jit_compile"> If True, compile the model training step with XLA. XLA is an optimizing compiler for machine learning. jit_compile is not enabled for by default. Note that jit_compile=True may not necessarily work for all models.</param>
        public void Compile(StringOrInstance optimizer, string[] loss, string[] metrics = null, float[] loss_weights = null,
                      string[] weighted_metrics = null, bool run_eagerly = false, int steps_per_execution = 1, bool jit_compile = false)
        {
            var args = new Dictionary<string, object>();
            args["optimizer"] = optimizer;
            args["loss"] = loss;
            args["metrics"] = metrics;
            args["loss_weights"] = loss_weights;
            args["weighted_metrics"] = weighted_metrics;
            args["run_eagerly"] = run_eagerly;
            args["steps_per_execution"] = steps_per_execution;
            args["jit_compile"] = jit_compile;

            InvokeMethod("compile", args);
        }

        /// <summary>
        /// Compute the total loss, validate it, and return it.
        /// Subclasses can optionally override this method to provide custom loss computation logic.
        /// </summary>
        /// <param name="x">Input data.</param>
        /// <param name="y">Target data.</param>
        /// <param name="y_pred">Predictions returned by the model (output of model(x))</param>
        /// <param name="sample_weight">Sample weights for weighting the loss function.</param>
        public void ComputeLoss(NDarray x, NDarray y, NDarray y_pred = null, NDarray sample_weight = null)
        {
            var args = new Dictionary<string, object>();
            args["x"] = x;
            args["y"] = y;
            args["y_pred"] = y_pred;
            args["sample_weight"] = sample_weight;

            InvokeMethod("compute_loss", args);
        }

        /// <summary>
        /// Update metric states and collect all metrics to be returned.
        /// Subclasses can optionally override this method to provide custom metric updating and collection logic.
        /// </summary>
        /// <param name="x">Input data.</param>
        /// <param name="y">Target data.</param>
        /// <param name="y_pred">Predictions returned by the model (output of model(x))</param>
        /// <param name="sample_weight">Sample weights for weighting the loss function.</param>
        public void ComputeMetrics(NDarray x, NDarray y, NDarray y_pred = null, NDarray sample_weight = null)
        {
            var args = new Dictionary<string, object>();
            args["x"] = x;
            args["y"] = y;
            args["y_pred"] = y_pred;
            args["sample_weight"] = sample_weight;

            InvokeMethod("compute_metrics", args);
        }

        /// <summary>
        /// Trains the model for a given number of epochs (iterations on a dataset).
        /// </summary>
        /// <param name="x">Numpy array of training data (if the model has a single input), or list of Numpy arrays (if the model has multiple inputs). If input layers in the model are named, you can also pass a dictionary mapping input names to Numpy arrays. x can be None (default) if feeding from framework-native tensors (e.g. TensorFlow data tensors).</param>
        /// <param name="y">Numpy array of target (label) data (if the model has a single output), or list of Numpy arrays (if the model has multiple outputs). If output layers in the model are named, you can also pass a dictionary mapping output names to Numpy arrays. y can be None (default) if feeding from framework-native tensors (e.g. TensorFlow data tensors).</param>
        /// <param name="batch_size">Integer or None. Number of samples per gradient update. If unspecified, batch_sizewill default to 32.</param>
        /// <param name="epochs">Integer. Number of epochs to train the model. An epoch is an iteration over the entire x and y data provided. Note that in conjunction with initial_epoch, epochs is to be understood as "final epoch". The model is not trained for a number of iterations given by epochs, but merely until the epoch of index epochs is reached.</param>
        /// <param name="verbose">Integer. 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.</param>
        /// <param name="callbacks">List of keras.callbacks.Callback instances. List of callbacks to apply during training and validation (if ). See callbacks.</param>
        /// <param name="validation_split">Float between 0 and 1. Fraction of the training data to be used as validation data. The model will set apart this fraction of the training data, will not train on it, and will evaluate the loss and any model metrics on this data at the end of each epoch. The validation data is selected from the last samples in the x and y data provided, before shuffling.</param>
        /// <param name="validation_data">tuple (x_val, y_val) or tuple (x_val, y_val, val_sample_weights) on which to evaluate the loss and any model metrics at the end of each epoch. The model will not be trained on this data. validation_data will override validation_split.</param>
        /// <param name="shuffle">Boolean (whether to shuffle the training data before each epoch) or str (for 'batch'). 'batch' is a special option for dealing with the limitations of HDF5 data; it shuffles in batch-sized chunks. Has no effect when steps_per_epoch is not None.</param>
        /// <param name="class_weight">Optional dictionary mapping class indices (integers) to a weight (float) value, used for weighting the loss function (during training only). This can be useful to tell the model to "pay more attention" to samples from an under-represented class.</param>
        /// <param name="sample_weight">Optional Numpy array of weights for the training samples, used for weighting the loss function (during training only). You can either pass a flat (1D) Numpy array with the same length as the input samples (1:1 mapping between weights and samples), or in the case of temporal data, you can pass a 2D array with shape (samples, sequence_length), to apply a different weight to every timestep of every sample. In this case you should make sure to specifysample_weight_mode="temporal" in compile().</param>
        /// <param name="initial_epoch">Integer. Epoch at which to start training (useful for resuming a previous training run).</param>
        /// <param name="steps_per_epoch">Integer or None. Total number of steps (batches of samples) before declaring one epoch finished and starting the next epoch. When training with input tensors such as TensorFlow data tensors, the default None is equal to the number of samples in your dataset divided by the batch size, or 1 if that cannot be determined.</param>
        /// <param name="validation_steps">Only relevant if steps_per_epoch is specified. Total number of steps (batches of samples) to validate before stopping.</param>
        /// <param name="validation_batch_size">Integer or None. Number of samples per validation batch. If unspecified, will default to batch_size. Do not specify the validation_batch_size if your data is in the form of datasets, generators, or keras.utils.Sequence instances</param>
        /// <param name="validation_freq">Only relevant if validation data is provided. Integer or collections.abc.Container instance (e.g. list, tuple, etc.). If an integer, specifies how many training epochs to run before a new validation run is performed, e.g. validation_freq=2 runs validation every 2 epochs. If a Container, specifies the epochs on which to run validation, e.g. validation_freq=[1, 2, 10] runs validation at the end of the 1st, 2nd, and 10th epochs.</param>
        /// <param name="max_queue_size">Integer. Used for generator or keras.utils.Sequence input only. Maximum size for the generator queue. If unspecified, max_queue_size will default to 10.</param>
        /// <param name="workers">Integer. Used for generator or keras.utils.Sequence input only. Maximum number of processes to spin up when using process-based threading. If unspecified, workers will default to 1.</param>
        /// <param name="use_multiprocessing">Boolean. Used for generator or keras.utils.Sequence input only. If True, use process-based threading. If unspecified, use_multiprocessing will default to False. .</param>
        /// <returns>A History object. Its History.history attribute is a record of training loss values and metrics values at successive epochs, as well as validation loss values and validation metrics values (if applicable).</returns>
        public History Fit(NDarray x, NDarray y, int? batch_size = null, int epochs = 1, int verbose = 1, Callback[] callbacks = null,
                        float validation_split = 0.0f, NDarray[] validation_data = null, bool shuffle = true, Dictionary<int, float> class_weight = null,
                        NDarray sample_weight = null, int initial_epoch = 0, int? steps_per_epoch = null, int? validation_steps = null,
                        int? validation_batch_size = null, int[] validation_freq = null, int max_queue_size = 10, int workers = 1, bool use_multiprocessing = false)
        {
            var args = new Dictionary<string, object>();
            args["x"] = x;
            args["y"] = y;
            args["batch_size"] = batch_size;
            args["epochs"] = epochs;
            args["verbose"] = verbose;
            args["callbacks"] = callbacks;
            args["validation_split"] = validation_split;
            if (validation_data != null)
            {
                if (validation_data.Length == 2)
                    args["validation_data"] = new PyTuple (new PyObject[] { validation_data[0].PyObject, validation_data[1].PyObject });
                else if (validation_data.Length == 3)
                    args["validation_data"] = new PyTuple (new PyObject[] { validation_data[0].PyObject, validation_data[1].PyObject, validation_data[2].PyObject });
            }

            args["shuffle"] = shuffle;
            if (class_weight != null)
                args["class_weight"] = ToDict(class_weight);
            args["sample_weight"] = sample_weight;
            args["initial_epoch"] = initial_epoch;
            args["steps_per_epoch"] = steps_per_epoch;
            args["validation_steps"] = validation_steps;
            args["validation_batch_size"] = validation_batch_size;
            args["validation_freq"] = validation_freq;
            args["max_queue_size"] = max_queue_size;
            args["workers"] = workers;
            args["use_multiprocessing"] = use_multiprocessing;

            PyObject py = InvokeMethod("fit", args);

            return new History(py);
        }


        /// <summary>
        /// Trains the model for a given number of epochs (iterations on a dataset).
        /// </summary>
        /// <param name="x">Numpy array of training data (if the model has a single input), or list of Numpy arrays (if the model has multiple inputs). If input layers in the model are named, you can also pass a dictionary mapping input names to Numpy arrays. x can be None (default) if feeding from framework-native tensors (e.g. TensorFlow data tensors).</param>
        /// <param name="y">Numpy array of target (label) data (if the model has a single output), or list of Numpy arrays (if the model has multiple outputs). If output layers in the model are named, you can also pass a dictionary mapping output names to Numpy arrays. y can be None (default) if feeding from framework-native tensors (e.g. TensorFlow data tensors).</param>
        /// <param name="batch_size">Integer or None. Number of samples per gradient update. If unspecified, batch_sizewill default to 32.</param>
        /// <param name="epochs">Integer. Number of epochs to train the model. An epoch is an iteration over the entire x and y data provided. Note that in conjunction with initial_epoch, epochs is to be understood as "final epoch". The model is not trained for a number of iterations given by epochs, but merely until the epoch of index epochs is reached.</param>
        /// <param name="verbose">Integer. 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.</param>
        /// <param name="callbacks">List of keras.callbacks.Callback instances. List of callbacks to apply during training and validation (if ). See callbacks.</param>
        /// <param name="validation_split">Float between 0 and 1. Fraction of the training data to be used as validation data. The model will set apart this fraction of the training data, will not train on it, and will evaluate the loss and any model metrics on this data at the end of each epoch. The validation data is selected from the last samples in the x and y data provided, before shuffling.</param>
        /// <param name="validation_data">tuple (x_val, y_val) or tuple (x_val, y_val, val_sample_weights) on which to evaluate the loss and any model metrics at the end of each epoch. The model will not be trained on this data. validation_data will override validation_split.</param>
        /// <param name="shuffle">Boolean (whether to shuffle the training data before each epoch) or str (for 'batch'). 'batch' is a special option for dealing with the limitations of HDF5 data; it shuffles in batch-sized chunks. Has no effect when steps_per_epoch is not None.</param>
        /// <param name="class_weight">Optional dictionary mapping class indices (integers) to a weight (float) value, used for weighting the loss function (during training only). This can be useful to tell the model to "pay more attention" to samples from an under-represented class.</param>
        /// <param name="sample_weight">Optional Numpy array of weights for the training samples, used for weighting the loss function (during training only). You can either pass a flat (1D) Numpy array with the same length as the input samples (1:1 mapping between weights and samples), or in the case of temporal data, you can pass a 2D array with shape (samples, sequence_length), to apply a different weight to every timestep of every sample. In this case you should make sure to specifysample_weight_mode="temporal" in compile().</param>
        /// <param name="initial_epoch">Integer. Epoch at which to start training (useful for resuming a previous training run).</param>
        /// <param name="steps_per_epoch">Integer or None. Total number of steps (batches of samples) before declaring one epoch finished and starting the next epoch. When training with input tensors such as TensorFlow data tensors, the default None is equal to the number of samples in your dataset divided by the batch size, or 1 if that cannot be determined.</param>
        /// <param name="validation_steps">Only relevant if steps_per_epoch is specified. Total number of steps (batches of samples) to validate before stopping.</param>
        /// <param name="validation_batch_size">Integer or None. Number of samples per validation batch. If unspecified, will default to batch_size. Do not specify the validation_batch_size if your data is in the form of datasets, generators, or keras.utils.Sequence instances</param>
        /// <param name="validation_freq">Only relevant if validation data is provided. Integer or collections.abc.Container instance (e.g. list, tuple, etc.). If an integer, specifies how many training epochs to run before a new validation run is performed, e.g. validation_freq=2 runs validation every 2 epochs. If a Container, specifies the epochs on which to run validation, e.g. validation_freq=[1, 2, 10] runs validation at the end of the 1st, 2nd, and 10th epochs.</param>
        /// <param name="max_queue_size">Integer. Used for generator or keras.utils.Sequence input only. Maximum size for the generator queue. If unspecified, max_queue_size will default to 10.</param>
        /// <param name="workers">Integer. Used for generator or keras.utils.Sequence input only. Maximum number of processes to spin up when using process-based threading. If unspecified, workers will default to 1.</param>
        /// <param name="use_multiprocessing">Boolean. Used for generator or keras.utils.Sequence input only. If True, use process-based threading. If unspecified, use_multiprocessing will default to False. .</param>
        /// <returns>A History object. Its History.history attribute is a record of training loss values and metrics values at successive epochs, as well as validation loss values and validation metrics values (if applicable).</returns>
        public History Fit(NDarray x, NDarray[] y, int? batch_size = null, int epochs = 1, int verbose = 1, Callback[] callbacks = null,
                        float validation_split = 0.0f, NDarray[] validation_data = null, bool shuffle = true, Dictionary<int, float> class_weight = null,
                        NDarray sample_weight = null, int initial_epoch = 0, int? steps_per_epoch = null, int? validation_steps = null,
                        int? validation_batch_size = null, int[] validation_freq = null, int max_queue_size = 10, int workers = 1, bool use_multiprocessing = false)
        {
            var args = new Dictionary<string, object>();
            args["x"] = x;
            args["y"] = y;
            args["batch_size"] = batch_size;
            args["epochs"] = epochs;
            args["verbose"] = verbose;
            args["callbacks"] = callbacks;
            args["validation_split"] = validation_split;
            if (validation_data != null)
            {
                if (validation_data.Length == 2)
                    args["validation_data"] = new PyTuple(new PyObject[] { validation_data[0].PyObject, validation_data[1].PyObject });
                else if (validation_data.Length == 3)
                    args["validation_data"] = new PyTuple(new PyObject[] { validation_data[0].PyObject, validation_data[1].PyObject, validation_data[2].PyObject });
            }

            args["shuffle"] = shuffle;
            if (class_weight != null)
                args["class_weight"] = ToDict(class_weight);
            args["sample_weight"] = sample_weight;
            args["initial_epoch"] = initial_epoch;
            args["steps_per_epoch"] = steps_per_epoch;
            args["validation_steps"] = validation_steps;
            args["validation_batch_size"] = validation_batch_size;
            args["validation_freq"] = validation_freq;
            args["max_queue_size"] = max_queue_size;
            args["workers"] = workers;
            args["use_multiprocessing"] = use_multiprocessing;

            PyObject py = InvokeMethod("fit", args);

            return new History(py);
        }

        /// <summary>
        /// Returns the loss value & metrics values for the model in test mode.        Computation is done in batches.
        /// </summary>
        /// <param name="x"> Numpy array of test data (if the model has a single input), or list of Numpy arrays (if the model has multiple inputs). If input layers in the model are named, you can also pass a dictionary mapping input names to Numpy arrays. x can be None (default) if feeding from framework-native tensors (e.g. TensorFlow data tensors).</param>
        /// <param name="y"> Numpy array of target (label) data (if the model has a single output), or list of Numpy arrays (if the model has multiple outputs). If output layers in the model are named, you can also pass a dictionary mapping output names to Numpy arrays. y can be None (default) if feeding from framework-native tensors (e.g. TensorFlow data tensors).</param>
        /// <param name="batch_size"> Integer or None. Number of samples per evaluation step. If unspecified, batch_sizewill default to 32.</param>
        /// <param name="verbose"> 0 or 1. Verbosity mode. 0 = silent, 1 = progress bar.</param>
        /// <param name="sample_weight"> Optional Numpy array of weights for the test samples, used for weighting the loss function. You can either pass a flat (1D) Numpy array with the same length as the input samples (1:1 mapping between weights and samples), or in the case of temporal data, you can pass a 2D array with shape (samples, sequence_length), to apply a different weight to every timestep of every sample. In this case you should make sure to specifysample_weight_mode="temporal" in compile().</param>
        /// <param name="steps"> Integer or None. Total number of steps (batches of samples) before declaring the evaluation round finished. Ignored with the default value of None.</param>
        /// <param name="callbacks"> List of keras.callbacks.Callback instances. List of callbacks to apply during evaluation. See callbacks.</param>
        /// <param name="max_queue_size"> Integer. Used for generator or keras.utils.Sequence input only. Maximum size for the generator queue. If unspecified, max_queue_size will default to 10.</param>
        /// <param name="workers"> Integer. Used for generator or keras.utils.Sequence input only. Maximum number of processes to spin up when using process-based threading. If unspecified, workers will default to 1.</param>
        /// <param name="use_multiprocessing"> Boolean. Used for generator or keras.utils.Sequence input only. If True, use process-based threading. If unspecified, use_multiprocessing will default to False. </param>
        /// <param name="return_dict"> If True, loss and metric results are returned as a dict, with each key being the name of the metric. If False, they are returned as a list.</param>
        /// <returns>Scalar test loss (if the model has a single output and no metrics) or list of scalars (if the model has multiple outputs and/or metrics). The attribute model.metrics_names will give you the display labels for the scalar outputs.</returns>
        public double[] Evaluate(NDarray x, NDarray y, int? batch_size = null, int verbose = 1, 
            NDarray sample_weight = null, int? steps = null, Callback[] callbacks = null, int max_queue_size=10,
            int workers = 1, bool use_multiprocessing = false, bool return_dict = false)
        {
            var args = new Dictionary<string, object>();
            args["x"] = x.PyObject;
            args["y"] = y.PyObject;
            args["batch_size"] = batch_size;
            args["verbose"] = verbose;
            args["sample_weight"] = sample_weight;
            args["steps"] = steps;
            args["callbacks"] = callbacks != null ? callbacks : null;
            args["max_queue_size"] = max_queue_size;
            args["workers"] = workers;
            args["use_multiprocessing"] = use_multiprocessing;
            args["return_dict"] = return_dict;

            return InvokeMethod("evaluate", args)?.As<double[]>();
        }

        /// <summary>
        /// Generates output predictions for the input samples.
        /// Computation is done in batches.
        /// </summary>
        /// <param name="x">The input data, as a Numpy array (or list of Numpy arrays if the model has multiple inputs).</param>
        /// <param name="batch_size">Integer. If unspecified, it will default to 32.</param>
        /// <param name="verbose">Verbosity mode, 0 or 1.</param>
        /// <param name="steps">Total number of steps (batches of samples) before declaring the prediction round finished. Ignored with the default value of None.</param>
        /// <param name="callbacks">List of keras.callbacks.Callback instances. List of callbacks to apply during prediction. See callbacks.</param>
        /// <param name="max_queue_size">Integer. Used for generator or keras.utils.Sequence input only. Maximum size for the generator queue. If unspecified, max_queue_size will default to 10.</param>
        /// <param name="workers">Integer. Used for generator or keras.utils.Sequence input only. Maximum number of processes to spin up when using process-based threading. If unspecified, workers will default to 1.</param>
        /// <param name="use_multiprocessing">Boolean. Used for generator or keras.utils.Sequence input only. If True, use process-based threading. If unspecified, use_multiprocessing will default to False. .</param>
        /// <returns>Numpy array(s) of predictions.</returns>
        public NDarray Predict(NDarray x, int? batch_size = null, int verbose = 1, int? steps = null
            , Callback[] callbacks = null, int max_queue_size = 10, int workers = 1, bool use_multiprocessing = false)
        {
            var args = new Dictionary<string, object>();
            args["x"] = x;
            args["batch_size"] = batch_size;
            args["verbose"] = verbose;
            args["steps"] = steps;
            args["callbacks"] = callbacks != null ? callbacks : null;
            args["max_queue_size"] = max_queue_size;
            args["workers"] = workers;
            args["use_multiprocessing"] = use_multiprocessing;

            return new NDarray(InvokeMethod("predict", args));
        }

        /// <summary>
        /// Generates output predictions for the input samples.
        /// Computation is done in batches.
        /// </summary>
        /// <param name="x">The input data, as a Numpy array (or list of Numpy arrays if the model has multiple inputs).</param>
        /// <param name="batch_size">Integer. If unspecified, it will default to 32.</param>
        /// <param name="verbose">Verbosity mode, 0 or 1.</param>
        /// <param name="steps">Total number of steps (batches of samples) before declaring the prediction round finished. Ignored with the default value of None.</param>
        /// <param name="callbacks">List of keras.callbacks.Callback instances. List of callbacks to apply during prediction. See callbacks.</param>
        /// <returns>Numpy array(s) of predictions.</returns>
        public NDarray[] PredictMultipleOutputs(NDarray x, int? batch_size = null, int verbose = 1, int? steps = null, Callback[] callbacks = null)
        {
            var args = new Dictionary<string, object>();
            args["x"] = x;
            args["batch_size"] = batch_size;
            args["verbose"] = verbose;
            args["steps"] = steps;
            args["callbacks"] = callbacks != null ? callbacks : null;

            var res = InvokeMethod("predict", args);
            var resTuple = PyTuple.AsTuple(res);

            var length = resTuple.Length();
            var finalRes = new NDarray[length];
            for (int i = 0; i < length; i++)
            {
                finalRes[i] = new NDarray(resTuple[i]);
            }
            return finalRes;
        }


        /// <summary>
        /// Generates output predictions for the list of inputs given samples.
        /// Computation is done in batches.
        /// </summary>
        /// <param name="x">The List of input data, as a Numpy array (or list of Numpy arrays if the model has multiple inputs).</param>
        /// <param name="batch_size">Integer. If unspecified, it will default to 32.</param>
        /// <param name="verbose">Verbosity mode, 0 or 1.</param>
        /// <param name="steps">Total number of steps (batches of samples) before declaring the prediction round finished. Ignored with the default value of None.</param>
        /// <param name="callbacks">List of keras.callbacks.Callback instances. List of callbacks to apply during prediction. See callbacks.</param>
        /// <param name="max_queue_size">Integer. Used for generator or keras.utils.Sequence input only. Maximum size for the generator queue. If unspecified, max_queue_size will default to 10.</param>
        /// <param name="workers">Integer. Used for generator or keras.utils.Sequence input only. Maximum number of processes to spin up when using process-based threading. If unspecified, workers will default to 1.</param>
        /// <param name="use_multiprocessing">Boolean. Used for generator or keras.utils.Sequence input only. If True, use process-based threading. If unspecified, use_multiprocessing will default to False. .</param>
        /// <returns>Numpy array(s) of predictions.</returns>
        public NDarray Predict(List<NDarray> x, int? batch_size = null, int verbose = 1, int? steps = null, 
            Callback[] callbacks = null, int max_queue_size = 10, int workers = 1, bool use_multiprocessing = false)
        {

            var args = new Dictionary<string, object>();

            PyObject[] items = new PyObject[x.Count];
            for (int i = 0; i < x.Count; i++)
            {
                items[i] = x[i].PyObject;
            }
            PyTuple x_tuple = new PyTuple(items);

            args["x"] = x_tuple;
            args["batch_size"] = batch_size;
            args["verbose"] = verbose;
            args["steps"] = steps;
            args["callbacks"] = callbacks != null ? callbacks : null;
            args["max_queue_size"] = max_queue_size;
            args["workers"] = workers;
            args["use_multiprocessing"] = use_multiprocessing;

            return new NDarray(InvokeMethod("predict", args));
        }


        /// <summary>
        /// Runs a single gradient update on a single batch of data.
        /// </summary>
        /// <param name="x">Numpy array of training data, or list of Numpy arrays if the model has multiple inputs. If all inputs in the model are named, you can also pass a dictionary mapping input names to Numpy arrays.</param>
        /// <param name="y">Numpy array of target data, or list of Numpy arrays if the model has multiple outputs. If all outputs in the model are named, you can also pass a dictionary mapping output names to Numpy arrays.</param>
        /// <param name="sample_weight">Optional array of the same length as x, containing weights to apply to the model's loss for each sample. In the case of temporal data, you can pass a 2D array with shape (samples, sequence_length), to apply a different weight to every timestep of every sample. In this case you should make sure to specify sample_weight_mode="temporal" in compile().</param>
        /// <param name="class_weight">Optional dictionary mapping class indices (integers) to a weight (float) to apply to the model's loss for the samples from this class during training. This can be useful to tell the model to "pay more attention" to samples from an under-represented class.</param>
        /// <param name="reset_metrics">Optional array of the same length as x, containing weights to apply to the model's loss for each sample. In the case of temporal data, you can pass a 2D array with shape (samples, sequence_length), to apply a different weight to every timestep of every sample. In this case you should make sure to specify sample_weight_mode="temporal" in compile().</param> 
        /// <param name="return_dict">If True, loss and metric results are returned as a dict, with each key being the name of the metric. If False, they are returned as a list.</param>
        /// <returns>Scalar training loss (if the model has a single output and no metrics) or list of scalars (if the model has multiple outputs and/or metrics). The attribute model.metrics_names will give you the display labels for the scalar outputs.</returns>

        public double[] TrainOnBatch(NDarray x, NDarray y, NDarray sample_weight = null, Dictionary<int, float> class_weight = null, bool reset_metrics = false, bool return_dict = false)
        {
            var args = new Dictionary<string, object>();
            args["x"] = x;
            args["y"] = y;
            args["sample_weight"] = sample_weight;
            args["class_weight"] = class_weight;
            args["reset_metrics"] = reset_metrics;
            args["return_dict"] = return_dict;

            var pyresult = InvokeMethod("train_on_batch", args);
            if (pyresult == null) return default;
            double[] result;
            if (!pyresult.IsIterable())
                result = new double[] { pyresult.As<double>() };
            else
                result = pyresult.As<double[]>();
            pyresult.Dispose();
            return result;
        }

        /// <summary>
        /// Runs a single gradient update on a single batch of data.
        /// </summary>
        /// <param name="x">Numpy array of training data, or list of Numpy arrays if the model has multiple inputs. If all inputs in the model are named, you can also pass a dictionary mapping input names to Numpy arrays.</param>
        /// <param name="y">Numpy array of target data, or list of Numpy arrays if the model has multiple outputs. If all outputs in the model are named, you can also pass a dictionary mapping output names to Numpy arrays.</param>
        /// <param name="sample_weight">Optional array of the same length as x, containing weights to apply to the model's loss for each sample. In the case of temporal data, you can pass a 2D array with shape (samples, sequence_length), to apply a different weight to every timestep of every sample. In this case you should make sure to specify sample_weight_mode="temporal" in compile().</param>
        /// <param name="class_weight">Optional dictionary mapping class indices (integers) to a weight (float) to apply to the model's loss for the samples from this class during training. This can be useful to tell the model to "pay more attention" to samples from an under-represented class.</param>
        /// <param name="reset_metrics">Optional array of the same length as x, containing weights to apply to the model's loss for each sample. In the case of temporal data, you can pass a 2D array with shape (samples, sequence_length), to apply a different weight to every timestep of every sample. In this case you should make sure to specify sample_weight_mode="temporal" in compile().</param> 
        /// <param name="return_dict">If True, loss and metric results are returned as a dict, with each key being the name of the metric. If False, they are returned as a list.</param>
        /// <returns>Scalar training loss (if the model has a single output and no metrics) or list of scalars (if the model has multiple outputs and/or metrics). The attribute model.metrics_names will give you the display labels for the scalar outputs.</returns>
        public double[] TrainOnBatch(NDarray[] x, NDarray y, NDarray sample_weight = null, Dictionary<int, float> class_weight = null, bool reset_metrics = false, bool return_dict = false)
        {
            var args = new Dictionary<string, object>();
            args["x"] = x;
            args["y"] = y;
            args["sample_weight"] = sample_weight;
            args["class_weight"] = class_weight;
            args["reset_metrics"] = reset_metrics;
            args["return_dict"] = return_dict;

            var pyresult = InvokeMethod("train_on_batch", args);
            if (pyresult == null) return default;
            double[] result;
            if (!pyresult.IsIterable())
                result = new double[] { pyresult.As<double>() };
            else
                result = pyresult.As<double[]>();
            pyresult.Dispose();
            return result;
        }

        /// <summary>
        /// Tests the on batch.
        /// </summary>
        /// <param name="x">Numpy array of test data, or list of Numpy arrays if the model has multiple inputs. If all inputs in the model are named, you can also pass a dictionary mapping input names to Numpy arrays.</param>
        /// <param name="y">Numpy array of target data, or list of Numpy arrays if the model has multiple outputs. If all outputs in the model are named, you can also pass a dictionary mapping output names to Numpy arrays.</param>
        /// <param name="sample_weight">Optional array of the same length as x, containing weights to apply to the model's loss for each sample. In the case of temporal data, you can pass a 2D array with shape (samples, sequence_length), to apply a different weight to every timestep of every sample. In this case you should make sure to specify sample_weight_mode="temporal" in compile().</param>
        /// <param name="reset_metrics">Optional array of the same length as x, containing weights to apply to the model's loss for each sample. In the case of temporal data, you can pass a 2D array with shape (samples, sequence_length), to apply a different weight to every timestep of every sample. In this case you should make sure to specify sample_weight_mode="temporal" in compile().</param> 
        /// <param name="return_dict">If True, loss and metric results are returned as a dict, with each key being the name of the metric. If False, they are returned as a list.</param>
        /// <returns>Scalar test loss (if the model has a single output and no metrics) or list of scalars (if the model has multiple outputs and/or metrics). The attribute model.metrics_names will give you the display labels for the scalar outputs.</returns>
        public double[] TestOnBatch(NDarray x, NDarray y, NDarray sample_weight = null, bool reset_metrics = false, bool return_dict = false)
        {
            var args = new Dictionary<string, object>();
            args["x"] = x;
            args["y"] = y;
            args["sample_weight"] = sample_weight;
            args["reset_metrics"] = reset_metrics;
            args["return_dict"] = return_dict;

            //return InvokeMethod("test_on_batch", args)?.As<double[]>();
            var pyresult = InvokeMethod("test_on_batch", args);
            if (pyresult == null) return default;
            double[] result;
            if (!pyresult.IsIterable())
                result = new double[] { pyresult.As<double>() };
            else
                result = pyresult.As<double[]>();
            pyresult.Dispose();
            return result;
        }

        /// <summary>
        /// Tests the on batch.
        /// </summary>
        /// <param name="x">Numpy array of test data, or list of Numpy arrays if the model has multiple inputs. If all inputs in the model are named, you can also pass a dictionary mapping input names to Numpy arrays.</param>
        /// <param name="y">Numpy array of target data, or list of Numpy arrays if the model has multiple outputs. If all outputs in the model are named, you can also pass a dictionary mapping output names to Numpy arrays.</param>
        /// <param name="sample_weight">Optional array of the same length as x, containing weights to apply to the model's loss for each sample. In the case of temporal data, you can pass a 2D array with shape (samples, sequence_length), to apply a different weight to every timestep of every sample. In this case you should make sure to specify sample_weight_mode="temporal" in compile().</param>
        /// <param name="reset_metrics">Optional array of the same length as x, containing weights to apply to the model's loss for each sample. In the case of temporal data, you can pass a 2D array with shape (samples, sequence_length), to apply a different weight to every timestep of every sample. In this case you should make sure to specify sample_weight_mode="temporal" in compile().</param> 
        /// <param name="return_dict">If True, loss and metric results are returned as a dict, with each key being the name of the metric. If False, they are returned as a list.</param>
        /// <returns>Scalar test loss (if the model has a single output and no metrics) or list of scalars (if the model has multiple outputs and/or metrics). The attribute model.metrics_names will give you the display labels for the scalar outputs.</returns>
        public double[] TestOnBatch(NDarray[] x, NDarray y, NDarray sample_weight = null, bool reset_metrics = false, bool return_dict = false)
        {
            var args = new Dictionary<string, object>();
            args["x"] = x;
            args["y"] = y;
            args["sample_weight"] = sample_weight;
            args["reset_metrics"] = reset_metrics;
            args["return_dict"] = return_dict;

            //return InvokeMethod("test_on_batch", args)?.As<double[]>();
            var pyresult = InvokeMethod("test_on_batch", args);
            if (pyresult == null) return default;
            double[] result;
            if (!pyresult.IsIterable())
                result = new double[] { pyresult.As<double>() };
            else
                result = pyresult.As<double[]>();
            pyresult.Dispose();
            return result;
        }

        /// <summary>
        /// Returns predictions for a single batch of samples.
        /// </summary>
        /// <param name="x">Input samples, as a Numpy array.</param>
        /// <returns>Numpy array(s) of predictions.</returns>
        public NDarray PredictOnBatch(NDarray x)
        {
            var args = new Dictionary<string, object>();
            args["x"] = x;

            return new NDarray(InvokeMethod("predict_on_batch", args));
        }

        /// <summary>
        /// Returns predictions for a single batch of samples.
        /// </summary>
        /// <param name="x">Input samples, as a Numpy array.</param>
        /// <returns>Numpy array(s) of predictions.</returns>
        public NDarray PredictOnBatch(NDarray[] x)
        {
            var args = new Dictionary<string, object>();
            args["x"] = x;

            return new NDarray(InvokeMethod("predict_on_batch", args));
        }

        /// <summary>
        /// The logic for one inference step.
        /// </summary>
        /// <param name="data">A nested structure of Tensors.</param>
        /// <returns>Numpy array(s) of predictions.</returns>
        public NDarray PredictStep(NDarray data)
        {
            var args = new Dictionary<string, object>();
            args["data"] = data;

            return new NDarray(InvokeMethod("predict_step", args));
        }

        /// <summary>
        /// The logic for one inference step.
        /// </summary>
        /// <param name="data">A nested structure of Tensors.</param>
        /// <returns>Numpy array(s) of predictions.</returns>
        public NDarray PredictStep(NDarray[] data)
        {
            var args = new Dictionary<string, object>();
            args["data"] = data;

            return new NDarray(InvokeMethod("predict_step", args));
        }

        public History FitGenerator(Sequence generator, int? steps_per_epoch = null, int epochs = 1, int verbose = 1, Callback[] callbacks = null,
                    Sequence validation_data = null, int? validation_steps = null, int validation_freq = 1, Dictionary<int, float> class_weight = null,
                    int max_queue_size = 10, int workers = 1, bool use_multiprocessing = false, bool shuffle = true, int initial_epoch = 0)
        {
            var args = new Dictionary<string, object>();
            args["generator"] = generator;
            args["steps_per_epoch"] = steps_per_epoch;
            args["epochs"] = epochs;
            args["verbose"] = verbose;
            args["callbacks"] = callbacks;
            if (validation_data != null)
            {
                args["validation_data"] = validation_data;
                //if (validation_data.Length == 2)
                //    args["validation_data"] = new NDarray[] { validation_data[0], validation_data[1] };
                //else if (validation_data.Length == 3)
                //    args["validation_data"] = new NDarray[] { validation_data[0], validation_data[1], validation_data[2] };
            }

            args["validation_steps"] = validation_steps;
            //args["validation_freq"] = validation_freq;
            args["class_weight"] = class_weight;
            args["max_queue_size"] = max_queue_size;
            args["workers"] = workers;
            args["use_multiprocessing"] = use_multiprocessing;
            args["shuffle"] = shuffle;
            args["initial_epoch"] = initial_epoch;

            var py = InvokeMethod("fit_generator", args);

            return new History(py);
        }

        public double[] EvaluateGenerator(Sequence generator, int? steps = null, Callback[] callbacks = null,
                                int max_queue_size = 10, int workers = 1, bool use_multiprocessing = false, int verbose = 0)
        {
            var args = new Dictionary<string, object>();
            args["generator"] = generator;
            args["steps"] = steps;
            args["callbacks"] = callbacks;
            args["max_queue_size"] = max_queue_size;
            args["workers"] = workers;
            args["use_multiprocessing"] = use_multiprocessing;
            args["verbose"] = verbose;
            var pyresult = InvokeMethod("evaluate_generator", args);
            if (pyresult == null) return default;
            double[] result;
            if (!pyresult.IsIterable())
                result = new double[] { pyresult.As<double>() };
            else
                result = pyresult.As<double[]>();
            pyresult.Dispose();
            return result;
        }

        public NDarray PredictGenerator(Sequence generator, int? steps = null, Callback[] callbacks = null,
                                int max_queue_size = 10, int workers = 1, bool use_multiprocessing = false, int verbose = 0)
        {
            var args = new Dictionary<string, object>();
            args["generator"] = generator;
            args["steps"] = steps;
            args["callbacks"] = callbacks;
            args["max_queue_size"] = max_queue_size;
            args["workers"] = workers;
            args["use_multiprocessing"] = use_multiprocessing;
            args["verbose"] = verbose;
            var py = InvokeMethod("predict_generator", args);

            return new NDarray(py);
        }


        /// <summary>
        /// Converts the model to json.
        /// </summary>
        /// <returns></returns>
        public string ToJson()
        {
            return PyInstance.to_json().ToString();
        }

        /// <summary>
        /// Saves the weight of the trained model to a file.
        /// </summary>
        /// <param name="filepath">str or pathlib.Path object. Path where to save the model.</param>
        /// <param name="overwrite">Whether we should overwrite any existing model at the target location, or instead ask the user via an interactive prompt.</param>
        /// <param name="save_format">Either "keras", "tf", "h5", indicating whether to save the model in the native Keras format (.keras), in the TensorFlow SavedModel format (referred to as "SavedModel" below), or in the legacy HDF5 format (.h5). Defaults to "tf" in TF 2.X, and "h5" in TF 1.X.</param>
        public void SaveWeight(string filepath, bool overwrite = true, string save_format = "tf")
        {
            PyInstance.save_weights(filepath, overwrite, save_format);
        }

        /// <summary>
        /// Saves a model as a TensorFlow SavedModel or HDF5 file.
        /// </summary>
        /// <param name="filepath">str or pathlib.Path object. Path where to save the model.</param>
        /// <param name="overwrite">Whether we should overwrite any existing model at the target location, or instead ask the user via an interactive prompt.</param>
        /// <param name="save_format">Either "keras", "tf", "h5", indicating whether to save the model in the native Keras format (.keras), in the TensorFlow SavedModel format (referred to as "SavedModel" below), or in the legacy HDF5 format (.h5). Defaults to "tf" in TF 2.X, and "h5" in TF 1.X.</param>
        public void Save(string filepath, bool overwrite = true, string save_format = "tf")
        {
            PyInstance.save(filepath: filepath, overwrite: overwrite, save_format: save_format);
        }

        /// <summary>
        /// Retrieves the weights of the model
        /// </summary>
        /// <returns>A flat list of Numpy arrays</returns>
        public List<NDarray> GetWeights()
        {
            var args = new Dictionary<string, object>();
            var pyWeights = PyInstance.get_weights();
            List<NDarray> weights = new List<NDarray>();

            foreach (PyObject weightsArray in pyWeights)
            {
                var n = np.array(new NDarray(weightsArray));
                weights.Add(n);
            }

            return weights;
        }

        /// <summary>
        /// Sets the weights of the model
        /// </summary>
        /// <param name="weights">A list of Numpy arrays with shapes and types matching the output of model.GetWeights()</param>
        public void SetWeights(List<NDarray> weights)
        {
            PyList list = new PyList();
            foreach (var item in weights)
            {
                list.Append(item.PyObject);
            }

            PyInstance.set_weights(list);
        }

        /// <summary>
        /// Loads the weight to the model from a file.
        /// </summary>
        /// <param name="filepath">String, path to the weights file to load. For weight files in TensorFlow format, this is the file prefix (the same as was passed to save_weights()).</param>
        /// <param name="skip_mismatch">A list of Numpy arrays with shapes and types matching the output of model.GetWeights()</param>
        /// <param name="by_name">A list of Numpy arrays with shapes and types matching the output of model.GetWeights()</param>
        /// <param name="options">A list of Numpy arrays with shapes and types matching the output of model.GetWeights()</param>
        public void LoadWeight(string filepath, bool skip_mismatch= false, bool by_name= false, CheckpointOptions options= null)
        {
            PyObject optionPyObject = null;
            if (options != null)
                optionPyObject = options.ToPython();

            PyInstance.load_weights(filepath, skip_mismatch, by_name, optionPyObject);
        }

        /// <summary>
        /// Loads the model.
        /// </summary>
        /// <param name="path">The path.</param>
        /// <returns></returns>
        public static BaseModel LoadModel(string filepath, Dictionary<string, PyObject> custom_objects = null, bool compile = true)
        {
            var model = new BaseModel();
            PyDict dict = null;
            if (custom_objects != null)
            {
                dict = new PyDict();
                foreach (var item in custom_objects)
                {
                    dict[item.Key] = item.Value.ToPython();
                }
            }

            model.PyInstance = Instance.keras.models.load_model(filepath, custom_objects: dict, compile: compile);

            return model;
        }

        /// <summary>
        /// Load the model from json.
        /// </summary>
        /// <param name="json_string">The json string.</param>
        /// <returns>The model</returns>
        public static BaseModel ModelFromJson(string json_string)
        {
            var model = new BaseModel();
            model.PyInstance = Instance.keras.models.model_from_json(json_string: json_string);

            return model;
        }

        /// <summary>
        /// Load the model from yaml.
        /// </summary>
        /// <param name="json_string">The json string.</param>
        /// <returns>The model</returns>
        public static BaseModel ModelFromYaml(string json_string)
        {
            var model = new BaseModel();
            model.PyInstance = Instance.keras.models.model_from_yaml(json_string: json_string);

            return model;
        }

        /// <summary>
        /// Saves keras model to onnx.
        /// </summary>
        /// <param name="filePath">The file path.</param>
        public void SaveOnnx(string filePath)
        {
            var onnx_model = Instance.keras2onnx.convert_keras(model: (PyObject)this.PyInstance);
            System.IO.File.WriteAllText(filePath, onnx_model.ToString());
        }

        /// <summary>
        /// Summaries the specified line length.
        /// </summary>
        /// <param name="line_length">Length of the line.</param>
        /// <param name="positions">The positions.</param>
        public void Summary(int? line_length = null, float[] positions = null, string print_fn = null, bool expand_nested = false, bool show_trainable = false, int[] layer_range = null)
        {
            PyInstance.summary(line_length: line_length, positions: positions, print_fn: print_fn, expand_nested: expand_nested, show_trainable: show_trainable, layer_range: layer_range);
        }

        /// <summary>
        /// Saves the tensorflow js format.
        /// </summary>
        /// <param name="artifacts_dir">The artifacts dir.</param>
        /// <param name="quantize">if set to <c>true</c> [quantize].</param>
        public void SaveTensorflowJSFormat(string artifacts_dir, bool quantize = false)
        {
            Instance.tfjs.converters.save_keras_model(model: this.PyInstance, artifacts_dir: artifacts_dir);
        }

        /// <summary>
        /// Export the model
        /// </summary>
        /// <param name="filepath">File path to export</param>
        public void Export(string filepath)
        {
            PyInstance.export(filepath);
        }

        /// <summary>
        /// Get the layer based on name and/or index
        /// </summary>
        /// <param name="name">Name of the layer</param>
        /// <param name="index">Index of the layer</param>
        /// <returns></returns>
        public BaseLayer GetLayer(string name = null, int? index = null)
        {
            return (BaseLayer)PyInstance.get_layer(name, index);
        }

        /// <summary>
        /// Resets the state of all the metrics in the model.
        /// </summary>
        public void ResetMetrics()
        {
            PyInstance.reset_metrics();
        }

        /// <summary>
        /// Reset the states of the model
        /// </summary>
        public void ResetStates()
        {
            PyInstance.reset_states();
        }
    }
}
