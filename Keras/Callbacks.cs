using Keras.Layers;
using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using Python.Runtime;
using Numpy;
using System.IO;
using static System.Net.WebRequestMethods;
using Keras.Models;

namespace Keras.Callbacks
{
    /// <summary>
    /// Abstract base class used to build new callbacks.
    /// The logs dictionary that callback methods take as argument will contain keys for quantities relevant to the current batch or epoch.
    /// Currently, the.fit() method of the Sequential model class will include the following quantities in the logs that it passes to its callbacks:
    /// on_epoch_end: logs include acc and loss, and optionally include val_loss(if validation is enabled in fit), and val_acc(if validation and accuracy monitoring are enabled). on_batch_begin: logs include size, the number of samples in the current batch.on_batch_end: logs include loss, and optionally acc(if accuracy monitoring is enabled).
    /// </summary>
    /// <seealso cref="Keras.Base" />
    public class Callback : Base
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="Callback"/> class.
        /// </summary>
        public Callback()
        {
            PyInstance = Instance.keras.callbacks.Callback();
        }

        public Callback(PyObject py)
        {
            PyInstance = py;
        }

        public static Callback Custom(string name, string fileOrcode, bool isFile = true)
        {
            string code = "";
            if(isFile)
            {
                code = System.IO.File.ReadAllText(fileOrcode);
            }
            else
            {
                code = fileOrcode;
            }

            PyObject py = PyModule.FromString(name, code);
            py = py.InvokeMethod(name);
            return new Callback(py);
        }

        public T Get<T>(string property)
        {
            return ((PyObject)PyInstance).GetAttr(property).As<T>();
        }

    }

    /// <summary>
    /// Callback that accumulates epoch averages of metrics.    This callback is automatically applied to every Keras model.
    /// </summary>
    /// <seealso cref="Keras.Callbacks.Callback" />
    public class BaseLogger : Callback
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="BaseLogger"/> class.
        /// </summary>
        /// <param name="stateful_metrics">Iterable of string names of metrics that should not be averaged over an epoch. Metrics in this list will be logged as-is in on_epoch_end. All others will be averaged in on_epoch_end.</param>
        public BaseLogger(params string[] stateful_metrics)
        {
            Parameters["stateful_metrics"] = stateful_metrics!=null ? stateful_metrics.ToList() : null;
            PyInstance = Instance.keras.callbacks.BaseLogger;
            Init();
        }
    }

    /// <summary>
    /// Callback that terminates training when a NaN loss is encountered.
    /// </summary>
    /// <seealso cref="Keras.Callbacks.Callback" />
    public class TerminateOnNaN : Callback
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="TerminateOnNaN"/> class.
        /// </summary>
        public TerminateOnNaN()
        {
            PyInstance = Instance.keras.callbacks.TerminateOnNaN;
            Init();
        }
    }

    /// <summary>
    /// Callback that prints metrics to stdout.
    /// </summary>
    /// <seealso cref="Keras.Callbacks.Callback" />
    public class ProgbarLogger : Callback
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="ProgbarLogger"/> class.
        /// </summary>
        /// <param name="count_mode">One of "steps" or "samples". Whether the progress bar should count samples seen or steps (batches) seen.</param>
        /// <param name="stateful_metrics"> Iterable of string names of metrics that should not be averaged over an epoch. Metrics in this list will be logged as-is. All others will be averaged over time (e.g. loss, etc).</param>
        public ProgbarLogger(string count_mode = "samples", params string[] stateful_metrics)
        {
            Parameters["count_mode"] = count_mode;
            Parameters["stateful_metrics"] = stateful_metrics != null ? stateful_metrics.ToList() : null;
            PyInstance = Instance.keras.callbacks.ProgbarLogger;
            Init();
        }
    }

    /// <summary>
    /// Callback that records events into a History object.    This callback is automatically applied to every Keras model.The History object gets returned by the fit method of models.
    /// </summary>
    /// <seealso cref="Keras.Callbacks.Callback" />
    public class History : Callback
    {
        /// <summary>
        /// Gets the epoch.
        /// </summary>
        /// <value>
        /// The epoch.
        /// </value>
        public int[] Epoch
        {
            get
            {
                return ((PyObject)PyInstance.epoch).As<int[]>();
            }
        }

        /// <summary>
        /// Gets the history logs.
        /// </summary>
        /// <value>
        /// The history logs.
        /// </value>
        public Dictionary<string, double[]> HistoryLogs
        {
            get
            {
                PyDict dict = new PyDict(PyInstance.history);
                Dictionary<string, double[]> result = new Dictionary<string, double[]>();
                var keys = dict.Keys().As<string[]>();
                foreach (var item in keys)
                {
                    result.Add(item, dict[item].As<double[]>());
                }

                return result;
            }
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="History"/> class.
        /// </summary>
        public History()
        {
            PyInstance = keras.callbacks.History;
            Init();
        }

        public History(PyObject py)
        {
            PyInstance = py;
        }
    }

    /// <summary>
    /// Save the model after every epoch.
    /// filepath can contain named formatting options, which will be filled with the values of epoch and keys in logs(passed in on_epoch_end).
    /// For example: if filepath is weights.{epoch:02d}-{val_loss:.2f}.hdf5, then the model checkpoints will be saved with the epoch number and the validation loss in the filename.
    /// </summary>
    /// <seealso cref="Keras.Callbacks.Callback" />
    public class ModelCheckpoint : Callback
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="ModelCheckpoint"/> class.
        /// </summary>
        /// <param name="filepath">string, path to save the model file.</param>
        /// <param name="monitor">quantity to monitor.</param>
        /// <param name="verbose">verbosity mode, 0 or 1.</param>
        /// <param name="save_best_only">if save_best_only=True, the latest best model according to the quantity monitored will not be overwritten.</param>
        /// <param name="save_weights_only"> if True, then only the model's weights will be saved (model.save_weights(filepath)), else the full model is saved (model.save(filepath)).</param>
        /// <param name="mode">one of {auto, min, max}. If save_best_only=True, the decision to overwrite the current save file is made based on either the maximization or the minimization of the monitored quantity. For  val_acc, this should be max, for val_loss this should be min, etc. In auto mode, the direction is automatically inferred from the name of the monitored quantity.</param>
        /// <param name="save_freq">'epoch' or integer. When using 'epoch', the callback saves the model after each epoch. When using integer, the callback saves the model at end of this many batches.</param>
        public ModelCheckpoint(string filepath, string monitor = "val_loss", int verbose = 0, bool save_best_only = false
                    , bool save_weights_only = false, string mode = "auto", string save_freq= "epoch")
        {
            Parameters["filepath"] = filepath;
            Parameters["monitor"] = monitor;
            Parameters["verbose"] = verbose;
            Parameters["save_best_only"] = save_best_only;
            Parameters["save_weights_only"] = save_weights_only;
            Parameters["mode"] = mode;
            Parameters["save_freq"] = save_freq;
            //ToDo: extend options parameter
            PyInstance = Instance.keras.callbacks.ModelCheckpoint;
            Init();
        }
    }

    /// <summary>
    /// Stop training when a monitored quantity has stopped improving.
    /// </summary>
    /// <seealso cref="Keras.Callbacks.Callback" />
    public class EarlyStopping : Callback
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="EarlyStopping"/> class.
        /// </summary>
        /// <param name="monitor">quantity to be monitored.</param>
        /// <param name="min_delta"> minimum change in the monitored quantity to qualify as an improvement, i.e. an absolute change of less than min_delta, will count as no improvement.</param>
        /// <param name="patience">number of epochs with no improvement after which training will be stopped.</param>
        /// <param name="verbose">verbosity mode.</param>
        /// <param name="mode">one of {auto, min, max}. In min mode, training will stop when the quantity monitored has stopped decreasing; in max mode it will stop when the quantity monitored has stopped increasing; in auto mode, the direction is automatically inferred from the name of the monitored quantity.</param>
        /// <param name="baseline"> Baseline value for the monitored quantity to reach. Training will stop if the model doesn't show improvement over the baseline.</param>
        /// <param name="restore_best_weights"> whether to restore model weights from the epoch with the best value of the monitored quantity. If False, the model weights obtained at the last step of training are used.</param>
        /// <param name="start_from_epoch"> Number of epochs to wait before starting to monitor improvement. This allows for a warm-up period in which no improvement is expected and thus training will not be stopped.</param>
        public EarlyStopping(string monitor = "val_loss", float min_delta = 0, int patience = 0, int verbose = 0, string mode = "auto", 
            float? baseline = null, bool restore_best_weights = false, int start_from_epoch = 0)
        {
            Parameters["monitor"] = monitor;
            Parameters["min_delta"] = min_delta;
            Parameters["patience"] = patience;
            Parameters["verbose"] = verbose;
            Parameters["mode"] = mode;
            Parameters["baseline"] = baseline;
            Parameters["restore_best_weights"] = restore_best_weights;
            Parameters["start_from_epoch"] = start_from_epoch;

            PyInstance = Instance.keras.callbacks.EarlyStopping;
            Init();
        }
    }

    /// <summary>
    /// Callback used to stream events to a server.
    /// Requires the requests library.Events are sent to root + '/publish/epoch/end/' by default. Calls are HTTP POST, with a data argument which is a JSON-encoded dictionary of event data.If send_as_json is set to True, the content type of the request will be application/json.Otherwise the serialized JSON will be send within a form
    /// </summary>
    /// <seealso cref="Keras.Callbacks.Callback" />
    public class RemoteMonitor : Callback
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="RemoteMonitor"/> class.
        /// </summary>
        /// <param name="root">String; root url of the target server.</param>
        /// <param name="path">String; path relative to root to which the events will be sent.</param>
        /// <param name="field">String; JSON field under which the data will be stored. The field is used only if the payload is sent within a form (i.e. send_as_json is set to False).</param>
        /// <param name="headers">Dictionary; optional custom HTTP headers.</param>
        /// <param name="send_as_json">Boolean; whether the request should be send as application/json.</param>
        public RemoteMonitor(string root = "http://localhost:9000", string path = "/publish/epoch/end/", string field = "data", Dictionary<string, string> headers = null, bool send_as_json = false)
        {
            Parameters["root"] = root;
            Parameters["path"] = path;
            Parameters["field"] = field;
            Parameters["headers"] = headers;
            Parameters["send_as_json"] = send_as_json;

            PyInstance = Instance.keras.callbacks.RemoteMonitor;
            Init();
        }
    }

    /// <summary>
    /// Learning rate scheduler.
    /// </summary>
    /// <seealso cref="Keras.Callbacks.Callback" />
    public class LearningRateScheduler : Callback
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="LearningRateScheduler"/> class.
        /// </summary>
        /// <param name="schedule">a function that takes an epoch index as input (integer, indexed from 0) and current learning rate and returns a new learning rate as output (float).</param>
        /// <param name="verbose">int. 0: quiet, 1: update messages.</param>
        public LearningRateScheduler(object schedule, int verbose= 0)
        {
            Parameters["schedule"] = schedule;
            Parameters["verbose"] = verbose;

            PyInstance = Instance.keras.callbacks.LearningRateScheduler;
            Init();
        }
    }

    /// <summary>
    /// TensorBoard basic visualizations. TensorBoard is a visualization tool provided with TensorFlow.
    /// This callback writes a log for TensorBoard, which allows you to visualize dynamic graphs of your training and test metrics, as well as activation histograms for the different layers in your model.
    /// If you have installed TensorFlow with pip, you should be able to launch TensorBoard from the command line:    tensorboard --logdir=/ full_path_to_your_logs
    /// When using a backend other than TensorFlow, TensorBoard will still work(if you have TensorFlow installed), but the only feature available will be the display of the losses and metrics plots.
    /// </summary>
    /// <seealso cref="Keras.Callbacks.Callback" />
    public class TensorBoard : Callback
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="LearningRateScheduler" /> class.
        /// </summary>
        /// <param name="log_dir"> the path of the directory where to save the log files to be parsed by TensorBoard.</param>
        /// <param name="histogram_freq"> frequency (in epochs) at which to compute activation and weight histograms for the layers of the model. If set to 0, histograms won't be computed. Validation data (or split) must be specified for histogram visualizations.</param>
        /// <param name="write_graph"> whether to visualize the graph in TensorBoard. The log file can become quite large when write_graph is set to True.</param>
        /// <param name="write_images"> whether to write model weights to visualize as image in TensorBoard.</param>
        /// <param name="write_steps_per_second"> whether to log the training steps per second into TensorBoard. This supports both epoch and batch frequency logging.</param>
        /// <param name="update_freq"> 'batch' or 'epoch' or integer. When using 'epoch', writes the losses and metrics to TensorBoard after every epoch. </param>
        /// <param name="embeddings_freq"> frequency (in epochs) at which selected embedding layers will be saved. If set to 0, embeddings won't be computed. Data to be visualized in TensorBoard's Embedding tab must be passed as embeddings_data.</param>
        /// <param name="embeddings_metadata"> a dictionary which maps layer name to a file name in which metadata for this embedding layer is saved. See the details about metadata files format. In case if the same metadata file is used for all embedding layers, string can be passed.</param>
        public TensorBoard(string log_dir= "./logs", int histogram_freq= 0, bool write_graph= true, bool write_images= false, int? write_steps_per_second = null,
                    string update_freq = "epoch", int embeddings_freq= 0, Dictionary<string, string> embeddings_metadata= null)
        {
            Parameters["log_dir"] = log_dir;
            Parameters["histogram_freq"] = histogram_freq;
            Parameters["write_graph"] = write_graph;
            Parameters["write_images"] = write_images;
            Parameters["write_steps_per_second"] = write_steps_per_second;
            Parameters["update_freq"] = update_freq;
            Parameters["embeddings_freq"] = embeddings_freq;
            Parameters["embeddings_metadata"] = embeddings_metadata;
            
            PyInstance = Instance.keras.callbacks.TensorBoard;
            Init();
        }
    }

    /// <summary>
    /// Reduce learning rate when a metric has stopped improving.
    /// Models often benefit from reducing the learning rate by a factor of 2-10 once learning stagnates.
    /// This callback monitors a quantity and if no improvement is seen for a 'patience' number of epochs, the learning rate is reduced.
    /// </summary>
    /// <seealso cref="Keras.Callbacks.Callback" />
    public class ReduceLROnPlateau : Callback
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="LearningRateScheduler" /> class.
        /// </summary>
        /// <param name="monitor"> quantity to be monitored.</param>
        /// <param name="factor"> factor by which the learning rate will be reduced. new_lr = lr * factor</param>
        /// <param name="patience"> number of epochs with no improvement after which learning rate will be reduced.</param>
        /// <param name="verbose"> int. 0: quiet, 1: update messages.</param>
        /// <param name="mode"> one of {auto, min, max}. In min mode, lr will be reduced when the quantity monitored has stopped decreasing; in max mode it will be reduced when the quantity monitored has stopped increasing; in auto mode, the direction is automatically inferred from the name of the monitored quantity.</param>
        /// <param name="min_delta"> threshold for measuring the new optimum, to only focus on significant changes.</param>
        /// <param name="cooldown"> number of epochs to wait before resuming normal operation after lr has been reduced.</param>
        /// <param name="min_lr"> lower bound on the learning rate.</param>

        public ReduceLROnPlateau(string monitor= "val_loss", float factor= 0.1f, int patience= 10, int verbose= 0, string mode= "auto", float min_delta= 0.0001f, int cooldown= 0, float min_lr= 0)
        {
            Parameters["monitor"] = monitor;
            Parameters["factor"] = factor;
            Parameters["patience"] = patience;
            Parameters["verbose"] = verbose;
            Parameters["mode"] = mode;
            Parameters["min_delta"] = min_delta;
            Parameters["cooldown"] = cooldown;
            Parameters["min_lr"] = min_lr;

            PyInstance = Instance.keras.callbacks.ReduceLROnPlateau;
            Init();
        }
    }

    /// <summary>
    /// Callback that streams epoch results to a csv file.
    /// Supports all values that can be represented as a string, including 1D iterables such as np.ndarray.
    /// </summary>
    /// <seealso cref="Keras.Callbacks.Callback" />
    public class CSVLogger : Callback
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="LearningRateScheduler" /> class.
        /// </summary>
        /// <param name="filename">filename of the csv file, e.g. 'run/log.csv'.</param>
        /// <param name="separator">string used to separate elements in the csv file.</param>
        /// <param name="append">True: append if file exists (useful for continuing training). False: overwrite existing file,</param>
        public CSVLogger(string filename, string separator = ",", bool append = false)
        {
            Parameters["filename"] = filename;
            Parameters["separator"] = separator;
            Parameters["append"] = append;

            PyInstance = Instance.keras.callbacks.CSVLogger;
            Init();
        }
    }

    /// <summary>
    /// BackupAndRestore callback is intended to recover training from an interruption that has happened in the middle of a Model.fit execution, 
    /// by backing up the training states in a temporary checkpoint file (with the help of a tf.train.CheckpointManager), at the end of each epoch. 
    /// </summary>
    public class BackupAndRestore : Callback
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="BackupAndRestore" /> class.
        /// </summary>
        /// <param name="backup_dir">String, path to store the checkpoint. e.g. backup_dir = os.path.join(working_dir, 'backup')</param>
        /// <param name="save_freq">'epoch', integer, or False. When set to 'epoch' the callback saves the checkpoint at the end of each epoch</param>
        /// <param name="delete_checkpoint">Boolean, default to True. This BackupAndRestore callback works by saving a checkpoint to back up the training state</param>
        /// <param name="save_before_preemption">A boolean value instructing whether to turn on the automatic checkpoint saving for preemption/maintenance events. </param>
        public BackupAndRestore(string backup_dir, string save_freq = "epoch", bool delete_checkpoint = true, bool save_before_preemption = false)
        {
            Parameters["backup_dir"] = backup_dir;
            Parameters["save_freq"] = save_freq;
            Parameters["delete_checkpoint"] = delete_checkpoint;
            Parameters["save_before_preemption"] = save_before_preemption;

            PyInstance = Instance.keras.callbacks.BackupAndRestore;
            Init();
        }
    }

    /// <summary>
    /// Container abstracting a list of callbacks.
    /// </summary>
    public class CallbackList : Callback
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="CallbackList" /> class.
        /// </summary>
        /// <param name="callbacks">List of Callback instances.</param>
        /// <param name="add_history">Whether a History callback should be added, if one does not already exist in the callbacks list.</param>
        /// <param name="add_progbar">Whether a ProgbarLogger callback should be added, if one does not already exist in the callbacks list.</param>
        /// <param name="model">The Model these callbacks are used with.</param>
        public CallbackList(List<Callback> callbacks, bool add_history = false, bool add_progbar = false, BaseModel model = null)
        {
            Parameters["callbacks"] = callbacks;
            Parameters["add_history"] = add_history;
            Parameters["add_progbar"] = add_progbar;
            Parameters["model"] = model.ToPython();

            PyInstance = Instance.keras.callbacks.CallbackList;
            Init();
        }
    }
}
