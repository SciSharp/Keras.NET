using Keras.Layers;
using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;

namespace Keras.Callbacks
{
    public class Callback : Base
    {
        public Callback()
        {
            __self__ = keras.callbacks.Callback;
        }
    }

    public class BaseLogger : Callback
    {
        public BaseLogger(params string[] stateful_metrics)
        {
            Parameters["stateful_metrics"] = stateful_metrics!=null ? stateful_metrics.ToList() : null;
            __self__ = keras.callbacks.BaseLogger;
        }
    }

    public class TerminateOnNaN : Callback
    {
        public TerminateOnNaN()
        {
            __self__ = keras.callbacks.TerminateOnNaN;
        }
    }

    public class ProgbarLogger : Callback
    {
        public ProgbarLogger(string count_mode = "samples", params string[] stateful_metrics)
        {
            Parameters["count_mode"] = count_mode;
            Parameters["stateful_metrics"] = stateful_metrics != null ? stateful_metrics.ToList() : null;
            __self__ = keras.callbacks.ProgbarLogger;
        }
    }

    public class History : Callback
    {
        public History()
        {
            __self__ = keras.callbacks.History;
        }
    }

    public class ModelCheckpoint : Callback
    {
        public ModelCheckpoint(string filepath, string monitor = "val_loss", int verbose = 0, bool save_best_only = true
                    , bool save_weights_only = false, string mode = "auto", int period = 1)
        {
            Parameters["filepath"] = filepath;
            Parameters["monitor"] = monitor;
            Parameters["verbose"] = verbose;
            Parameters["save_best_only"] = save_best_only;
            Parameters["save_weights_only"] = save_weights_only;
            Parameters["mode"] = mode;
            Parameters["period"] = period;

            __self__ = keras.callbacks.ModelCheckpoint;
        }
    }

    public class EarlyStopping : Callback
    {
        public EarlyStopping(string monitor = "val_loss", float min_delta = 0, int patience = 0, int verbose = 0, string mode = "auto", float? baseline = null, bool restore_best_weights = false)
        {
            Parameters["monitor"] = monitor;
            Parameters["min_delta"] = min_delta;
            Parameters["patience"] = patience;
            Parameters["verbose"] = verbose;
            Parameters["mode"] = mode;
            Parameters["baseline"] = baseline;
            Parameters["restore_best_weights"] = restore_best_weights;

            __self__ = keras.callbacks.EarlyStopping;
        }
    }

    public class RemoteMonitor : Callback
    {
        public RemoteMonitor(string root = "http://localhost:9000", string path = "/publish/epoch/end/", string field = "data", Dictionary<string, string> headers = null, bool send_as_json = false)
        {
            Parameters["root"] = root;
            Parameters["path"] = path;
            Parameters["field"] = field;
            Parameters["headers"] = headers;
            Parameters["send_as_json"] = send_as_json;

            __self__ = keras.callbacks.RemoteMonitor;
        }
    }

    public class LearningRateScheduler : Callback
    {
        public LearningRateScheduler(object schedule, int verbose= 0)
        {
            Parameters["schedule"] = schedule;
            Parameters["verbose"] = verbose;

            __self__ = keras.callbacks.LearningRateScheduler;
        }
    }
}
