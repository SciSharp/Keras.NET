using Numpy;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Keras.Models
{
    public class BaseModel : Base
    {
        public void Compile(string optimizer, string loss, string[] metrics = null, float[] loss_weights = null,
                       string sample_weight_mode = "None", string[] weighted_metrics = null, NDarray[] target_tensors = null)
        {
            var args = new Dictionary<string, object>();
            args["optimizer"] = optimizer;
            args["loss"] = loss;
            args["metrics"] = metrics != null ? metrics : null;
            args["loss_weights"] = loss_weights != null ? loss_weights : null;
            args["sample_weight_mode"] = sample_weight_mode != null ? sample_weight_mode : null;
            args["weighted_metrics"] = weighted_metrics != null ? weighted_metrics : null;
            args["target_tensors"] = target_tensors != null ? target_tensors : null;

            InvokeMethod("compile", args);
        }

        public void Fit(NDarray x, NDarray y, int? batch_size = null, int epochs = 1, int verbose = 1, Callback[] callbacks = null,
                        float validation_split = 0.0f, NDarray[] validation_data = null, bool shuffle = true, Dictionary<int, float> class_weight = null,
                        NDarray sample_weight = null, int initial_epoch = 0, int? steps_per_epoch = null, int? validation_steps = null, int validation_freq = 1)
        {
            var args = new Dictionary<string, object>();
            args["x"] = x;
            args["y"] = y;
            args["batch_size"] = batch_size;
            args["epochs"] = epochs;
            args["verbose"] = verbose;
            args["callbacks"] = callbacks != null ? callbacks : null;
            args["validation_split"] = validation_split;
            if (validation_data != null)
            {
                if (validation_data.Length == 2)
                    args["validation_data"] = (validation_data[0], validation_data[1]);
                else if (validation_data.Length == 3)
                    args["validation_data"] = (validation_data[0], validation_data[1], validation_data[2]);
            }

            args["shuffle"] = shuffle;
            args["class_weight"] = class_weight;
            args["sample_weight"] = sample_weight;
            args["initial_epoch"] = initial_epoch;
            args["steps_per_epoch"] = steps_per_epoch;
            args["validation_steps"] = validation_steps;
            args["validation_freq"] = validation_freq;

            InvokeMethod("fit", args);
        }

        public void Evaluate(NDarray x, NDarray y, int? batch_size = null, int verbose = 1, NDarray sample_weight = null, int? steps = null, Callback[] callbacks = null)
        {
            var args = new Dictionary<string, object>();
            args["x"] = x;
            args["y"] = y;
            args["batch_size"] = batch_size;
            args["verbose"] = verbose;
            args["sample_weight"] = sample_weight;
            args["steps"] = steps;
            args["callbacks"] = callbacks != null ? callbacks : null;

            InvokeMethod("evaluate", args);
        }

        public void Predict(NDarray x, NDarray y, int verbose = 1, int? steps = null, Callback[] callbacks = null)
        {
            var args = new Dictionary<string, object>();
            args["x"] = x;
            args["y"] = y;
            args["verbose"] = verbose;
            args["steps"] = steps;
            args["callbacks"] = callbacks != null ? callbacks : null;

            InvokeMethod("predict", args);
        }

        public void TrainOnBatch(NDarray x, NDarray y, NDarray sample_weight = null, Dictionary<int, float> class_weight = null)
        {
            var args = new Dictionary<string, object>();
            args["x"] = x;
            args["y"] = y;
            args["sample_weight"] = sample_weight;
            args["class_weight"] = class_weight;

            InvokeMethod("train_on_batch", args);
        }

        public void TestOnBatch(NDarray x, NDarray y, NDarray sample_weight = null)
        {
            var args = new Dictionary<string, object>();
            args["x"] = x;
            args["y"] = y;
            args["sample_weight"] = sample_weight;

            InvokeMethod("train_on_batch", args);
        }

        public void PredictOnBatch(NDarray x)
        {
            var args = new Dictionary<string, object>();
            args["x"] = x;

            InvokeMethod("train_on_batch", args);
        }

        //ToDo: Implement Generators
    }
}
