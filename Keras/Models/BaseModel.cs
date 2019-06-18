using Numpy;
using Python.Runtime;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Keras.Models
{
    public class BaseModel : Base
    {
        public void Compile(StringOrInstance optimizer, string loss, string[] metrics = null, float[] loss_weights = null,
                       string sample_weight_mode = "None", string[] weighted_metrics = null, NDarray[] target_tensors = null)
        {
            var args = new Dictionary<string, object>();
            args["optimizer"] = optimizer.PyObject;
            args["loss"] = loss;
            args["metrics"] = metrics;
            args["loss_weights"] = loss_weights;
            args["sample_weight_mode"] = sample_weight_mode;
            args["weighted_metrics"] = weighted_metrics;
            args["target_tensors"] = target_tensors;

            InvokeMethod("compile", args);

            //__self__.compile(optimizer: optimizer, loss: loss, metrics: metrics!=null ? metrics.ToList() : null, loss_weights: loss_weights, sample_weight_mode: sample_weight_mode,
            //            weighted_metrics: weighted_metrics, target_tensors: target_tensors);
        }

        public void Fit(NDarray x, NDarray y, int? batch_size = null, int epochs = 1, int verbose = 1, Callback[] callbacks = null,
                        float validation_split = 0.0f, NDarray[] validation_data = null, bool shuffle = true, Dictionary<int, float> class_weight = null,
                        NDarray sample_weight = null, int initial_epoch = 0, int? steps_per_epoch = null, int? validation_steps = null)
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
                    args["validation_data"] = new NDarray[] { validation_data[0], validation_data[1] };
                else if (validation_data.Length == 3)
                    args["validation_data"] = new NDarray[] { validation_data[0], validation_data[1], validation_data[2] };
            }

            args["shuffle"] = shuffle;
            args["class_weight"] = class_weight;
            args["sample_weight"] = sample_weight;
            args["initial_epoch"] = initial_epoch;
            args["steps_per_epoch"] = steps_per_epoch;
            args["validation_steps"] = validation_steps;

            InvokeMethod("fit", args);
        }

        public double[] Evaluate(NDarray x, NDarray y, int? batch_size = null, int verbose = 1, NDarray sample_weight = null, int? steps = null, Callback[] callbacks = null)
        {
            var args = new Dictionary<string, object>();
            args["x"] = x.PyObject;
            args["y"] = y.PyObject;
            args["batch_size"] = batch_size;
            args["verbose"] = verbose;
            args["sample_weight"] = sample_weight;
            args["steps"] = steps;
            args["callbacks"] = callbacks != null ? callbacks : null;

            return InvokeMethod("evaluate", args)?.As<double[]>();
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

        public string ToJson()
        {
            return __self__.to_json().ToString();
        }

        public void SaveWeight(string path)
        {
            __self__.save_weights(path);
        }

        public void LoadWeight(string path)
        {
            __self__.load_weights(path);
        }

        public static BaseModel ModelFromJson(string json_string)
        {
            var model = new BaseModel();
            model.__self__ = Instance.self.models.model_from_json(json_string: json_string);

            return model;
        }

        public static BaseModel ModelFromYaml(string json_string)
        {
            var model = new BaseModel();
            model.__self__ = Instance.self.models.model_from_yaml(json_string: json_string);

            return model;
        }
    }
}
