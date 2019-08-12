using Keras.Applications;
using Keras.Helper;
using Numpy;
using Python.Runtime;
using System;
using System.Collections.Generic;
using System.Text;

namespace Keras.PreProcessing.Image
{
    /// <summary>
    /// Generate batches of tensor image data with real-time data augmentation. The data will be looped over (in batches).
    /// </summary>
    public class ImageDataGenerator : Base
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="ImageDataGenerator"/> class.
        /// </summary>
        /// <param name="featurewise_center"> Boolean. Set input mean to 0 over the dataset, feature-wise.</param>
        /// <param name="samplewise_center"> Boolean. Set each sample mean to 0.</param>
        /// <param name="featurewise_std_normalization"> Boolean. Divide inputs by std of the dataset, feature-wise.</param>
        /// <param name="samplewise_std_normalization"> Boolean. Divide each input by its std.</param>
        /// <param name="zca_epsilon"> epsilon for ZCA whitening. Default is 1e-6.</param>
        /// <param name="zca_whitening"> Boolean. Apply ZCA whitening.</param>
        /// <param name="rotation_range"> Int. Degree range for random rotations.</param>
        /// <param name="width_shift_range"> Float, 1-D array-like or int </param>
        /// <param name="height_shift_range"> Float, 1-D array-like or int</param>
        /// <param name="brightness_range"> Tuple or list of two floats. Range for picking a brightness shift value from.</param>
        /// <param name="shear_range"> Float. Shear Intensity (Shear angle in counter-clockwise direction in degrees)</param>
        /// <param name="zoom_range"> Float or [lower, upper]. Range for random zoom. If a float, [lower, upper] = [1-zoom_range, 1+zoom_range].</param>
        /// <param name="channel_shift_range"> Float. Range for random channel shifts.</param>
        /// <param name="fill_mode"> One of {"constant", "nearest", "reflect" or "wrap"}. Default is 'nearest'. Points outside the boundaries of the input are filled according to the given mode, 'constant'-> kkkkkkkk|abcd|kkkkkkkk (cval=k), 'nearest'-> aaaaaaaa|abcd|dddddddd, 'reflect'-> abcddcba|abcd|dcbaabcd, 'wrap'-> abcdabcd|abcd|abcdabcd</param>
        /// <param name="cval"> Float or Int. Value used for points outside the boundaries when fill_mode = "constant".</param>
        /// <param name="horizontal_flip"> Boolean. Randomly flip inputs horizontally.</param>
        /// <param name="vertical_flip"> Boolean. Randomly flip inputs vertically.</param>
        /// <param name="rescale"> rescaling factor. Defaults to None. If None or 0, no rescaling is applied, otherwise we multiply the data by the value provided (after applying all other transformations).</param>
        /// <param name="preprocessing_function"> function that will be implied on each input. The function will run after the image is resized and augmented. The function should take one argument: one image (Numpy tensor with rank 3), and should output a Numpy tensor with the same shape.</param>
        /// <param name="data_format"> Image data format, either "channels_first" or "channels_last". "channels_last" mode means that the images should have shape  (samples, height, width, channels), "channels_first" mode means that the images should have shape  (samples, channels, height, width). It defaults to the image_data_format value found in your Keras config file at ~/.keras/keras.json. If you never set it, then it will be "channels_last".</param>
        /// <param name="validation_split"> Float. Fraction of images reserved for validation (strictly between 0 and 1).</param>
        /// <param name="dtype"> Dtype to use for the generated arrays.</param>
        public ImageDataGenerator(bool featurewise_center = false, bool samplewise_center = false, bool featurewise_std_normalization = false,
                                    bool samplewise_std_normalization = false, bool zca_whitening = false, float zca_epsilon = 1e-06f, int rotation_range = 0,
                                    float width_shift_range = 0.0f, float height_shift_range = 0.0f, float[] brightness_range = null, float shear_range = 0.0f,
                                    float zoom_range = 0.0f, float channel_shift_range = 0.0f, string fill_mode = "nearest", float cval = 0.0f,
                                    bool horizontal_flip = false, bool vertical_flip = false, float? rescale = null, PyObject preprocessing_function = null,
                                    string data_format = "", float validation_split = 0.0f, string dtype = "")
        {
            Parameters["featurewise_center"] = featurewise_center;
            Parameters["samplewise_center"] = samplewise_center;
            Parameters["featurewise_std_normalization"] = featurewise_std_normalization;
            Parameters["samplewise_std_normalization"] = samplewise_std_normalization;
            Parameters["zca_whitening"] = zca_whitening;
            Parameters["zca_epsilon"] = zca_epsilon;
            Parameters["rotation_range"] = rotation_range;
            Parameters["width_shift_range"] = width_shift_range;
            Parameters["height_shift_range"] = height_shift_range;
            Parameters["brightness_range"] = brightness_range;
            Parameters["shear_range"] = shear_range;
            Parameters["zoom_range"] = zoom_range;
            Parameters["channel_shift_range"] = channel_shift_range;
            Parameters["fill_mode"] = fill_mode;
            Parameters["cval"] = cval;
            Parameters["horizontal_flip"] = horizontal_flip;
            Parameters["vertical_flip"] = vertical_flip;
            Parameters["rescale"] = rescale;
            Parameters["preprocessing_function"] = preprocessing_function;
            Parameters["data_format"] = data_format;
            Parameters["validation_split"] = validation_split;
            Parameters["dtype"] = dtype;

            PyInstance = Instance.keras.preprocessing.image.ImageDataGenerator;
            Init();
        }

        /// <summary>
        /// Applies the transform.
        /// </summary>
        /// <param name="x">3D tensor, single image.</param>
        /// <param name="transform_parameters">Dictionary with string - parameter pairs describing the transformation. Currently, the following parameters from the dictionary are used:
        /// <list type="bullet"><item>'theta': Float.Rotation angle in degrees.</item><item>'tx': Float.Shift in the x direction.</item><item>'ty': Float.Shift in the y direction.</item><item>'shear': Float.Shear angle in degrees.</item><item>'zx': Float.Zoom in the x direction.</item><item>'zy': Float.Zoom in the y direction.</item><item>'flip_horizontal': Boolean.Horizontal flip.</item><item>'flip_vertical': Boolean.Vertical flip.</item><item>'channel_shift_intencity': Float.Channel shift intensity.</item><item>'brightness': Float.Brightness shift intensity.</item></list></param>
        /// <returns>A transformed version of the input (same shape).</returns>
        public NDarray ApplyTransform(NDarray x, Dictionary<string, object> transform_parameters)
        {
            Dictionary<string, object> parameters = new Dictionary<string, object>();
            parameters["x"] = x;
            parameters["transform_parameters"] = transform_parameters;

            return new NDarray(InvokeMethod("apply_transform", parameters));
        }

        /// <summary>
        /// Fits the data generator to some sample data.
        /// This computes the internal data stats related to the data-dependent transformations, based on an array of sample data.
        /// Only required if featurewise_center or featurewise_std_normalization or zca_whitening are set to True.
        /// </summary>
        /// <param name="x">The x.</param>
        /// <param name="augment">if set to <c>true</c> [augment].</param>
        /// <param name="rounds">The rounds.</param>
        /// <param name="seed">The seed.</param>
        public void Fit(NDarray x, bool augment = false, int rounds = 1, int? seed = null)
        {
            Dictionary<string, object> parameters = new Dictionary<string, object>();
            parameters["x"] = x;
            parameters["augment"] = augment;
            parameters["rounds"] = rounds;
            parameters["seed"] = seed;

            InvokeMethod("fit", parameters);
        }

        /// <summary>
        /// Flows the specified x.
        /// </summary>
        /// <param name="x"> Input data. Numpy array of rank 4 or a tuple. If tuple, the first element should contain the images and the second element another numpy array or a list of numpy arrays that gets passed to the output without any modifications. Can be used to feed the model miscellaneous data along with the images. In case of grayscale data, the channels axis of the image array should have value 1, in case of RGB data, it should have value 3, and in case of RGBA data, it should have value 4.</param>
        /// <param name="y"> Labels.</param>
        /// <param name="batch_size"> Int (default: 32).</param>
        /// <param name="shuffle"> Boolean (default: True).</param>
        /// <param name="sample_weight"> Sample weights.</param>
        /// <param name="seed"> Int (default: None).</param>
        /// <param name="save_to_dir"> None or str (default: None). This allows you to optionally specify a directory to which to save the augmented pictures being generated (useful for visualizing what you are doing).</param>
        /// <param name="save_prefix"> Str (default: ''). Prefix to use for filenames of saved pictures (only relevant if save_to_dir is set).</param>
        /// <param name="save_format"> one of "png", "jpeg" (only relevant if save_to_dir is set). Default: "png".</param>
        /// <param name="subset"> Subset of data ("training" or "validation") if validation_split is set in ImageDataGenerator.</param>
        /// <returns>An Iterator yielding tuples of (x, y) where x is a numpy array of image data (in the case of a single image input) or a list of numpy arrays (in the case with additional inputs) and y is a numpy array of corresponding labels. If 'sample_weight' is not None, the yielded tuples are of the form (x, y, sample_weight). If y is None, only the numpy array x is returned.</returns>
        public KerasIterator Flow(NDarray x, NDarray y= null, int batch_size= 32, bool shuffle= true, NDarray sample_weight= null, int? seed= null, 
                            string save_to_dir= "", string save_prefix= "", string save_format= "png", string subset= "")
        {
            Dictionary<string, object> parameters = new Dictionary<string, object>();
            parameters["x"] = x;
            parameters["y"] = y;
            parameters["batch_size"] = batch_size;
            parameters["shuffle"] = shuffle;
            parameters["sample_weight"] = sample_weight;
            parameters["seed"] = seed;
            parameters["save_to_dir"] = save_to_dir;
            parameters["save_prefix"] = save_prefix;
            parameters["save_format"] = save_format;
            parameters["subset"] = subset;

            return new KerasIterator(InvokeMethod("flow", parameters));
        }

        ///// <summary>
        ///// Flows from dataframe.
        ///// </summary>
        ///// <param name="directory"> string, path to the target directory. It should contain one subdirectory per class. Any PNG, JPG, BMP, PPM or TIF images inside each of the subdirectories directory tree will be included in the generator. See this script for more details.</param>
        ///// <param name="target_size"> Tuple of integers (height, width), default: (256, 256). The dimensions to which all images found will be resized.</param>
        ///// <param name="color_mode"> One of "grayscale", "rgb", "rgba". Default: "rgb". Whether the images will be converted to have 1, 3, or 4 channels.</param>
        ///// <param name="classes"> Optional list of class subdirectories (e.g. ['dogs', 'cats']). Default: None. If not provided, the list of classes will be automatically inferred from the subdirectory names/structure under directory, where each subdirectory will be treated as a different class (and the order of the classes, which will map to the label indices, will be alphanumeric). The dictionary containing the mapping from class names to class indices can be obtained via the attribute class_indices.</param>
        ///// <param name="class_mode"> One of "categorical", "binary", "sparse", "input", or None. Default: "categorical". Determines the type of label arrays that are returned:</param>
        ///// <param name="batch_size"> Size of the batches of data (default: 32).</param>
        ///// <param name="shuffle"> Whether to shuffle the data (default: True) If set to False, sorts the data in alphanumeric order.</param>
        ///// <param name="seed"> Optional random seed for shuffling and transformations.</param>
        ///// <param name="save_to_dir"> None or str (default: None). This allows you to optionally specify a directory to which to save the augmented pictures being generated (useful for visualizing what you are doing).</param>
        ///// <param name="save_prefix"> Str. Prefix to use for filenames of saved pictures (only relevant if save_to_dir is set).</param>
        ///// <param name="save_format"> One of "png", "jpeg" (only relevant if save_to_dir is set). Default: "png".</param>
        ///// <param name="follow_links"> Whether to follow symlinks inside class subdirectories (default: False).</param>
        ///// <param name="subset"> Subset of data ("training" or "validation") if validation_split is set in ImageDataGenerator.</param>
        ///// <param name="interpolation"> Interpolation method used to resample the image if the target size is different from that of the loaded image. Supported methods are "nearest", "bilinear", and "bicubic". If PIL version 1.1.3 or newer is installed, "lanczos" is also supported. If PIL version 3.4.0 or newer is installed, "box" and "hamming" are also supported. By default, "nearest" is used.</param>

        ///// <returns>A DirectoryIterator yielding tuples of (x, y) where x is a numpy array containing a batch of images with shape (batch_size, *target_size, channels) and y is a numpy array of corresponding labels.</returns>
        //public KerasIterator FlowFromDataframe(string directory, Tuple<int, int> target_size = null, string color_mode = "rgb", string[] classes = null, string class_mode = "categorical",
        //                                        int batch_size = 32, bool shuffle = true, int? seed = null, string save_to_dir = "", string save_prefix = "", string save_format = "png",
        //                                        bool follow_links = false, string subset = "", string interpolation = "nearest")
        //{
        //    Dictionary<string, object> parameters = new Dictionary<string, object>();
        //    parameters["directory"] = directory;
        //    parameters["target_size"] = target_size == null ? new Shape(256, 256) : new Shape(target_size.Item1, target_size.Item2);
        //    parameters["color_mode"] = color_mode;
        //    parameters["classes"] = classes;
        //    parameters["class_mode"] = class_mode;
        //    parameters["batch_size"] = batch_size;
        //    parameters["shuffle"] = shuffle;
        //    parameters["save_to_dir"] = save_to_dir;
        //    parameters["save_format"] = save_format;
        //    parameters["follow_links"] = follow_links;
        //    parameters["subset"] = subset;
        //    parameters["interpolation"] = interpolation;

        //    return new KerasIterator(InvokeMethod("flow_from_dataframe", parameters));
        //}

        /// <summary>
        /// Flows from dataframe.
        /// </summary>
        /// <param name="directory"> string, path to the target directory. It should contain one subdirectory per class. Any PNG, JPG, BMP, PPM or TIF images inside each of the subdirectories directory tree will be included in the generator. See this script for more details.</param>
        /// <param name="target_size"> Tuple of integers (height, width), default: (256, 256). The dimensions to which all images found will be resized.</param>
        /// <param name="color_mode"> One of "grayscale", "rgb", "rgba". Default: "rgb". Whether the images will be converted to have 1, 3, or 4 channels.</param>
        /// <param name="classes"> Optional list of class subdirectories (e.g. ['dogs', 'cats']). Default: None. If not provided, the list of classes will be automatically inferred from the subdirectory names/structure under directory, where each subdirectory will be treated as a different class (and the order of the classes, which will map to the label indices, will be alphanumeric). The dictionary containing the mapping from class names to class indices can be obtained via the attribute class_indices.</param>
        /// <param name="class_mode"> One of "categorical", "binary", "sparse", "input", or None. Default: "categorical". Determines the type of label arrays that are returned:</param>
        /// <param name="batch_size"> Size of the batches of data (default: 32).</param>
        /// <param name="shuffle"> Whether to shuffle the data (default: True) If set to False, sorts the data in alphanumeric order.</param>
        /// <param name="seed"> Optional random seed for shuffling and transformations.</param>
        /// <param name="save_to_dir"> None or str (default: None). This allows you to optionally specify a directory to which to save the augmented pictures being generated (useful for visualizing what you are doing).</param>
        /// <param name="save_prefix"> Str. Prefix to use for filenames of saved pictures (only relevant if save_to_dir is set).</param>
        /// <param name="save_format"> One of "png", "jpeg" (only relevant if save_to_dir is set). Default: "png".</param>
        /// <param name="follow_links"> Whether to follow symlinks inside class subdirectories (default: False).</param>
        /// <param name="subset"> Subset of data ("training" or "validation") if validation_split is set in ImageDataGenerator.</param>
        /// <param name="interpolation"> Interpolation method used to resample the image if the target size is different from that of the loaded image. Supported methods are "nearest", "bilinear", and "bicubic". If PIL version 1.1.3 or newer is installed, "lanczos" is also supported. If PIL version 3.4.0 or newer is installed, "box" and "hamming" are also supported. By default, "nearest" is used.</param>
        /// <returns>A DirectoryIterator yielding tuples of (x, y) where x is a numpy array containing a batch of images with shape (batch_size, *target_size, channels) and y is a numpy array of corresponding labels.</returns>
        public KerasIterator FlowFromDirectory(string directory, Tuple<int, int> target_size = null, string color_mode = "rgb", string[] classes = null, string class_mode = "categorical",
                                                int batch_size = 32, bool shuffle = true, int? seed = null, string save_to_dir = "", string save_prefix = "", string save_format = "png",
                                                bool follow_links = false, string subset = "", string interpolation = "nearest")
        {
            Dictionary<string, object> parameters = new Dictionary<string, object>();
            parameters["directory"] = directory;
            parameters["target_size"] = target_size == null ? new Shape(256, 256) : new Shape(target_size.Item1, target_size.Item2);
            parameters["color_mode"] = color_mode;
            parameters["classes"] = classes;
            parameters["class_mode"] = class_mode;
            parameters["batch_size"] = batch_size;
            parameters["shuffle"] = shuffle;
            parameters["save_to_dir"] = save_to_dir;
            parameters["save_format"] = save_format;
            parameters["follow_links"] = follow_links;
            parameters["subset"] = subset;
            parameters["interpolation"] = interpolation;

            return new KerasIterator(InvokeMethod("flow_from_directory", parameters));
        }
    }

    public class ImageUtil : Base
    {
        static dynamic caller = Instance.keras.preprocessing.image;
        /// <summary>
        /// Array to img.
        /// </summary>
        /// <param name="x">The x.</param>
        /// <param name="data_format">The data format.</param>
        /// <param name="scale">if set to <c>true</c> [scale].</param>
        /// <param name="dtype">The dtype.</param>
        /// <returns></returns>
        public static dynamic ArrayToImg(NDarray x, string data_format = "", bool scale = true, string dtype = "")
        {
            Dictionary<string, object> parameters = new Dictionary<string, object>();
            parameters["x"] = x;
            parameters["data_format"] = data_format;
            parameters["scale"] = scale;
            parameters["dtype"] = dtype;

            return InvokeStaticMethod(caller, "array_to_img", parameters);
        }

        /// <summary>
        /// Images to array.
        /// </summary>
        /// <param name="image">The image.</param>
        /// <param name="data_format">The data format.</param>
        /// <param name="dtype">The dtype.</param>
        /// <returns></returns>
        public static NDarray ImageToArray(dynamic image, string data_format = "", string dtype = "")
        {
            Dictionary<string, object> parameters = new Dictionary<string, object>();
            parameters["image"] = image;
            parameters["data_format"] = data_format;
            parameters["dtype"] = dtype;

            return new NDarray((PyObject)InvokeStaticMethod(caller, "img_to_array", parameters));
        }

        /// <summary>
        /// Loads the img.
        /// </summary>
        /// <param name="path">The path.</param>
        /// <param name="color_mode">The color mode.</param>
        /// <param name="target_size">Size of the target.</param>
        /// <param name="interpolation">The interpolation.</param>
        /// <returns></returns>
        public static dynamic LoadImg(string path, string color_mode = "rgb", Shape target_size=null, string interpolation = "nearest")
        {
            Dictionary<string, object> parameters = new Dictionary<string, object>();
            parameters["path"] = path;
            parameters["color_mode"] = color_mode;
            parameters["target_size"] = target_size;
            parameters["interpolation"] = interpolation;

            return InvokeStaticMethod(caller, "load_img", parameters);
        }

        /// <summary>
        /// Decodes the predictions.
        /// </summary>
        /// <param name="preds">The preds.</param>
        /// <param name="top">The top.</param>
        /// <returns></returns>
        public static ImageNetPrediction[] DecodePredictions(NDarray preds, int top = 3)
        {
            Dictionary<string, object> parameters = new Dictionary<string, object>();
            parameters["preds"] = preds;
            parameters["top"] = top;
            var predobj = (PyObject)InvokeStaticMethod(Instance.keras.applications.resnet50, "decode_predictions", parameters);
            var d = predobj.ToString();
            var list = TupleSolver.TupleToList<object>(predobj);
            return null;
        }

        /// <summary>
        /// Preprocesses the input.
        /// </summary>
        /// <param name="x">The input tensor.</param>
        /// <returns></returns>
        public static NDarray PreprocessInput(NDarray x, string data_format = "channels_last", string mode= "caffe")
        {
            Dictionary<string, object> parameters = new Dictionary<string, object>();
            parameters["x"] = x;
            parameters["data_format"] = data_format;
            parameters["mode"] = mode;
            return new NDarray((PyObject)InvokeStaticMethod(Instance.keras.applications.resnet50, "preprocess_input", parameters));
        }
    }
}
