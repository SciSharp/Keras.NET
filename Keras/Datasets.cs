using Numpy;
using Python.Runtime;
using System;
using System.Collections.Generic;
using System.Text;

namespace Keras.Datasets
{
    public class BostonHousing : Base
    {
        public static void LoadData(string path = "boston_housing.npz", float test_split = 0.2f, int seed = 113)
        {
            Instance.self.datasets.boston_housing.load_data(path: path, test_split: test_split, seed: seed);
        }
    }

    public class Cifar10 : Base
    {
        public static void LoadData()
        {
            Instance.self.datasets.cifar10.load_data();
        }
    }

    public class Cifar100 : Base
    {
        public static void LoadData(string label_mode = "fine")
        {
            Instance.self.datasets.cifar10.load_data(label_mode: label_mode);
        }
    }

    public class FashionMNIST : Base
    {
        public static void LoadData()
        {
            Instance.self.datasets.fashion_mnist.load_data();
        }
    }

    public class MNIST : Base
    {
        public static void LoadData(string path = "mnist.npz")
        {
            var d = new PyTuple(Instance.self.datasets.mnist.load_data(path: path)).As<Tuple<List<NDarray>>>();//.As<Tuple<Tuple<NDarray<byte>, NDarray<byte>>, Tuple<NDarray<byte>, NDarray<byte>>>>();
        }
    }

    public class IMDB : Base
    {
        public static void LoadData(string path= "imdb.npz", int? num_words= null, int skip_top= 0, int? maxlen= null, int seed= 113,
                                int start_char= 1, int oov_char= 2, int index_from= 3)
        {
            Instance.self.datasets.imdb.load_data(path: path, num_words: num_words, skip_top: skip_top, maxlen: maxlen, seed: seed, start_char: start_char,
                                                oov_char: oov_char, index_from: index_from);
        }
    }

    public class Reuters : Base
    {
        public static void LoadData(string path = "reuters.npz", int? num_words = null, int skip_top = 0, int? maxlen = null, float test_split = 0.2f,
                                int seed = 113, int start_char = 1, int oov_char = 2, int index_from = 3)
        {
            Instance.self.datasets.reuters.load_data(path: path, num_words: num_words, skip_top: skip_top, maxlen: maxlen, test_split: test_split, seed: seed, start_char: start_char,
                                                oov_char: oov_char, index_from: index_from);
        }
    }
}
