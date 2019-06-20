using Keras.Helper;
using Numpy;
using Python.Runtime;
using System;
using System.Collections.Generic;
using System.Text;

namespace Keras.Datasets
{
    public class BostonHousing : Base
    {
        public static ((NDarray, NDarray), (NDarray, NDarray)) LoadData(string path = "boston_housing.npz", float test_split = 0.2f, int seed = 113)
        {
            var dlist = TupleSolver.TupleToList(Instance.keras.datasets.boston_housing.load_data(path: path, test_split: test_split, seed: seed));
            return ((dlist[0], dlist[1]), (dlist[2], dlist[3]));
        }
    }

    public class Cifar10 : Base
    {
        public static ((NDarray, NDarray), (NDarray, NDarray)) LoadData()
        {
            var dlist = TupleSolver.TupleToList(Instance.keras.datasets.cifar10.load_data());
            return ((dlist[0], dlist[1]), (dlist[2], dlist[3]));
        }
    }

    public class Cifar100 : Base
    {
        public static ((NDarray, NDarray), (NDarray, NDarray)) LoadData(string label_mode = "fine")
        {
            var dlist = TupleSolver.TupleToList(Instance.keras.datasets.cifar10.load_data(label_mode: label_mode));
            return ((dlist[0], dlist[1]), (dlist[2], dlist[3]));
        }
    }

    public class FashionMNIST : Base
    {
        public static ((NDarray, NDarray), (NDarray, NDarray)) LoadData()
        {
            var dlist = TupleSolver.TupleToList(Instance.keras.datasets.fashion_mnist.load_data());
            return ((dlist[0], dlist[1]), (dlist[2], dlist[3]));
        }
    }

    public class MNIST : Base
    {
        public static ((NDarray, NDarray), (NDarray, NDarray)) LoadData(string path = "mnist.npz")
        {
            var dlist = TupleSolver.TupleToList(Instance.keras.datasets.mnist.load_data(path: path));
            return ((dlist[0], dlist[1]), (dlist[2], dlist[3]));
        }
    }

    public class IMDB : Base
    {
        public static ((NDarray, NDarray), (NDarray, NDarray)) LoadData(string path= "imdb.npz", int? num_words= null, int skip_top= 0, int? maxlen= null, int seed= 113,
                                int start_char= 1, int oov_char= 2, int index_from= 3)
        {
            var dlist = TupleSolver.TupleToList(Instance.keras.datasets.imdb.load_data(path: path, num_words: num_words, skip_top: skip_top, maxlen: maxlen, seed: seed, start_char: start_char,
                                                oov_char: oov_char, index_from: index_from));
            return ((dlist[0], dlist[1]), (dlist[2], dlist[3]));
        }
    }

    public class Reuters : Base
    {
        public static ((NDarray, NDarray), (NDarray, NDarray)) LoadData(string path = "reuters.npz", int? num_words = null, int skip_top = 0, int? maxlen = null, float test_split = 0.2f,
                                int seed = 113, int start_char = 1, int oov_char = 2, int index_from = 3)
        {
            var dlist = TupleSolver.TupleToList(Instance.keras.datasets.reuters.load_data(path: path, num_words: num_words, skip_top: skip_top, maxlen: maxlen, test_split: test_split, seed: seed, start_char: start_char,
                                                oov_char: oov_char, index_from: index_from));
            return ((dlist[0], dlist[1]), (dlist[2], dlist[3]));
        }
    }
}
