using Keras.Utils;
using Numpy;
using Numpy.Models;
using Python.Runtime;
using System;
using System.Collections.Generic;
using System.Text;

namespace Keras.PreProcessing.Text
{
    public class Tokenizer : Base
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="Tokenizer" /> class.
        /// </summary>
        /// <param name="num_words"> the maximum number of words to keep, based on word frequency. Only the most common num_words-1 words will be kept.</param>
        /// <param name="filters"> a string where each element is a character that will be filtered from the texts. The default is all punctuation, plus tabs and line breaks, minus the ' character.</param>
        /// <param name="lower"> boolean. Whether to convert the texts to lowercase.</param>
        /// <param name="split"> str. Separator for word splitting.</param>
        /// <param name="char_level"> if True, every character will be treated as a token.</param>
        /// <param name="oov_token"> if given, it will be added to word_index and used to replace out-of-vocabulary words during text_to_sequence calls</param>
        public Tokenizer(int? num_words= null, string filters= "!\"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n", bool lower= true, string split= " ", 
                                bool char_level= false, int? oov_token= null, int document_count= 0)
        {
            Parameters["num_words"] = num_words;
            Parameters["filters"] = filters;
            Parameters["lower"] = lower;
            Parameters["split"] = split;
            Parameters["char_level"] = char_level;
            Parameters["oov_token"] = oov_token;
            Parameters["document_count"] = document_count;

            PyInstance = Instance.keras.preprocessing.text.Tokenizer;
        }

        public void FitOnTexts(string[] texts)
        {
            Dictionary<string, object> parameters = new Dictionary<string, object>();
            parameters["texts"] = texts;

            InvokeMethod("fit_on_texts", parameters);
        }

        public void FitOnSequences(Sequence[] sequences)
        {
            Dictionary<string, object> parameters = new Dictionary<string, object>();
            parameters["sequences"] = sequences;

            InvokeMethod("fit_on_sequences", parameters);
        }

        public Sequence[] TextsToSequences(string[] texts)
        {
            Dictionary<string, object> parameters = new Dictionary<string, object>();
            parameters["texts"] = texts;

            PyList pyList = new PyList(InvokeMethod("texts_to_sequences", parameters));
            List<Sequence> result = new List<Sequence>();
            foreach (PyObject item in pyList)
            {
                result.Add(new Sequence(item));
            }

            return result.ToArray();
        }

        public string[] SequencesToTexts(Sequence[] sequences)
        {
            Dictionary<string, object> parameters = new Dictionary<string, object>();
            parameters["sequences"] = sequences;
            PyObject py = InvokeMethod("texts_to_sequences", parameters);
            return py.As<string[]>();
        }

        public Matrix TextsToMatrix(string[] texts, string mode = "binary")
        {
            Dictionary<string, object> parameters = new Dictionary<string, object>();
            parameters["texts"] = texts;
            parameters["mode"] = mode;

            PyObject py = InvokeMethod("texts_to_matrix", parameters);
            return new Matrix(py);
        }
    }

    public class TextUtil : Base
    {
        static dynamic caller = Instance.keras.preprocessing.text;

        /// <summary>
        /// Converts a text to a sequence of indexes in a fixed-size hashing space.
        /// </summary>
        /// <param name="text"> Input text (string).</param>
        /// <param name="n"> Dimension of the hashing space.</param>
        /// <param name="hash_function"> defaults to python hash function, can be 'md5' or any function that takes in input a string and returns a int. Note that 'hash' is not a stable hashing function, so it is not consistent across different runs, while 'md5' is a stable hashing function.</param>
        /// <param name="filters"> list (or concatenation) of characters to filter out, such as punctuation. Default: !"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n, includes basic punctuation, tabs, and newlines.</param>
        /// <param name="lower"> boolean. Whether to set the text to lowercase.</param>
        /// <param name="split"> str. Separator for word splitting.</param>
        /// <returns>
        /// A list of integer word indices (unicity non-guaranteed).
        /// 0 is a reserved index that won't be assigned to any word.
        /// Two or more words may be assigned to the same index, due to possible collisions by the hashing function.
        /// The probability of a collision is in relation to the dimension of the hashing space and the number of distinct objects.
        /// </returns>
        public static int[] HashingTrick(string text, int n, string hash_function= "", string filters= "!\"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n", bool lower= true, string split= " ")
        {
            Dictionary<string, object> parameters = new Dictionary<string, object>();
            parameters["text"] = text;
            parameters["n"] = n;
            parameters["hash_function"] = hash_function;
            parameters["filters"] = filters;
            parameters["lower"] = lower;
            parameters["split"] = split;

            return ((PyObject)InvokeStaticMethod(caller, "hashing_trick", parameters)).As<int[]>();
        }

        /// <summary>
        /// One-hot encodes a text into a list of word indexes of size n.
        /// This is a wrapper to the hashing_trick function using hash as the hashing function; unicity of word to index mapping non-guaranteed.
        /// </summary>
        /// <param name="text">The text.</param>
        /// <param name="n">The n.</param>
        /// <param name="filters">The filters.</param>
        /// <param name="lower">if set to <c>true</c> [lower].</param>
        /// <param name="split">The split.</param>
        /// <returns></returns>
        public static int[,] OneHot(string text, int n, string filters = "!\"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n", bool lower = true, string split = " ")
        {
            Dictionary<string, object> parameters = new Dictionary<string, object>();
            parameters["text"] = text;
            parameters["n"] = n;
            parameters["filters"] = filters;
            parameters["lower"] = lower;
            parameters["split"] = split;

            return ((PyObject)InvokeStaticMethod(caller, "one_hot", parameters)).As<int[,]>();
        }

        /// <summary>
        /// Texts to word sequence.
        /// </summary>
        /// <param name="text">The text.</param>
        /// <param name="filters">The filters.</param>
        /// <param name="lower">if set to <c>true</c> [lower].</param>
        /// <param name="split">The split.</param>
        /// <returns></returns>
        public static string[] TextToWordSequence(string text, string filters = "!\"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n", bool lower = true, string split = " ")
        {
            Dictionary<string, object> parameters = new Dictionary<string, object>();
            parameters["text"] = text;
            parameters["filters"] = filters;
            parameters["lower"] = lower;
            parameters["split"] = split;

            return ((PyObject)InvokeStaticMethod(caller, "text_to_word_sequence", parameters)).As<string[]>();
        }
    }
}
