using Keras.Utils;
using Numpy;
using Python.Runtime;
using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;

namespace Keras.PreProcessing.sequence
{
    /// <summary>
    /// Sequence class
    /// </summary>
    public class SequenceUtil : Base
    {
        static dynamic caller = Instance.keras.preprocessing.sequence;

        /// <summary>
        /// Pads sequences to the same length.
        /// This function transforms a list of num_samples sequences(lists of integers) into a 2D Numpy array of shape(num_samples, num_timesteps). num_timesteps is either the maxlen argument if provided, or the length of the longest sequence otherwise.
        /// Sequences that are shorter than num_timesteps are padded with value at the end.
        /// Sequences longer than num_timesteps are truncated so that they fit the desired length.The position where padding or truncation happens is determined by the arguments padding and truncating, respectively.
        /// Pre-padding is the default.
        /// </summary>
        /// <param name="sequences">The sequences.</param>
        /// <param name="maxlen">The maxlen.</param>
        /// <param name="dtype">The dtype.</param>
        /// <param name="padding">The padding.</param>
        /// <param name="truncating">The truncating.</param>
        /// <param name="value">The value.</param>
        /// <returns>Numpy array with shape (len(sequences), maxlen)</returns>
        public static NDarray PadSequences(NDarray sequences, int? maxlen = null, string dtype = "int32", string padding = "pre", string truncating = "pre", float value = 0)
        {
            Dictionary<string, object> parameters = new Dictionary<string, object>();
            parameters["sequences"] = sequences;
            parameters["maxlen"] = maxlen;
            parameters["dtype"] = dtype;
            parameters["padding"] = padding;
            parameters["truncating"] = truncating;
            parameters["value"] = value;
            var py = InvokeStaticMethod(caller, "pad_sequences", parameters);
            return new NDarray((PyObject)py);
        }

        /// <summary>
        /// Skips the grams.
        /// </summary>
        /// <param name="sequence"> A word sequence (sentence), encoded as a list of word indices (integers). If using a sampling_table, word indices are expected to match the rank of the words in a reference dataset (e.g. 10 would encode the 10-th most frequently occurring token). Note that index 0 is expected to be a non-word and will be skipped.</param>
        /// <param name="vocabulary_size"> Int, maximum possible word index + 1</param>
        /// <param name="window_size"> Int, size of sampling windows (technically half-window). The window of a word w_i will be [i - window_size, i + window_size+1].</param>
        /// <param name="negative_samples"> Float >= 0. 0 for no negative (i.e. random) samples. 1 for same number as positive samples.</param>
        /// <param name="shuffle"> Whether to shuffle the word couples before returning them.</param>
        /// <param name="categorical"> bool. if False, labels will be integers (eg. [0, 1, 1 .. ]), if True, labels will be categorical, e.g. [[1,0],[0,1],[0,1] .. ].</param>
        /// <param name="sampling_table"> 1D array of size vocabulary_size where the entry i encodes the probability to sample a word of rank i.</param>
        /// <param name="seed"> Random seed.</param>
        /// <returns>couples, labels: where couples are int pairs and  labels are either 0 or 1.</returns>
        public static NDarray SkipGrams(NDarray sequence, int vocabulary_size, int window_size= 4, float negative_samples= 1.0f, bool shuffle= true,
                                        bool categorical= false, NDarray sampling_table= null, int? seed= null)
        {
            Dictionary<string, object> parameters = new Dictionary<string, object>();
            parameters["sequence"] = sequence;
            parameters["vocabulary_size"] = vocabulary_size;
            parameters["window_size"] = window_size;
            parameters["negative_samples"] = negative_samples;
            parameters["shuffle"] = shuffle;
            parameters["categorical"] = categorical;
            parameters["sampling_table"] = sampling_table;
            parameters["seed"] = seed;


            return new NDarray((PyObject)InvokeStaticMethod(caller, "skipgrams", parameters));
        }

        /// <summary>
        /// Generates a word rank-based probabilistic sampling table.
        /// Used for generating the sampling_table argument for skipgrams.sampling_table[i] is the probability of sampling the word i-th most common word in a dataset(more common words should be sampled less frequently, for balance).
        /// The sampling probabilities are generated according to the sampling distribution used in word2vec:
        /// </summary>
        /// <param name="size">The size.</param>
        /// <param name="sampling_factor">The sampling factor.</param>
        /// <returns>A 1D Numpy array of length size where the ith entry is the probability that a word of rank i should be sampled.</returns>
        public static NDarray MakeSamplingTable(int size, float sampling_factor = 1e-05f)
        {
            Dictionary<string, object> parameters = new Dictionary<string, object>();
            parameters["size"] = size;
            parameters["sampling_factor"] = sampling_factor;

            return new NDarray((PyObject)InvokeStaticMethod(caller, "make_sampling_table", parameters));
        }
    }
}
