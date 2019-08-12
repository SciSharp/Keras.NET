using Keras.Utils;
using Numpy;
using System;
using System.Collections.Generic;
using System.Text;

namespace Keras.PreProcessing.sequence
{
    /// <summary>
    /// Utility class for generating batches of temporal data.
    /// This class takes in a sequence of data-points gathered at equal intervals, along with time series parameters such as stride, length of history, etc., 
    /// to produce batches for training/validation.
    /// </summary>
    /// <seealso cref="Keras.Base" />
    public class TimeseriesGenerator : Utils.Sequence
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="TimeseriesGenerator"/> class.
        /// </summary>
        /// <param name="data">The data.</param>
        /// <param name="targets">The targets.</param>
        /// <param name="length">The length.</param>
        /// <param name="sampling_rate">The sampling rate.</param>
        /// <param name="stride">The stride.</param>
        /// <param name="start_index">The start index.</param>
        /// <param name="end_index">The end index.</param>
        /// <param name="shuffle">if set to <c>true</c> [shuffle].</param>
        /// <param name="reverse">if set to <c>true</c> [reverse].</param>
        /// <param name="batch_size">Size of the batch.</param>
        public TimeseriesGenerator(NDarray data, NDarray targets, int length, int sampling_rate= 1, int stride= 1, 
                                int start_index= 0, int? end_index= null, bool shuffle= false, bool reverse= false, int batch_size= 128)
        {
            Parameters["data"] = data;
            Parameters["targets"] = targets;
            Parameters["length"] = length;
            Parameters["sampling_rate"] = sampling_rate;
            Parameters["stride"] = stride;
            Parameters["start_index"] = start_index;
            Parameters["end_index"] = end_index;
            Parameters["shuffle"] = shuffle;
            Parameters["reverse"] = reverse;
            Parameters["batch_size"] = batch_size;

            PyInstance = Instance.keras.preprocessing.sequence.TimeseriesGenerator;
        }
    }
}
