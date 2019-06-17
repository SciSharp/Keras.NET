using Numpy.Models;
using System;
using System.Collections.Generic;
using System.Text;

namespace Keras.Layers
{
    public class Embedding : BaseLayer
    {
        public Embedding(int input_dim, int output_dim, string embeddings_initializer= "uniform", string embeddings_regularizer= "",
                    string activity_regularizer= "", string embeddings_constraint= "", bool mask_zero= false, int? input_length= null, Shape input_shape = null)
        {
            Parameters["input_dim"] = input_dim;
            Parameters["output_dim"] = output_dim;
            Parameters["embeddings_initializer"] = embeddings_initializer;
            Parameters["embeddings_regularizer"] = embeddings_regularizer;
            Parameters["activity_regularizer"] = activity_regularizer;
            Parameters["embeddings_constraint"] = embeddings_constraint;
            Parameters["mask_zero"] = mask_zero;
            Parameters["input_length"] = input_length;
            Parameters["input_shape"] = input_shape;

            __self__ = Instance.self.layers.Embedding;
        }
    }
}
