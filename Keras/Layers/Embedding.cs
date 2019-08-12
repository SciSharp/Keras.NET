namespace Keras.Layers
{
    /// <summary>
    /// Turns positive integers (indexes) into dense vectors of fixed size. eg. [[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]]
    /// This layer can only be used as the first layer in a model.
    /// </summary>
    /// <seealso cref="Keras.Layers.BaseLayer" />
    public class Embedding : BaseLayer
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="Embedding"/> class.
        /// </summary>
        /// <param name="input_dim"> int > 0. Size of the vocabulary, i.e. maximum integer index + 1.</param>
        /// <param name="output_dim"> int >= 0. Dimension of the dense embedding.</param>
        /// <param name="embeddings_initializer"> Initializer for the embeddings matrix (see initializers).</param>
        /// <param name="embeddings_regularizer"> Regularizer function applied to the embeddings matrix (see regularizer).</param>
        /// <param name="activity_regularizer"> Regularizer function applied to the output of the layer (its "activation"). (see regularizer).</param>
        /// <param name="embeddings_constraint"> Constraint function applied to the embeddings matrix (see constraints).</param>
        /// <param name="mask_zero"> Whether or not the input value 0 is a special "padding" value that should be masked out. This is useful when using recurrent layers which may take variable length input. If this is True then all subsequent layers in the model need to support masking or an exception will be raised. If mask_zero is set to True, as a consequence, index 0 cannot be used in the vocabulary (input_dim should equal size of vocabulary + 1).</param>
        /// <param name="input_length"> Length of input sequences, when it is constant. This argument is required if you are going to connect Flatten then Dense layers upstream (without it, the shape of the dense outputs cannot be computed).</param>
        /// <param name="input_shape">2D tensor with shape: (batch_size, sequence_length).</param>
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

            PyInstance = Instance.keras.layers.Embedding;
            Init();
        }
    }
}
