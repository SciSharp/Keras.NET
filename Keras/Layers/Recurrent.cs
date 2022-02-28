using System;
using System.Collections.Generic;
using System.Text;

namespace Keras.Layers
{
    /// <summary>
    /// Base class for recurrent layers. 
    /// This layer supports masking for input data with a variable number of timesteps. To introduce masks to your data, use an Embedding layer with the mask_zero parameter set to True.
    /// <para>
    /// You can set RNN layers to be 'stateful', which means that the states computed for the samples in one batch will be reused as initial states for the samples in the next batch. This assumes a one-to-one mapping between samples in different successive batches.
    /// To enable statefulness: - specify stateful = True in the layer constructor. - specify a fixed batch size for your model, by passing if sequential model: batch_input_shape = (...) to the first layer in your model. else for functional model with 1 or more Input layers: batch_shape = (...) to all the first layers in your model.This is the expected shape of your inputs including the batch size.It should be a tuple of integers, e.g. (32, 10, 100). - specify shuffle = False when calling fit().
    /// To reset the states of your model, call.reset_states() on either a specific layer, or on your entire model.
    /// </para>
    /// <para>
    /// You can specify the initial state of RNN layers symbolically by calling them with the keyword argument initial_state. 
    /// The value of initial_state should be a tensor or list of tensors representing the initial state of the RNN layer.
    /// You can specify the initial state of RNN layers numerically by calling reset_states with the keyword argument states.The value of states should be a numpy array or list of numpy arrays representing the initial state of the RNN layer.
    /// </para>
    /// <para>
    /// You can pass "external" constants to the cell using the constants keyword argument of RNN.__call__ (as well as RNN.call) method. This requires that the cell. 
    /// Call method accepts the same keyword argument constants. Such constants can be used to condition the cell transformation on additional static inputs (not changing over time), a.k.a. an attention mechanism.
    /// </para>
    /// </summary>
    /// <seealso cref="Keras.Layers.BaseLayer" />
    public class RNN : BaseLayer
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="RNN"/> class.
        /// </summary>
        public RNN()
        {

        }

        /// <summary>
        /// Initializes a new instance of the <see cref="RNN" /> class.
        /// </summary>
        /// <param name="cell">A RNN cell instance.</param>
        /// <param name="return_sequences">Boolean. Whether to return the last output in the output sequence, or the full sequence.</param>
        /// <param name="return_state">Boolean. Whether to return the last state in addition to the output.</param>
        /// <param name="go_backwards">Boolean (default False). If True, process the input sequence backwards and return the reversed sequence.</param>
        /// <param name="stateful">Boolean (default False). If True, the last state for each sample at index i in a batch will be used as initial state for the sample of index i in the following batch.</param>
        /// <param name="unroll">Boolean (default False). If True, the network will be unrolled, else a symbolic loop will be used. Unrolling can speed-up a RNN, although it tends to be more memory-intensive. Unrolling is only suitable for short sequences.</param>
        /// <param name="input_dim">dimensionality of the input (integer). This argument (or alternatively, the keyword argument input_shape) is required when using this layer as the first layer in a model..</param>
        /// <param name="input_length">Length of input sequences, to be specified when it is constant. This argument is required if you are going to connect  Flatten then Dense layers upstream (without it, the shape of the dense outputs cannot be computed). Note that if the recurrent layer is not the first layer in your model, you would need to specify the input length at the level of the first layer (e.g. via the input_shape argument)</param>
        /// <param name="input_shape">3D tensor with shape (batch_size, timesteps, input_dim).</param>
        public RNN(RNN cell, bool return_sequences = false, bool return_state = false, bool go_backwards = false, bool stateful = false, bool unroll = false, int? input_dim = null, int? input_length = null, Shape input_shape = null)
        {
            Parameters["cell"] = cell.PyInstance;
            Parameters["return_sequences"] = return_sequences;
            Parameters["return_state"] = return_state;
            Parameters["go_backwards"] = go_backwards;
            Parameters["stateful"] = stateful;
            Parameters["unroll"] = unroll;
            PyInstance = Instance.keras.layers.RNN;
            Init();
        }
    }

    /// <summary>
    /// Fully-connected RNN where the output is to be fed back to input.
    /// </summary>
    public class SimpleRNN : RNN
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="SimpleRNN" /> class.
        /// </summary>
        /// <param name="units"> Positive integer, dimensionality of the output space.</param>
        /// <param name="activation"> Activation function to use (see activations). Default: hyperbolic tangent (tanh). If you pass None, no activation is applied (ie. "linear" activation: a(x) = x).</param>
        /// <param name="use_bias"> Boolean, whether the layer uses a bias vector.</param>
        /// <param name="kernel_initializer"> Initializer for the kernel weights matrix, used for the linear transformation of the inputs (see initializers).</param>
        /// <param name="recurrent_initializer"> Initializer for the recurrent_kernel weights matrix, used for the linear transformation of the recurrent state (see initializers).</param>
        /// <param name="bias_initializer"> Initializer for the bias vector (see initializers).</param>
        /// <param name="kernel_regularizer"> Regularizer function applied to the kernel weights matrix (see regularizer).</param>
        /// <param name="recurrent_regularizer"> Regularizer function applied to the recurrent_kernel weights matrix (see regularizer).</param>
        /// <param name="bias_regularizer"> Regularizer function applied to the bias vector (see regularizer).</param>
        /// <param name="activity_regularizer"> Regularizer function applied to the output of the layer (its "activation"). (see regularizer).</param>
        /// <param name="kernel_constraint"> Constraint function applied to the kernel weights matrix (see constraints).</param>
        /// <param name="recurrent_constraint"> Constraint function applied to the recurrent_kernel weights matrix (see constraints).</param>
        /// <param name="bias_constraint"> Constraint function applied to the bias vector (see constraints).</param>
        /// <param name="dropout"> Float between 0 and 1. Fraction of the units to drop for the linear transformation of the inputs.</param>
        /// <param name="recurrent_dropout"> Float between 0 and 1. Fraction of the units to drop for the linear transformation of the recurrent state.</param>
        /// <param name="return_sequences"> Boolean. Whether to return the last output in the output sequence, or the full sequence.</param>
        /// <param name="return_state"> Boolean. Whether to return the last state in addition to the output.</param>
        /// <param name="go_backwards"> Boolean (default False). If True, process the input sequence backwards and return the reversed sequence.</param>
        /// <param name="stateful"> Boolean (default False). If True, the last state for each sample at index i in a batch will be used as initial state for the sample of index i in the following batch.</param>
        /// <param name="unroll"> Boolean (default False). If True, the network will be unrolled, else a symbolic loop will be used. Unrolling can speed-up a RNN, although it tends to be more memory-intensive. Unrolling is only suitable for short sequences.</param>

        public SimpleRNN(int units, string activation= "tanh", bool use_bias= true, string kernel_initializer= "glorot_uniform", string recurrent_initializer= "orthogonal"
            , string bias_initializer= "zeros", string kernel_regularizer= "", string recurrent_regularizer= "", string bias_regularizer= "", 
            string activity_regularizer= "", string kernel_constraint= "", string recurrent_constraint= "", string bias_constraint= "", float dropout= 0.0f,
            float recurrent_dropout= 0.0f, bool return_sequences= false, bool return_state= false, bool go_backwards= false, bool stateful= false, bool unroll= false)
        {
            Parameters["units"] = units;
            Parameters["activation"] = activation;
            Parameters["use_bias"] = use_bias;
            Parameters["kernel_initializer"] = kernel_initializer;
            Parameters["recurrent_initializer"] = recurrent_initializer;
            Parameters["bias_initializer"] = bias_initializer;
            Parameters["kernel_regularizer"] = kernel_regularizer;
            Parameters["recurrent_regularizer"] = recurrent_regularizer;
            Parameters["bias_regularizer"] = bias_regularizer;
            Parameters["activity_regularizer"] = activity_regularizer;
            Parameters["kernel_constraint"] = kernel_constraint;
            Parameters["recurrent_constraint"] = recurrent_constraint;
            Parameters["bias_constraint"] = bias_constraint;
            Parameters["dropout"] = dropout;
            Parameters["recurrent_dropout"] = recurrent_dropout;
            Parameters["return_sequences"] = return_sequences;
            Parameters["return_state"] = return_state;
            Parameters["go_backwards"] = go_backwards;
            Parameters["stateful"] = stateful;
            Parameters["unroll"] = unroll;
            PyInstance = Instance.keras.layers.SimpleRNN;
            Init();
        }
    }

    /// <summary>
    /// Cell class for SimpleRNN.
    /// </summary>
    /// <seealso cref="Keras.Layers.RNN" />
    public class SimpleRNNCell : RNN
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="SimpleRNNCell" /> class.
        /// </summary>
        /// <param name="units">Positive integer, dimensionality of the output space.</param>
        /// <param name="activation">Activation function to use (see activations). Default: hyperbolic tangent (tanh). If you pass None, no activation is applied (ie. "linear" activation: a(x) = x).</param>
        /// <param name="use_bias">Boolean, whether the layer uses a bias vector.</param>
        /// <param name="kernel_initializer">Initializer for the kernel weights matrix, used for the linear transformation of the inputs (see initializers).</param>
        /// <param name="recurrent_initializer">Initializer for the recurrent_kernel weights matrix, used for the linear transformation of the recurrent state (see initializers).</param>
        /// <param name="bias_initializer">Initializer for the bias vector (see initializers).</param>
        /// <param name="kernel_regularizer">Regularizer function applied to the kernel weights matrix (see regularizer).</param>
        /// <param name="recurrent_regularizer">Regularizer function applied to the recurrent_kernel weights matrix (see regularizer).</param>
        /// <param name="bias_regularizer">Regularizer function applied to the bias vector (see regularizer).</param>
        /// <param name="activity_regularizer">Regularizer function applied to the output of the layer (its "activation"). (see regularizer).</param>
        /// <param name="kernel_constraint">Constraint function applied to the kernel weights matrix (see constraints).</param>
        /// <param name="recurrent_constraint">Constraint function applied to the recurrent_kernel weights matrix (see constraints).</param>
        /// <param name="bias_constraint">Constraint function applied to the bias vector (see constraints).</param>
        /// <param name="dropout">Float between 0 and 1. Fraction of the units to drop for the linear transformation of the inputs.</param>
        /// <param name="recurrent_dropout">Float between 0 and 1. Fraction of the units to drop for the linear transformation of the recurrent state.</param>

        public SimpleRNNCell(int units, string activation = "tanh", bool use_bias = true, string kernel_initializer = "glorot_uniform", string recurrent_initializer = "orthogonal"
            , string bias_initializer = "zeros", string kernel_regularizer = "", string recurrent_regularizer = "", string bias_regularizer = "",
            string activity_regularizer = "", string kernel_constraint = "", string recurrent_constraint = "", string bias_constraint = "", float dropout = 0.0f, float recurrent_dropout = 0.0f)
        {
            Parameters["units"] = units;
            Parameters["activation"] = activation;
            Parameters["use_bias"] = use_bias;
            Parameters["kernel_initializer"] = kernel_initializer;
            Parameters["recurrent_initializer"] = recurrent_initializer;
            Parameters["bias_initializer"] = bias_initializer;
            Parameters["kernel_regularizer"] = kernel_regularizer;
            Parameters["recurrent_regularizer"] = recurrent_regularizer;
            Parameters["bias_regularizer"] = bias_regularizer;
            Parameters["activity_regularizer"] = activity_regularizer;
            Parameters["kernel_constraint"] = kernel_constraint;
            Parameters["recurrent_constraint"] = recurrent_constraint;
            Parameters["bias_constraint"] = bias_constraint;
            Parameters["dropout"] = dropout;
            Parameters["recurrent_dropout"] = recurrent_dropout;
            PyInstance = Instance.keras.layers.SimpleRNNCell;
            Init();
        }
    }

    /// <summary>
    /// Gated Recurrent Unit - Cho et al. 2014.
    /// There are two variants.The default one is based on 1406.1078v3 and has reset gate applied to hidden state before matrix multiplication. The other one is based on original 1406.1078v1 and has the order reversed.
    /// The second variant is compatible with CuDNNGRU (GPU-only) and allows inference on CPU.Thus it has separate biases for kernel and recurrent_kernel.Use 'reset_after'=True and recurrent_activation='sigmoid'.
    /// </summary>
    public class GRU : RNN
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="GRU"/> class.
        /// </summary>
        /// <param name="units"> Positive integer, dimensionality of the output space.</param>
        /// <param name="activation"> Activation function to use (see activations). Default: hyperbolic tangent (tanh). If you pass None, no activation is applied (ie. "linear" activation: a(x) = x).</param>
        /// <param name="recurrent_activation"> Activation function to use for the recurrent step (see activations). Default: hard sigmoid (hard_sigmoid). If you pass None, no activation is applied (ie. "linear" activation: a(x) = x).</param>
        /// <param name="use_bias"> Boolean, whether the layer uses a bias vector.</param>
        /// <param name="kernel_initializer"> Initializer for the kernel weights matrix, used for the linear transformation of the inputs (see initializers).</param>
        /// <param name="recurrent_initializer"> Initializer for the recurrent_kernel weights matrix, used for the linear transformation of the recurrent state (see initializers).</param>
        /// <param name="bias_initializer"> Initializer for the bias vector (see initializers).</param>
        /// <param name="kernel_regularizer"> Regularizer function applied to the kernel weights matrix (see regularizer).</param>
        /// <param name="recurrent_regularizer"> Regularizer function applied to the recurrent_kernel weights matrix (see regularizer).</param>
        /// <param name="bias_regularizer"> Regularizer function applied to the bias vector (see regularizer).</param>
        /// <param name="activity_regularizer"> Regularizer function applied to the output of the layer (its "activation"). (see regularizer).</param>
        /// <param name="kernel_constraint"> Constraint function applied to the kernel weights matrix (see constraints).</param>
        /// <param name="recurrent_constraint"> Constraint function applied to the recurrent_kernel weights matrix (see constraints).</param>
        /// <param name="bias_constraint"> Constraint function applied to the bias vector (see constraints).</param>
        /// <param name="dropout"> Float between 0 and 1. Fraction of the units to drop for the linear transformation of the inputs.</param>
        /// <param name="recurrent_dropout"> Float between 0 and 1. Fraction of the units to drop for the linear transformation of the recurrent state.</param>
        /// <param name="implementation"> Implementation mode, either 1 or 2. Mode 1 will structure its operations as a larger number of smaller dot products and additions, whereas mode 2 will batch them into fewer, larger operations. These modes will have different performance profiles on different hardware and for different applications.</param>
        /// <param name="return_sequences"> Boolean. Whether to return the last output in the output sequence, or the full sequence.</param>
        /// <param name="return_state"> Boolean. Whether to return the last state in addition to the output.</param>
        /// <param name="go_backwards"> Boolean (default False). If True, process the input sequence backwards and return the reversed sequence.</param>
        /// <param name="stateful"> Boolean (default False). If True, the last state for each sample at index i in a batch will be used as initial state for the sample of index i in the following batch.</param>
        /// <param name="unroll"> Boolean (default False). If True, the network will be unrolled, else a symbolic loop will be used. Unrolling can speed-up a RNN, although it tends to be more memory-intensive. Unrolling is only suitable for short sequences.</param>
        /// <param name="reset_after"> GRU convention (whether to apply reset gate after or before matrix multiplication). False = "before" (default), True = "after" (CuDNN compatible).</param>
        public GRU(int units, string activation = "tanh", string recurrent_activation = "hard_sigmoid", bool use_bias = true, string kernel_initializer = "glorot_uniform"
            , string recurrent_initializer = "orthogonal", string bias_initializer = "zeros", string kernel_regularizer = "", string recurrent_regularizer = "",
            string bias_regularizer = "", string activity_regularizer = "", string kernel_constraint = "", string recurrent_constraint = "", string bias_constraint = "",
            float dropout = 0.0f, float recurrent_dropout = 0.0f, int implementation = 1, bool return_sequences = false, bool return_state = false, bool go_backwards = false,
            bool stateful = false, bool unroll = false, bool reset_after = false, Shape input_shape = null)
        {
            Parameters["units"] = units;
            Parameters["activation"] = activation;
            Parameters["recurrent_activation"] = recurrent_activation;
            Parameters["use_bias"] = use_bias;
            Parameters["kernel_initializer"] = kernel_initializer;
            Parameters["recurrent_initializer"] = recurrent_initializer;
            Parameters["bias_initializer"] = bias_initializer;
            Parameters["kernel_regularizer"] = kernel_regularizer;
            Parameters["recurrent_regularizer"] = recurrent_regularizer;
            Parameters["bias_regularizer"] = bias_regularizer;
            Parameters["activity_regularizer"] = activity_regularizer;
            Parameters["kernel_constraint"] = kernel_constraint;
            Parameters["recurrent_constraint"] = recurrent_constraint;
            Parameters["bias_constraint"] = bias_constraint;
            Parameters["dropout"] = dropout;
            Parameters["recurrent_dropout"] = recurrent_dropout;
            Parameters["implementation"] = implementation;
            Parameters["return_sequences"] = return_sequences;
            Parameters["return_state"] = return_state;
            Parameters["go_backwards"] = go_backwards;
            Parameters["stateful"] = stateful;
            Parameters["unroll"] = unroll;
            Parameters["reset_after"] = reset_after;
            
            if (input_shape != null)
            {
                Parameters["input_shape"] = input_shape;
            }
            
            PyInstance = Instance.keras.layers.GRU;
            Init();
        }
    }

    /// <summary>
    /// Fast GRU implementation backed by CuDNN. Can only be run on GPU, with the TensorFlow backend.
    /// </summary>
    /// <seealso cref="Keras.Layers.RNN" />
    public class CuDNNGRU : RNN
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="CuDNNGRU" /> class.
        /// </summary>
        /// <param name="units">Positive integer, dimensionality of the output space.</param>
        /// <param name="kernel_initializer">Initializer for the kernel weights matrix, used for the linear transformation of the inputs (see initializers).</param>
        /// <param name="recurrent_initializer">Initializer for the recurrent_kernel weights matrix, used for the linear transformation of the recurrent state (see initializers).</param>
        /// <param name="bias_initializer">Initializer for the bias vector (see initializers).</param>
        /// <param name="kernel_regularizer">Regularizer function applied to the kernel weights matrix (see regularizer).</param>
        /// <param name="recurrent_regularizer">Regularizer function applied to the recurrent_kernel weights matrix (see regularizer).</param>
        /// <param name="bias_regularizer">Regularizer function applied to the bias vector (see regularizer).</param>
        /// <param name="activity_regularizer">Regularizer function applied to the output of the layer (its "activation"). (see regularizer).</param>
        /// <param name="kernel_constraint">Constraint function applied to the kernel weights matrix (see constraints).</param>
        /// <param name="recurrent_constraint">Constraint function applied to the recurrent_kernel weights matrix (see constraints).</param>
        /// <param name="bias_constraint">Constraint function applied to the bias vector (see constraints).</param>
        /// <param name="return_sequences">Boolean. Whether to return the last output in the output sequence, or the full sequence.</param>
        /// <param name="return_state">Boolean. Whether to return the last state in addition to the output.</param>
        /// <param name="stateful">Boolean (default False). If True, the last state for each sample at index i in a batch will be used as initial state for the sample of index i in the following batch.</param>
        public CuDNNGRU(int units, string kernel_initializer = "glorot_uniform" , string recurrent_initializer = "orthogonal", string bias_initializer = "zeros", string kernel_regularizer = "", string recurrent_regularizer = "",
            string bias_regularizer = "", string activity_regularizer = "", string kernel_constraint = "", string recurrent_constraint = "", string bias_constraint = "",
            bool return_sequences = false, bool return_state = false, bool stateful = false)
        {
            Parameters["units"] = units;
            Parameters["kernel_initializer"] = kernel_initializer;
            Parameters["recurrent_initializer"] = recurrent_initializer;
            Parameters["bias_initializer"] = bias_initializer;
            Parameters["kernel_regularizer"] = kernel_regularizer;
            Parameters["recurrent_regularizer"] = recurrent_regularizer;
            Parameters["bias_regularizer"] = bias_regularizer;
            Parameters["activity_regularizer"] = activity_regularizer;
            Parameters["kernel_constraint"] = kernel_constraint;
            Parameters["recurrent_constraint"] = recurrent_constraint;
            Parameters["bias_constraint"] = bias_constraint;
            Parameters["return_sequences"] = return_sequences;
            Parameters["return_state"] = return_state;
            Parameters["stateful"] = stateful;
            PyInstance = Instance.keras.layers.CuDNNGRU;
            Init();
        }
    }

    /// <summary>
    /// Cell class for the GRU layer.
    /// </summary>
    /// <seealso cref="Keras.Layers.RNN" />
    public class GRUCell : RNN
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="GRUCell"/> class.
        /// </summary>
        /// <param name="units"> Positive integer, dimensionality of the output space.</param>
        /// <param name="activation"> Activation function to use (see activations). Default: hyperbolic tangent (tanh). If you pass None, no activation is applied (ie. "linear" activation: a(x) = x).</param>
        /// <param name="recurrent_activation"> Activation function to use for the recurrent step (see activations). Default: hard sigmoid (hard_sigmoid). If you pass None, no activation is applied (ie. "linear" activation: a(x) = x).</param>
        /// <param name="use_bias"> Boolean, whether the layer uses a bias vector.</param>
        /// <param name="kernel_initializer"> Initializer for the kernel weights matrix, used for the linear transformation of the inputs (see initializers).</param>
        /// <param name="recurrent_initializer"> Initializer for the recurrent_kernel weights matrix, used for the linear transformation of the recurrent state (see initializers).</param>
        /// <param name="bias_initializer"> Initializer for the bias vector (see initializers).</param>
        /// <param name="kernel_regularizer"> Regularizer function applied to the kernel weights matrix (see regularizer).</param>
        /// <param name="recurrent_regularizer"> Regularizer function applied to the recurrent_kernel weights matrix (see regularizer).</param>
        /// <param name="bias_regularizer"> Regularizer function applied to the bias vector (see regularizer).</param>
        /// <param name="activity_regularizer"> Regularizer function applied to the output of the layer (its "activation"). (see regularizer).</param>
        /// <param name="kernel_constraint"> Constraint function applied to the kernel weights matrix (see constraints).</param>
        /// <param name="recurrent_constraint"> Constraint function applied to the recurrent_kernel weights matrix (see constraints).</param>
        /// <param name="bias_constraint"> Constraint function applied to the bias vector (see constraints).</param>
        /// <param name="dropout"> Float between 0 and 1. Fraction of the units to drop for the linear transformation of the inputs.</param>
        /// <param name="recurrent_dropout"> Float between 0 and 1. Fraction of the units to drop for the linear transformation of the recurrent state.</param>
        /// <param name="implementation"> Implementation mode, either 1 or 2. Mode 1 will structure its operations as a larger number of smaller dot products and additions, whereas mode 2 will batch them into fewer, larger operations. These modes will have different performance profiles on different hardware and for different applications.</param>
        /// <param name="reset_after"> GRU convention (whether to apply reset gate after or before matrix multiplication). False = "before" (default), True = "after" (CuDNN compatible).</param>
        public GRUCell(int units, string activation = "tanh", string recurrent_activation = "hard_sigmoid", bool use_bias = true, string kernel_initializer = "glorot_uniform"
            , string recurrent_initializer = "orthogonal", string bias_initializer = "zeros", string kernel_regularizer = "", string recurrent_regularizer = "",
            string bias_regularizer = "", string activity_regularizer = "", string kernel_constraint = "", string recurrent_constraint = "", string bias_constraint = "",
            float dropout = 0.0f, float recurrent_dropout = 0.0f, int implementation = 1, bool reset_after = false)
        {
            Parameters["units"] = units;
            Parameters["activation"] = activation;
            Parameters["recurrent_activation"] = recurrent_activation;
            Parameters["use_bias"] = use_bias;
            Parameters["kernel_initializer"] = kernel_initializer;
            Parameters["recurrent_initializer"] = recurrent_initializer;
            Parameters["bias_initializer"] = bias_initializer;
            Parameters["kernel_regularizer"] = kernel_regularizer;
            Parameters["recurrent_regularizer"] = recurrent_regularizer;
            Parameters["bias_regularizer"] = bias_regularizer;
            Parameters["activity_regularizer"] = activity_regularizer;
            Parameters["kernel_constraint"] = kernel_constraint;
            Parameters["recurrent_constraint"] = recurrent_constraint;
            Parameters["bias_constraint"] = bias_constraint;
            Parameters["dropout"] = dropout;
            Parameters["recurrent_dropout"] = recurrent_dropout;
            Parameters["implementation"] = implementation;
            Parameters["reset_after"] = reset_after;
            PyInstance = Instance.keras.layers.GRUCell;
            Init();
        }
    }

    /// <summary>
    /// Long Short-Term Memory layer - Hochreiter 1997.
    /// </summary>
    public class LSTM : RNN
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="LSTM" /> class.
        /// </summary>
        /// <param name="units">Positive integer, dimensionality of the output space.</param>
        /// <param name="activation">Activation function to use (see activations). Default: hyperbolic tangent (tanh). If you pass None, no activation is applied (ie. "linear" activation: a(x) = x).</param>
        /// <param name="recurrent_activation">Activation function to use for the recurrent step (see activations). Default: hard sigmoid (hard_sigmoid). If you pass None, no activation is applied (ie. "linear" activation: a(x) = x).</param>
        /// <param name="use_bias">Boolean, whether the layer uses a bias vector.</param>
        /// <param name="kernel_initializer">Initializer for the kernel weights matrix, used for the linear transformation of the inputs (see initializers).</param>
        /// <param name="recurrent_initializer">Initializer for the recurrent_kernel weights matrix, used for the linear transformation of the recurrent state (see initializers).</param>
        /// <param name="bias_initializer">Initializer for the bias vector (see initializers).</param>
        /// <param name="unit_forget_bias">if set to <c>true</c> [unit forget bias].</param>
        /// <param name="kernel_regularizer">Regularizer function applied to the kernel weights matrix (see regularizer).</param>
        /// <param name="recurrent_regularizer">Regularizer function applied to the recurrent_kernel weights matrix (see regularizer).</param>
        /// <param name="bias_regularizer">Regularizer function applied to the bias vector (see regularizer).</param>
        /// <param name="activity_regularizer">Regularizer function applied to the output of the layer (its "activation"). (see regularizer).</param>
        /// <param name="kernel_constraint">Constraint function applied to the kernel weights matrix (see constraints).</param>
        /// <param name="recurrent_constraint">Constraint function applied to the recurrent_kernel weights matrix (see constraints).</param>
        /// <param name="bias_constraint">Constraint function applied to the bias vector (see constraints).</param>
        /// <param name="dropout">Float between 0 and 1. Fraction of the units to drop for the linear transformation of the inputs.</param>
        /// <param name="recurrent_dropout">Float between 0 and 1. Fraction of the units to drop for the linear transformation of the recurrent state.</param>
        /// <param name="implementation">Implementation mode, either 1 or 2. Mode 1 will structure its operations as a larger number of smaller dot products and additions, whereas mode 2 will batch them into fewer, larger operations. These modes will have different performance profiles on different hardware and for different applications.</param>
        /// <param name="return_sequences">Boolean. Whether to return the last output in the output sequence, or the full sequence.</param>
        /// <param name="return_state">Boolean. Whether to return the last state in addition to the output.</param>
        /// <param name="go_backwards">Boolean (default False). If True, process the input sequence backwards and return the reversed sequence.</param>
        /// <param name="stateful">Boolean (default False). If True, the last state for each sample at index i in a batch will be used as initial state for the sample of index i in the following batch.</param>
        /// <param name="unroll">Boolean (default False). If True, the network will be unrolled, else a symbolic loop will be used. Unrolling can speed-up a RNN, although it tends to be more memory-intensive. Unrolling is only suitable for short sequences.</param>
        /// <param name="batch_input_shape">Optional input batch size (integer or None).</param>
        public LSTM(int units, string activation = "tanh", string recurrent_activation = "hard_sigmoid", bool use_bias = true, StringOrInstance kernel_initializer = null
            , StringOrInstance recurrent_initializer = null, string bias_initializer = "zeros", bool unit_forget_bias = true, string kernel_regularizer = "",
            string recurrent_regularizer = "", string bias_regularizer = "", string activity_regularizer = "", string kernel_constraint = "", string recurrent_constraint = "",
            string bias_constraint = "", float dropout = 0.0f, float recurrent_dropout = 0.0f, int implementation = 1, bool return_sequences = false, bool return_state = false,
            bool go_backwards = false, bool stateful = false, bool unroll = false, Shape batch_input_shape = null, Shape input_shape = null)
        {
            Parameters["units"] = units;
            Parameters["activation"] = activation;
            Parameters["recurrent_activation"] = recurrent_activation;
            Parameters["use_bias"] = use_bias;
            Parameters["kernel_initializer"] = kernel_initializer;
            Parameters["recurrent_initializer"] = recurrent_initializer;
            Parameters["bias_initializer"] = bias_initializer;
            Parameters["unit_forget_bias"] = unit_forget_bias;
            Parameters["kernel_regularizer"] = kernel_regularizer;
            Parameters["recurrent_regularizer"] = recurrent_regularizer;
            Parameters["bias_regularizer"] = bias_regularizer;
            Parameters["activity_regularizer"] = activity_regularizer;
            Parameters["kernel_constraint"] = kernel_constraint;
            Parameters["recurrent_constraint"] = recurrent_constraint;
            Parameters["bias_constraint"] = bias_constraint;
            Parameters["dropout"] = dropout;
            Parameters["recurrent_dropout"] = recurrent_dropout;
            Parameters["implementation"] = implementation;
            Parameters["return_sequences"] = return_sequences;
            Parameters["return_state"] = return_state;
            Parameters["go_backwards"] = go_backwards;
            Parameters["stateful"] = stateful;
            Parameters["unroll"] = unroll;

            if (batch_input_shape != null)
            {
                Parameters["batch_input_shape"] = batch_input_shape;
            }
            else if (input_shape != null)
            {
                Parameters["input_shape"] = input_shape;
            }

            PyInstance = Instance.keras.layers.LSTM;
            Init();
        }
    }

    public class CuDNNLSTM : RNN
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="CuDNNLSTM" /> class.
        /// </summary>
        /// <param name="units">Positive integer, dimensionality of the output space.</param>
        /// <param name="kernel_initializer">Initializer for the kernel weights matrix, used for the linear transformation of the inputs (see initializers).</param>
        /// <param name="recurrent_initializer">Initializer for the recurrent_kernel weights matrix, used for the linear transformation of the recurrent state (see initializers).</param>
        /// <param name="bias_initializer">Initializer for the bias vector (see initializers).</param>
        /// <param name="unit_forget_bias">if set to <c>true</c> [unit forget bias].</param>
        /// <param name="kernel_regularizer">Regularizer function applied to the kernel weights matrix (see regularizer).</param>
        /// <param name="recurrent_regularizer">Regularizer function applied to the recurrent_kernel weights matrix (see regularizer).</param>
        /// <param name="bias_regularizer">Regularizer function applied to the bias vector (see regularizer).</param>
        /// <param name="activity_regularizer">Regularizer function applied to the output of the layer (its "activation"). (see regularizer).</param>
        /// <param name="kernel_constraint">Constraint function applied to the kernel weights matrix (see constraints).</param>
        /// <param name="recurrent_constraint">Constraint function applied to the recurrent_kernel weights matrix (see constraints).</param>
        /// <param name="bias_constraint">Constraint function applied to the bias vector (see constraints).</param>
        /// <param name="return_sequences">Boolean. Whether to return the last output in the output sequence, or the full sequence.</param>
        /// <param name="return_state">Boolean. Whether to return the last state in addition to the output.</param>
        /// <param name="stateful">Boolean (default False). If True, the last state for each sample at index i in a batch will be used as initial state for the sample of index i in the following batch.</param>
        public CuDNNLSTM(int units, string kernel_initializer = "glorot_uniform" , string recurrent_initializer = "orthogonal", string bias_initializer = "zeros", 
            bool unit_forget_bias = true, string kernel_regularizer = "", string recurrent_regularizer = "", string bias_regularizer = "", string activity_regularizer = "", 
            string kernel_constraint = "", string recurrent_constraint = "", string bias_constraint = "", bool return_sequences = false, bool return_state = false, bool stateful = false)
        {
            Parameters["units"] = units;
            Parameters["kernel_initializer"] = kernel_initializer;
            Parameters["recurrent_initializer"] = recurrent_initializer;
            Parameters["bias_initializer"] = bias_initializer;
            Parameters["unit_forget_bias"] = unit_forget_bias;
            Parameters["kernel_regularizer"] = kernel_regularizer;
            Parameters["recurrent_regularizer"] = recurrent_regularizer;
            Parameters["bias_regularizer"] = bias_regularizer;
            Parameters["activity_regularizer"] = activity_regularizer;
            Parameters["kernel_constraint"] = kernel_constraint;
            Parameters["recurrent_constraint"] = recurrent_constraint;
            Parameters["bias_constraint"] = bias_constraint;
            Parameters["return_sequences"] = return_sequences;
            Parameters["return_state"] = return_state;
            Parameters["stateful"] = stateful;
            PyInstance = Instance.keras.layers.CuDNNLSTM;
            Init();
        }
    }

    /// <summary>
    /// Cell class for the LSTM layer.
    /// </summary>
    /// <seealso cref="Keras.Layers.RNN" />
    public class LSTMCell : RNN
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="LSTMCell" /> class.
        /// </summary>
        /// <param name="units">Positive integer, dimensionality of the output space.</param>
        /// <param name="activation">Activation function to use (see activations). Default: hyperbolic tangent (tanh). If you pass None, no activation is applied (ie. "linear" activation: a(x) = x).</param>
        /// <param name="recurrent_activation">Activation function to use for the recurrent step (see activations). Default: hard sigmoid (hard_sigmoid). If you pass None, no activation is applied (ie. "linear" activation: a(x) = x).</param>
        /// <param name="use_bias">Boolean, whether the layer uses a bias vector.</param>
        /// <param name="kernel_initializer">Initializer for the kernel weights matrix, used for the linear transformation of the inputs (see initializers).</param>
        /// <param name="recurrent_initializer">Initializer for the recurrent_kernel weights matrix, used for the linear transformation of the recurrent state (see initializers).</param>
        /// <param name="bias_initializer">Initializer for the bias vector (see initializers).</param>
        /// <param name="unit_forget_bias">if set to <c>true</c> [unit forget bias].</param>
        /// <param name="kernel_regularizer">Regularizer function applied to the kernel weights matrix (see regularizer).</param>
        /// <param name="recurrent_regularizer">Regularizer function applied to the recurrent_kernel weights matrix (see regularizer).</param>
        /// <param name="bias_regularizer">Regularizer function applied to the bias vector (see regularizer).</param>
        /// <param name="activity_regularizer">Regularizer function applied to the output of the layer (its "activation"). (see regularizer).</param>
        /// <param name="kernel_constraint">Constraint function applied to the kernel weights matrix (see constraints).</param>
        /// <param name="recurrent_constraint">Constraint function applied to the recurrent_kernel weights matrix (see constraints).</param>
        /// <param name="bias_constraint">Constraint function applied to the bias vector (see constraints).</param>
        /// <param name="dropout">Float between 0 and 1. Fraction of the units to drop for the linear transformation of the inputs.</param>
        /// <param name="recurrent_dropout">Float between 0 and 1. Fraction of the units to drop for the linear transformation of the recurrent state.</param>
        /// <param name="implementation">Implementation mode, either 1 or 2. Mode 1 will structure its operations as a larger number of smaller dot products and additions, whereas mode 2 will batch them into fewer, larger operations. These modes will have different performance profiles on different hardware and for different applications.</param>
        public LSTMCell(int units, string activation = "tanh", string recurrent_activation = "hard_sigmoid", bool use_bias = true, string kernel_initializer = "glorot_uniform"
            , string recurrent_initializer = "orthogonal", string bias_initializer = "zeros", bool unit_forget_bias = true, string kernel_regularizer = "",
            string recurrent_regularizer = "", string bias_regularizer = "", string activity_regularizer = "", string kernel_constraint = "", string recurrent_constraint = "",
            string bias_constraint = "", float dropout = 0.0f, float recurrent_dropout = 0.0f, int implementation = 1)
        {
            Parameters["units"] = units;
            Parameters["activation"] = activation;
            Parameters["recurrent_activation"] = recurrent_activation;
            Parameters["use_bias"] = use_bias;
            Parameters["kernel_initializer"] = kernel_initializer;
            Parameters["recurrent_initializer"] = recurrent_initializer;
            Parameters["bias_initializer"] = bias_initializer;
            Parameters["unit_forget_bias"] = unit_forget_bias;
            Parameters["kernel_regularizer"] = kernel_regularizer;
            Parameters["recurrent_regularizer"] = recurrent_regularizer;
            Parameters["bias_regularizer"] = bias_regularizer;
            Parameters["activity_regularizer"] = activity_regularizer;
            Parameters["kernel_constraint"] = kernel_constraint;
            Parameters["recurrent_constraint"] = recurrent_constraint;
            Parameters["bias_constraint"] = bias_constraint;
            Parameters["dropout"] = dropout;
            Parameters["recurrent_dropout"] = recurrent_dropout;
            Parameters["implementation"] = implementation;
            PyInstance = Instance.keras.layers.LSTMCell;
            Init();
        }
    }

    /// <summary>
    /// Convolutional LSTM. It is similar to an LSTM layer, but the input transformations and recurrent transformations are both convolutional.
    /// </summary>
    /// <seealso cref="Keras.Layers.RNN" />
    public class ConvLSTM2D : RNN
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="ConvLSTM2D" /> class.
        /// </summary>
        /// <param name="filters">Integer, the dimensionality of the output space (i.e. the number output of filters in the convolution).</param>
        /// <param name="kernel_size">An integer or tuple/list of n integers, specifying the dimensions of the convolution window.</param>
        /// <param name="strides">An integer or tuple/list of n integers, specifying the strides of the convolution. Specifying any stride value != 1 is incompatible with specifying any dilation_rate value != 1.</param>
        /// <param name="padding">One of "valid" or "same" (case-insensitive).</param>
        /// <param name="data_format">A string, one of "channels_last" (default) or "channels_first". The ordering of the dimensions in the inputs. "channels_last" corresponds to inputs with shape (batch, time, ..., channels) while "channels_first" corresponds to inputs with shape (batch, time, channels, ...). It defaults to the image_data_format value found in your Keras config file at ~/.keras/keras.json. If you never set it, then it will be "channels_last".</param>
        /// <param name="dilation_rate">An integer or tuple/list of n integers, specifying the dilation rate to use for dilated convolution. Currently, specifying any dilation_rate value != 1 is incompatible with specifying any strides value != 1.</param>
        /// <param name="activation">Activation function to use (see activations). If you don't specify anything, no activation is applied (ie. "linear" activation: a(x) = x).</param>
        /// <param name="recurrent_activation">Activation function to use for the recurrent step (see activations).</param>
        /// <param name="use_bias">Boolean, whether the layer uses a bias vector.</param>
        /// <param name="kernel_initializer">Initializer for the kernel weights matrix, used for the linear transformation of the inputs. (see initializers).</param>
        /// <param name="recurrent_initializer">Initializer for the recurrent_kernel weights matrix, used for the linear transformation of the recurrent state. (see initializers).</param>
        /// <param name="bias_initializer">Initializer for the bias vector (see initializers).</param>
        /// <param name="unit_forget_bias">Boolean. If True, add 1 to the bias of the forget gate at initialization. Use in combination with bias_initializer="zeros". This is recommended in Jozefowicz et al. (2015).</param>
        /// <param name="kernel_regularizer">Regularizer function applied to the kernel weights matrix (see regularizer).</param>
        /// <param name="recurrent_regularizer">Regularizer function applied to the recurrent_kernel weights matrix (see regularizer).</param>
        /// <param name="bias_regularizer">Regularizer function applied to the bias vector (see regularizer).</param>
        /// <param name="activity_regularizer">Regularizer function applied to the output of the layer (its "activation"). (see regularizer).</param>
        /// <param name="kernel_constraint">Constraint function applied to the kernel weights matrix (see constraints).</param>
        /// <param name="recurrent_constraint">Constraint function applied to the recurrent_kernel weights matrix (see constraints).</param>
        /// <param name="bias_constraint">Constraint function applied to the bias vector (see constraints).</param>
        /// <param name="return_sequences">Boolean. Whether to return the last output in the output sequence, or the full sequence.</param>
        /// <param name="go_backwards">Boolean (default False). If True, process the input sequence backwards.</param>
        /// <param name="stateful">Boolean (default False). If True, the last state for each sample at index i in a batch will be used as initial state for the sample of index i in the following batch.</param>
        /// <param name="dropout">Float between 0 and 1. Fraction of the units to drop for the linear transformation of the inputs.</param>
        /// <param name="recurrent_dropout">Float between 0 and 1. Fraction of the units to drop for the linear transformation of the recurrent state.</param>
        /// <param name="input_shape">The input shape.</param>

        public ConvLSTM2D(int filters, Tuple<int, int> kernel_size, Tuple<int, int> strides = null, string padding = "valid", string data_format = "",
            Tuple<int, int> dilation_rate = null, string activation = "tanh", string recurrent_activation = "hard_sigmoid", bool use_bias = true,
            string kernel_initializer = "glorot_uniform", string recurrent_initializer = "orthogonal", string bias_initializer = "zeros",
            bool unit_forget_bias = true, string kernel_regularizer = "", string recurrent_regularizer = "", string bias_regularizer = "",
            string activity_regularizer = "", string kernel_constraint = "", string recurrent_constraint = "", string bias_constraint = "",
            bool return_sequences = false, bool go_backwards = false, bool stateful = false, float dropout = 0.0f, float recurrent_dropout = 0.0f, Shape input_shape = null)
        {
            Parameters["filters"] = filters;
            Parameters["kernel_size"] = new Shape(kernel_size.Item1, kernel_size.Item2);
            Parameters["strides"] = strides;
            Parameters["padding"] = padding;
            Parameters["data_format"] = data_format;
            Parameters["dilation_rate"] = dilation_rate != null ? new Shape(dilation_rate.Item1, dilation_rate.Item2) : new Shape(1, 1);
            Parameters["activation"] = activation;
            Parameters["recurrent_activation"] = recurrent_activation;
            Parameters["use_bias"] = use_bias;
            Parameters["kernel_initializer"] = kernel_initializer;
            Parameters["recurrent_initializer"] = recurrent_initializer;
            Parameters["bias_initializer"] = bias_initializer;
            Parameters["unit_forget_bias"] = unit_forget_bias;
            Parameters["kernel_regularizer"] = kernel_regularizer;
            Parameters["recurrent_regularizer"] = recurrent_regularizer;
            Parameters["bias_regularizer"] = bias_regularizer;
            Parameters["activity_regularizer"] = activity_regularizer;
            Parameters["kernel_constraint"] = kernel_constraint;
            Parameters["recurrent_constraint"] = recurrent_constraint;
            Parameters["bias_constraint"] = bias_constraint;
            Parameters["return_sequences"] = return_sequences;
            Parameters["go_backwards"] = go_backwards;
            Parameters["stateful"] = stateful;
            Parameters["dropout"] = dropout;
            Parameters["recurrent_dropout"] = recurrent_dropout;
            Parameters["input_shape"] = input_shape;

            PyInstance = Instance.keras.layers.ConvLSTM2D;
            Init();
        }
    }

    /// <summary>
    /// Cell class for the ConvLSTM2D layer.
    /// </summary>
    /// <seealso cref="Keras.Layers.RNN" />
    public class ConvLSTM2DCell : RNN
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="ConvLSTM2DCell" /> class.
        /// </summary>
        /// <param name="filters">Integer, the dimensionality of the output space (i.e. the number output of filters in the convolution).</param>
        /// <param name="kernel_size">An integer or tuple/list of n integers, specifying the dimensions of the convolution window.</param>
        /// <param name="strides">An integer or tuple/list of n integers, specifying the strides of the convolution. Specifying any stride value != 1 is incompatible with specifying any dilation_rate value != 1.</param>
        /// <param name="padding">One of "valid" or "same" (case-insensitive).</param>
        /// <param name="data_format">A string, one of "channels_last" (default) or "channels_first". The ordering of the dimensions in the inputs. "channels_last" corresponds to inputs with shape (batch, time, ..., channels) while "channels_first" corresponds to inputs with shape (batch, time, channels, ...). It defaults to the image_data_format value found in your Keras config file at ~/.keras/keras.json. If you never set it, then it will be "channels_last".</param>
        /// <param name="dilation_rate">An integer or tuple/list of n integers, specifying the dilation rate to use for dilated convolution. Currently, specifying any dilation_rate value != 1 is incompatible with specifying any strides value != 1.</param>
        /// <param name="activation">Activation function to use (see activations). If you don't specify anything, no activation is applied (ie. "linear" activation: a(x) = x).</param>
        /// <param name="recurrent_activation">Activation function to use for the recurrent step (see activations).</param>
        /// <param name="use_bias">Boolean, whether the layer uses a bias vector.</param>
        /// <param name="kernel_initializer">Initializer for the kernel weights matrix, used for the linear transformation of the inputs. (see initializers).</param>
        /// <param name="recurrent_initializer">Initializer for the recurrent_kernel weights matrix, used for the linear transformation of the recurrent state. (see initializers).</param>
        /// <param name="bias_initializer">Initializer for the bias vector (see initializers).</param>
        /// <param name="unit_forget_bias">Boolean. If True, add 1 to the bias of the forget gate at initialization. Use in combination with bias_initializer="zeros". This is recommended in Jozefowicz et al. (2015).</param>
        /// <param name="kernel_regularizer">Regularizer function applied to the kernel weights matrix (see regularizer).</param>
        /// <param name="recurrent_regularizer">Regularizer function applied to the recurrent_kernel weights matrix (see regularizer).</param>
        /// <param name="bias_regularizer">Regularizer function applied to the bias vector (see regularizer).</param>
        /// <param name="activity_regularizer">Regularizer function applied to the output of the layer (its "activation"). (see regularizer).</param>
        /// <param name="kernel_constraint">Constraint function applied to the kernel weights matrix (see constraints).</param>
        /// <param name="recurrent_constraint">Constraint function applied to the recurrent_kernel weights matrix (see constraints).</param>
        /// <param name="bias_constraint">Constraint function applied to the bias vector (see constraints).</param>
        /// <param name="dropout">Float between 0 and 1. Fraction of the units to drop for the linear transformation of the inputs.</param>
        /// <param name="recurrent_dropout">Float between 0 and 1. Fraction of the units to drop for the linear transformation of the recurrent state.</param>
        /// <param name="input_shape">The input shape.</param>

        public ConvLSTM2DCell(int filters, Tuple<int, int> kernel_size, Tuple<int, int> strides = null, string padding = "valid", string data_format = "",
            Tuple<int, int> dilation_rate = null, string activation = "tanh", string recurrent_activation = "hard_sigmoid", bool use_bias = true,
            string kernel_initializer = "glorot_uniform", string recurrent_initializer = "orthogonal", string bias_initializer = "zeros",
            bool unit_forget_bias = true, string kernel_regularizer = "", string recurrent_regularizer = "", string bias_regularizer = "",
            string activity_regularizer = "", string kernel_constraint = "", string recurrent_constraint = "", string bias_constraint = "",
            float dropout = 0.0f, float recurrent_dropout = 0.0f, Shape input_shape = null)
        {
            Parameters["filters"] = filters;
            Parameters["kernel_size"] = new Shape(kernel_size.Item1, kernel_size.Item2);
            Parameters["strides"] = strides;
            Parameters["padding"] = padding;
            Parameters["data_format"] = data_format;
            Parameters["dilation_rate"] = dilation_rate != null ? new Shape(dilation_rate.Item1, dilation_rate.Item2) : new Shape(1, 1);
            Parameters["activation"] = activation;
            Parameters["recurrent_activation"] = recurrent_activation;
            Parameters["use_bias"] = use_bias;
            Parameters["kernel_initializer"] = kernel_initializer;
            Parameters["recurrent_initializer"] = recurrent_initializer;
            Parameters["bias_initializer"] = bias_initializer;
            Parameters["unit_forget_bias"] = unit_forget_bias;
            Parameters["kernel_regularizer"] = kernel_regularizer;
            Parameters["recurrent_regularizer"] = recurrent_regularizer;
            Parameters["bias_regularizer"] = bias_regularizer;
            Parameters["activity_regularizer"] = activity_regularizer;
            Parameters["kernel_constraint"] = kernel_constraint;
            Parameters["recurrent_constraint"] = recurrent_constraint;
            Parameters["bias_constraint"] = bias_constraint;
            Parameters["dropout"] = dropout;
            Parameters["recurrent_dropout"] = recurrent_dropout;
            Parameters["input_shape"] = input_shape;

            PyInstance = Instance.keras.layers.ConvLSTM2DCell;
            Init();
        }
    }
}
