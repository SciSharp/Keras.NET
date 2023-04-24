using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace Keras.Optimizers
{
    /// <summary>
    /// Adam optimizer. Default parameters follow those provided in the original paper.
    /// </summary>
    /// <seealso cref="Keras.Base" />
    public class Adam : Base
    {
        public Adam(float learning_rate = 0.001f, float beta_1= 0.9f, float beta_2= 0.999f, float epsilon = 1e-07f, bool amsgrad = false, float? weight_decay = null,
            bool? clipnorm = null, bool? clipvalue = null, float? global_clipnorm = null, bool use_ema = false, float ema_momentum = 0.99f, 
            int? ema_overwrite_frequency = null, bool jit_compile = true)
        {
            Parameters["learning_rate"] = learning_rate;
            Parameters["beta_1"] = beta_1;
            Parameters["beta_2"] = beta_2;
            Parameters["epsilon"] = epsilon;
            Parameters["amsgrad"] = amsgrad;
            Parameters["weight_decay"] = weight_decay;
            Parameters["clipnorm"] = clipnorm;
            Parameters["clipvalue"] = clipvalue;
            Parameters["global_clipnorm"] = global_clipnorm;
            Parameters["use_ema"] = use_ema;
            Parameters["ema_momentum"] = ema_momentum;
            Parameters["ema_overwrite_frequency"] = ema_overwrite_frequency;
            Parameters["jit_compile"] = jit_compile;

            PyInstance = Instance.keras.optimizers.Adam;
            Init();
        }
    }

    /// <summary>
    /// Optimizer that implements the Adafactor algorithm.
    /// </summary>
    /// <seealso cref="Keras.Base" />
    public class Adafactor : Base
    {
        public Adafactor(float learning_rate = 0.001f, float beta_2_decay = -0.8f, float epsilon_1 = 1e-30f, float epsilon_2 = 1e-3f, float clip_threshold = 1, 
            bool relative_step = true, string name = null, float? weight_decay = null, bool? clipnorm = null, bool? clipvalue = null, 
            float? global_clipnorm = null, bool use_ema = false, float ema_momentum = 0.99f, int? ema_overwrite_frequency = null, bool jit_compile = true)
        {
            Parameters["learning_rate"] = learning_rate;
            Parameters["beta_2_decay"] = beta_2_decay;
            Parameters["epsilon_1"] = epsilon_1;
            Parameters["epsilon_2"] = epsilon_2;
            Parameters["clip_threshold"] = clip_threshold;
            Parameters["relative_step"] = relative_step;
            Parameters["name"] = name;
            Parameters["weight_decay"] = weight_decay;
            Parameters["clipnorm"] = clipnorm;
            Parameters["clipvalue"] = clipvalue;
            Parameters["global_clipnorm"] = global_clipnorm;
            Parameters["use_ema"] = use_ema;
            Parameters["ema_momentum"] = ema_momentum;
            Parameters["ema_overwrite_frequency"] = ema_overwrite_frequency;
            Parameters["jit_compile"] = jit_compile;

            PyInstance = Instance.keras.optimizers.Adafactor;
            Init();
        }
    }

    public class AdamW : Base
    {
        public AdamW(float learning_rate = 0.001f, float weight_decay = 0.004f, float beta_1 = 0.9f, float beta_2 = 0.999f, float epsilon = 1e-07f, bool amsgrad = false, 
            bool? clipnorm = null, bool? clipvalue = null, float? global_clipnorm = null, bool use_ema = false, float ema_momentum = 0.99f,
            int? ema_overwrite_frequency = null, bool jit_compile = true)
        {
            Parameters["learning_rate"] = learning_rate;
            Parameters["weight_decay"] = weight_decay;
            Parameters["beta_1"] = beta_1;
            Parameters["beta_2"] = beta_2;
            Parameters["epsilon"] = epsilon;
            Parameters["amsgrad"] = amsgrad;
            Parameters["clipnorm"] = clipnorm;
            Parameters["clipvalue"] = clipvalue;
            Parameters["global_clipnorm"] = global_clipnorm;
            Parameters["use_ema"] = use_ema;
            Parameters["ema_momentum"] = ema_momentum;
            Parameters["ema_overwrite_frequency"] = ema_overwrite_frequency;
            Parameters["jit_compile"] = jit_compile;

            PyInstance = Instance.keras.optimizers.AdamW;
            Init();
        }
    }
}

namespace Keras.Optimizers.Legacy
{
    /// <summary>
    /// Stochastic gradient descent optimizer.    Includes support for momentum, learning rate decay, and Nesterov momentum.
    /// </summary>
    /// <seealso cref="Keras.Base" />
    public class SGD : Base
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="SGD"/> class.
        /// </summary>
        /// <param name="lr">float >= 0. Learning rate.</param>
        /// <param name="momentum">float >= 0. Parameter that accelerates SGD in the relevant direction and dampens oscillations.</param>
        /// <param name="decay"> float >= 0. Learning rate decay over each update.</param>
        /// <param name="nesterov">boolean. Whether to apply Nesterov momentum.</param>
        public SGD(float lr = 0.01f, float momentum = 0.0f, float decay = 0.0f, bool nesterov = false)
        {
            Parameters["lr"] = lr;
            Parameters["momentum"] = momentum;
            Parameters["decay"] = decay;
            Parameters["nesterov"] = nesterov;

            PyInstance = Instance.keras.optimizers.legacy.SGD;
            Init();
        }
    }

    /// <summary>
    /// RMSProp optimizer.    It is recommended to leave the parameters of this optimizer at their default values (except the learning rate, which can be freely tuned).
    /// This optimizer is usually a good choice for recurrent neural networks.
    /// </summary>
    /// <seealso cref="Keras.Base" />
    public class RMSprop : Base
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="RMSprop"/> class.
        /// </summary>
        /// <param name="lr">float >= 0. Learning rate.</param>
        /// <param name="rho">The rho factor. float > 0</param>
        /// <param name="epsilon">float >= 0. Fuzz factor. If None, defaults to K.epsilon().</param>
        /// <param name="decay">float >= 0. Learning rate decay over each update.</param>
        public RMSprop(float lr = 0.01f, float rho = 0.9f, float? epsilon = null, float decay = 0.0f)
        {
            Parameters["lr"] = lr;
            Parameters["rho"] = rho;
            Parameters["epsilon"] = epsilon;
            Parameters["decay"] = decay;

            PyInstance = Instance.keras.optimizers.legacy.RMSprop;
            Init();
        }
    }

    /// <summary>
    /// Adagrad is an optimizer with parameter-specific learning rates, which are adapted relative to how frequently a parameter gets updated during training. The more updates a parameter receives, the smaller the learning rate.
    /// </summary>
    /// <seealso cref="Keras.Base" />
    public class Adagrad : Base
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="Adagrad"/> class.
        /// </summary>
        /// <param name="lr">float >= 0. Initial learning rate.</param>
        /// <param name="epsilon">float >= 0. If None, defaults to K.epsilon().</param>
        /// <param name="decay">float >= 0. Learning rate decay over each update..</param>
        public Adagrad(float lr = 0.01f, float? epsilon = null, float decay = 0.0f)
        {
            Parameters["lr"] = lr;
            Parameters["epsilon"] = epsilon;
            Parameters["decay"] = lr;

            PyInstance = Instance.keras.optimizers.legacy.Adagrad;
            Init();
        }
    }

    /// <summary>
    /// Adadelta is a more robust extension of Adagrad that adapts learning rates based on a moving window of gradient updates, instead of accumulating all past gradients. This way, Adadelta continues learning even when many updates have been done. Compared to Adagrad, in the original version of Adadelta you don't have to set an initial learning rate. In this version, initial learning rate and decay factor can be set, as in most other Keras optimizers.
    /// </summary>
    /// <seealso cref="Keras.Base" />
    public class Adadelta : Base
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="Adadelta"/> class.
        /// </summary>
        /// <param name="lr">float >= 0. Initial learning rate, defaults to 1. It is recommended to leave it at the default value.</param>
        /// <param name="rho">float >= 0. Adadelta decay factor, corresponding to fraction of gradient to keep at each time step.</param>
        /// <param name="epsilon">float >= 0. Fuzz factor. If None, defaults to K.epsilon().</param>
        /// <param name="decay">float >= 0. Initial learning rate decay.</param>
        public Adadelta(float lr = 1.0f, float rho = 0.95f, float? epsilon = null, float decay = 0.0f)
        {
            Parameters["lr"] = lr;
            Parameters["rho"] = rho;
            Parameters["epsilon"] = epsilon;
            Parameters["decay"] = decay;

            PyInstance = Instance.keras.optimizers.legacy.Adadelta;
            Init();
        }
    }

    /// <summary>
    /// Adam optimizer. Default parameters follow those provided in the original paper.
    /// </summary>
    /// <seealso cref="Keras.Base" />
    public class Adam : Base
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="Adam"/> class.
        /// </summary>
        /// <param name="lr">The lr.</param>
        /// <param name="beta_1">The beta 1.</param>
        /// <param name="beta_2">The beta 2.</param>
        /// <param name="epsilon">The epsilon.</param>
        /// <param name="decay">The decay.</param>
        /// <param name="amsgrad">boolean. Whether to apply the AMSGrad variant of this algorithm from the paper "On the Convergence of Adam and Beyond".</param>
        public Adam(float lr = 0.001f, float beta_1 = 0.9f, float beta_2 = 0.999f, float? epsilon = null, float decay = 0.0f, bool amsgrad = false)
        {
            Parameters["lr"] = lr;
            Parameters["beta_1"] = beta_1;
            Parameters["beta_2"] = beta_2;
            Parameters["epsilon"] = epsilon;
            Parameters["decay"] = decay;
            Parameters["amsgrad"] = amsgrad;

            PyInstance = Instance.keras.optimizers.legacy.Adam;
            Init();
        }
    }

    /// <summary>
    /// Adamax optimizer from Adam paper's Section 7. It is a variant of Adam based on the infinity norm.Default parameters follow those provided in the paper.
    /// </summary>
    /// <seealso cref="Keras.Base" />
    public class Adamax : Base
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="Adamax"/> class.
        /// </summary>
        /// <param name="lr">float >= 0. Learning rate.</param>
        /// <param name="beta_1">floats, 0 < beta < 1. Generally close to 1.</param>
        /// <param name="beta_2">floats, 0 < beta < 1. Generally close to 1.</param>
        /// <param name="epsilon">float >= 0. Fuzz factor. If None, defaults to K.epsilon().</param>
        /// <param name="decay">float >= 0. Learning rate decay over each update.</param>
        public Adamax(float lr = 0.002f, float beta_1 = 0.9f, float beta_2 = 0.999f, float? epsilon = null, float decay = 0.0f)
        {
            Parameters["lr"] = lr;
            Parameters["beta_1"] = beta_1;
            Parameters["beta_2"] = beta_2;
            Parameters["epsilon"] = epsilon;
            Parameters["decay"] = decay;

            PyInstance = Instance.keras.optimizers.legacy.Adamax;
            Init();
        }
    }

    /// <summary>
    /// Nesterov Adam optimizer.    Much like Adam is essentially RMSprop with momentum, Nadam is Adam RMSprop with Nesterov momentum.
    /// Default parameters follow those provided in the paper.It is recommended to leave the parameters of this optimizer at their default values.
    /// </summary>
    /// <seealso cref="Keras.Base" />
    public class Nadam : Base
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="Nadam"/> class.
        /// </summary>
        /// <param name="lr">float >= 0. Learning rate.</param>
        /// <param name="beta_1">floats, 0 < beta < 1. Generally close to 1.</param>
        /// <param name="beta_2">floats, 0 < beta < 1. Generally close to 1.</param>
        /// <param name="epsilon">float >= 0. Fuzz factor. If None, defaults to K.epsilon().</param>
        /// <param name="schedule_decay">floats, 0 < schedule_decay < 1.</param>
        public Nadam(float lr = 0.002f, float beta_1 = 0.9f, float beta_2 = 0.999f)
        {
            Parameters["lr"] = lr;
            Parameters["beta_1"] = beta_1;
            Parameters["beta_2"] = beta_2;

            PyInstance = Instance.keras.optimizers.legacy.Nadam;
            Init();
        }
    }


    /// <summary>
    /// "Follow The Regularized Leader" (FTRL) is an optimization algorithm developed at Google for click-through rate prediction in the early 2010s.  
    /// It is most suitable for shallow models with large and sparse feature spaces.
    /// </summary>
    /// <seealso cref="Keras.Base" />
    public class Ftrl : Base
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="Nadam"/> class.
        /// </summary>
        /// <param name="lr">float >= 0. Learning rate.</param>
        /// <param name="lrp">float >= 0. Learning rate Power.</param>
        /// <param name="iav">float <= 0. Initial Accumulator Value.</param>
        /// <param name="l1rs">float <= 0. Lambda 1 Regularization Strength.</param>
        /// <param name="l2rs">float <= 0. Lambda 2 Regularization Strength.</param>
        /// <param name="l2srs">float <= 0. Lambda 2 Shrinkage Regularization Strength.</param>
        /// <param name="beta">floats, 0 < beta < 1. Generally close to 1.</param>
        public Ftrl(float lr = 0.001f, float lrp = -0.5f, float iav = 0.1f, float l1rs = 0.0f, float l2rs = 0.0f, float l2srs = 0.0f, float beta = 0.0f)
        {
            Parameters["learning_rate"] = lr;
            Parameters["learning_rate_power"] = lrp;
            Parameters["initial_accumulator_value"] = iav;
            Parameters["l1_regularization_strength"] = l1rs;
            Parameters["l2_regularization_strength"] = l2rs;
            Parameters["l2_shrinkage_regularization_strength"] = l2srs;
            Parameters["beta"] = beta;

            PyInstance = Instance.keras.optimizers.legacy.Ftrl;
            Init();
        }
    }
}
