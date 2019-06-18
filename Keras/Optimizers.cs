using System;
using System.Collections.Generic;
using System.Text;

namespace Keras
{
    public class SGD : Base
    {
        public SGD(float lr = 0.01f, float momentum = 0.0f, float decay = 0.0f, bool nesterov = false)
        {
            Parameters["lr"] = lr;
            Parameters["momentum"] = momentum;
            Parameters["decay"] = decay;
            Parameters["nesterov"] = nesterov;

            __self__ = Instance.self.optimizers.SGD;
        }
    }

    public class RMSprop : Base
    {
        public RMSprop(float lr = 0.01f, float rho = 0.9f, float? epsilon = null, float decay = 0.0f)
        {
            Parameters["lr"] = lr;
            Parameters["rho"] = rho;
            Parameters["epsilon"] = epsilon;
            Parameters["decay"] = decay;

            __self__ = Instance.self.optimizers.RMSprop;
        }
    }

    public class Adagrad : Base
    {
        public Adagrad(float lr = 0.01f, float? epsilon = null, float decay = 0.0f)
        {
            Parameters["lr"] = lr;
            Parameters["epsilon"] = epsilon;
            Parameters["decay"] = lr;

            __self__ = Instance.self.optimizers.Adagrad;
        }
    }

    public class Adadelta : Base
    {
        public Adadelta(float lr = 1.0f, float rho = 0.95f, float? epsilon = null, float decay = 0.0f)
        {
            Parameters["lr"] = lr;
            Parameters["rho"] = rho;
            Parameters["epsilon"] = epsilon;
            Parameters["decay"] = decay;

            __self__ = Instance.self.optimizers.Adadelta;
        }
    }

    public class Adam : Base
    {
        public Adam(float lr = 0.001f, float beta_1= 0.9f, float beta_2= 0.999f, float? epsilon = null, float decay = 0.0f)
        {
            Parameters["lr"] = lr;
            Parameters["beta_1"] = beta_1;
            Parameters["beta_2"] = beta_2;
            Parameters["epsilon"] = epsilon;
            Parameters["decay"] = decay;

            __self__ = Instance.self.optimizers.Adam;
        }
    }

    public class Adamax : Base
    {
        public Adamax(float lr = 0.002f, float beta_1 = 0.9f, float beta_2 = 0.999f, float? epsilon = null, float decay = 0.0f)
        {
            Parameters["lr"] = lr;
            Parameters["beta_1"] = beta_1;
            Parameters["beta_2"] = beta_2;
            Parameters["epsilon"] = epsilon;
            Parameters["decay"] = decay;

            __self__ = Instance.self.optimizers.Adamax;
        }
    }

    public class Nadam : Base
    {
        public Nadam(float lr = 0.002f, float beta_1 = 0.9f, float beta_2 = 0.999f, float? epsilon = null, float schedule_decay = 0.004f)
        {
            Parameters["lr"] = lr;
            Parameters["beta_1"] = beta_1;
            Parameters["beta_2"] = beta_2;
            Parameters["epsilon"] = epsilon;
            Parameters["schedule_decay"] = schedule_decay;

            __self__ = Instance.self.optimizers.Adamax;
        }
    }
}
