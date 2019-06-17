using System;
using System.Collections.Generic;
using System.Text;

namespace Keras
{
    public class Zeros : Base
    {
        public Zeros()
        {
            __self__ = Instance.self.initializers.Zeros;
        }
    }

    public class Ones : Base
    {
        public Ones()
        {
            __self__ = Instance.self.initializers.Zeros;
        }
    }

    public class Constant : Base
    {
        public Constant(float value = 0)
        {
            Parameters["value"] = value;
            __self__ = Instance.self.initializers.Constant;
        }
    }
}
