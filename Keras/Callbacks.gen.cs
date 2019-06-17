using Keras.Layers;
using System;
using System.Collections.Generic;
using System.Text;

namespace Keras
{
    public class Callback : Base
    {
        public Callback()
        {
            __self__ = self.callbacks.Callback;
        }
    }

    public class BaseLogger : Base
    {
        public BaseLogger(List<string> stateful_metrics)
        {
            Parameters["stateful_metrics"] = stateful_metrics;
            __self__ = self.callbacks.BaseLogger;
        }
    }
}
