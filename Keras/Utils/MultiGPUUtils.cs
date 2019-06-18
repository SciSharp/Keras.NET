using Keras.Models;
using Python.Runtime;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Keras
{
    public partial class Utils : Base
    {
        public static BaseModel MultiGPUModel (BaseModel model, int[] gpus, bool cpu_merge = true, bool cpu_relocation = false)
        {
            BaseModel result = new BaseModel();
            result.__self__ = (PyObject)Instance.self.utils.multi_gpu_model(model: model.ToPython(), gpus: gpus.ToList(), cpu_merge: cpu_merge, cpu_relocation: cpu_relocation);
            return result;
        }
    }
}
