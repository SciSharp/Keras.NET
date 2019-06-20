namespace Keras.Models
{
    using global::Keras.Layers;
    using System.Collections.Generic;

    /// <summary>
    /// In the functional API, given some input tensor(s) and output tensor(s).
    /// This model will include all layers required in the computation of b given a.
    /// </summary>
    /// <seealso cref="Keras.Models.BaseModel" />
    public class Model : BaseModel
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="Model"/> class.
        /// </summary>
        /// <param name="inputs">The inputs layers.</param>
        /// <param name="outputs">The outputs layers.</param>
        public Model(BaseLayer[] inputs, BaseLayer[] outputs)
        {
            List<object> inputList = new List<object>();
            List<object> outputList = new List<object>();

            foreach (var item in inputs)
            {
                inputList.Add(item.ToPython());
            }

            foreach (var item in outputs)
            {
                outputList.Add(item.ToPython());
            }

            __self__ = Instance.keras.models.Model(inputs: inputs, outputs: outputs);
        }
    }
}
