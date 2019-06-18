using Keras.Layers;
using Keras.Models;
using System;
using System.Collections.Generic;
using System.Text;

namespace Keras
{
    public partial class Utils : Base
    {
        public static void PlotModel(BaseModel model, string to_file)
        {
            Instance.self.utils.plot_model(model: model.ToPython(), to_file: to_file);
        }

        public static dynamic ModelToDot(BaseModel model, bool show_shapes = false, bool show_layer_names = true, string rankdir = "TB", bool expand_nested = false, int dpi = 96, bool subgraph = false)
        {
            return Instance.self.utils.model_to_dot(model: model.ToPython(), show_shapes: show_shapes, show_layer_names: show_layer_names, rankdir: rankdir, expand_nested: expand_nested, dpi: dpi, subgraph: subgraph);
        }
    }
}
