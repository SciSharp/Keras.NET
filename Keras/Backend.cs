using System;
using System.Collections.Generic;
using System.Text;

namespace Keras
{
    public class Backend : Base
    {
        public static string ImageDataFormat()
        {
            return Instance.self.backend.image_data_format().ToString();
        }
    }
}
