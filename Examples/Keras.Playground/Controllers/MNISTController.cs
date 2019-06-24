using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Keras.Models;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Numpy;
using Python.Runtime;

namespace Keras.Playground.Controllers
{
    public class MNISTController : Controller
    {
        //static BaseModel model = null;
        public IActionResult Index()
        {
            return View();
        }

        [HttpPost]
        public IActionResult Index(IFormCollection form)
        {
            string mnist_data = form["mnist_input"];
            
            using (var state = Py.GIL())
            {
                //Get data to Numpy array
                float[] data = mnist_data.Split(',').Select(i => (Convert.ToSingle(i))).ToArray();
                NDarray x = np.array(data);
                x = x.reshape(1, 28, 28, 1);
                //Load saved model
                var model = Sequential.ModelFromJson(System.IO.File.ReadAllText("./MNIST/model.json"));
                model.LoadWeight("./MNIST/weights.h5");
                //if (model == null)
                //{
                //    model = Sequential.ModelFromJson(System.IO.File.ReadAllText("./MNIST/model.json"));
                //    model.LoadWeight("./MNIST/weights.h5");
                //}

                NDarray y = model.Predict(x);
                ViewBag.Accuracy = y.max().asscalar<float>() * 100;
                y = y.argmax();
                ViewBag.Result = y.asscalar<int>();
            }

            return View();
        }
    }
}