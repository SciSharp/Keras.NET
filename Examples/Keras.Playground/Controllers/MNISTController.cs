using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;

namespace Keras.Playground.Controllers
{
    public class MNISTController : Controller
    {
        //static BaseModel model = null;
        public IActionResult Index()
        {
            return View();
        }
    }
}