using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Mvc;

namespace Keras.Playground.Controllers
{
    public class Cifar10Controller : Controller
    {
        public IActionResult Index()
        {
            return View();
        }
    }
}