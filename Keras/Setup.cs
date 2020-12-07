using Python.Runtime;
using System;
using System.Collections.Generic;
using System.Text;

namespace Keras
{
    public enum SetupBackend
    {
        TensorflowCPU = 0,
        TensorflowGPU,
        CNTKCPU,
        CNTKGPU,
        PlaidML
    }

    public class PyModuleInfo
    {
        public string Name { get; set; }

        public string Version { get; set; }

        public string Summary { get; set; }

        public string HomePage { get; set; }

        public string Author { get; set; }

        public string AuthorEmail { get; set; }

        public string License { get; set; }

        public string Location { get; set; }

        public string Requires { get; set; }

        public string RequiredBy { get; set; }
    }

    public class Setup
    {
        private static string[] modules = new string[] { "keras", "numpy" };
        private static string[] backendmodules = new string[] { "tensorflow", "tensorflow-gpu", "cntk", "cntk-gpu", "plaidml-keras" };
        private static string pythonCommand = "python";
        private static string pipCommand = "pip";

        public static string KerasModule { get; set; } = "tensorflow.keras";

        public static void UseTfKeras()
        {
            KerasModule = "tensorflow.keras";
        }

        public static void UseOldKeras()
        {
            KerasModule = "keras";
        }

        public static void Run(SetupBackend backend = SetupBackend.TensorflowCPU)
        {
            int pyversion = CheckPythonVer();
            if (pyversion == 0)
                throw new Exception("Python 3.6 not found! Please download and install from https://www.python.org/downloads/release/python-368/");

            if(pyversion == 36)
            {
                foreach (var item in modules)
                {
                    InstallModule(item);
                }

                InstallModule(backendmodules[(int)backend]);
            }
            else
            {
                throw new Exception("Version not supported: " + pyversion);
            }
        }

        public static void InstallModule(string name)
        {
            if (CheckModule(name) == null)
            {
                Console.WriteLine("Installing {0}.....", name);
                string result = RunCommand(pipCommand, string.Format("install {0}", name));
                Console.Write("Done!");
            }
            else
            {
                Console.WriteLine("{0} already installed!", name);
            }
        }

        public static PyModuleInfo CheckModule(string name)
        {
            int pyversion = CheckPythonVer();
            if (pyversion == 0)
                throw new Exception("Python 3.6 not found");
            PyModuleInfo result = null;
            if (pyversion == 36)
            {
                string info = RunCommand(pipCommand, string.Format("show {0}", name));
                if(!string.IsNullOrWhiteSpace(info))
                {
                    string[] lines = info.Split('\n');
                    result = new PyModuleInfo();
                    foreach (var item in lines)
                    {
                        if(item.Contains("Name: "))
                        {
                            result.Name = item.Replace("Name: ", "").Trim();
                        }

                        if (item.Contains("Version: "))
                        {
                            result.Version = item.Replace("Version: ", "").Trim();
                        }

                        if (item.Contains("Summary: "))
                        {
                            result.Summary = item.Replace("Summary: ", "").Trim();
                        }

                        if (item.Contains("Author: "))
                        {
                            result.Author = item.Replace("Author: ", "").Trim();
                        }

                        if (item.Contains("Author-email: "))
                        {
                            result.AuthorEmail = item.Replace("Author-email: ", "").Trim();
                        }

                        if (item.Contains("License: "))
                        {
                            result.License = item.Replace("License: ", "").Trim();
                        }

                        if (item.Contains("Location: "))
                        {
                            result.Location = item.Replace("Location: ", "").Trim();
                        }

                        if (item.Contains("Requires: "))
                        {
                            result.Requires = item.Replace("Requires: ", "").Trim();
                        }

                        if (item.Contains("Required-by: "))
                        {
                            result.RequiredBy = item.Replace("Required-by: ", "").Trim();
                        }

                        if (item.Contains("Home-page: "))
                        {
                            result.HomePage = item.Replace("Home-page: ", "").Trim();
                        }
                    }
                }
            }
            else
            {
                throw new Exception("Version not supported: " + pyversion);
            }
            return result;
        }

        private static int CheckPythonVer()
        {
            try
            {
                string result = RunCommand("python", "--version");
                string[] versionSplit = result.Replace("Python", "").Trim().Split('.');

                return Convert.ToInt32(versionSplit[0] + versionSplit[1]);
            }
            catch
            {
                try
                {
                    string result = RunCommand("python3", "--version");
                    string[] versionSplit = result.Replace("Python", "").Trim().Split('.');

                    pythonCommand = "python3";
                    pipCommand = "pip3";
                    return Convert.ToInt32(versionSplit[0] + versionSplit[1]);
                }
                catch (Exception ex)
                {
                    return 0;
                }
            }
        }

        private static string RunCommand(string exec, string arguments)
        {
            System.Diagnostics.Process process = new System.Diagnostics.Process();
            System.Diagnostics.ProcessStartInfo startInfo = new System.Diagnostics.ProcessStartInfo();
            startInfo.WindowStyle = System.Diagnostics.ProcessWindowStyle.Hidden;
            startInfo.RedirectStandardOutput = true;
            startInfo.RedirectStandardError = true;
            startInfo.FileName = exec;
            startInfo.Arguments = arguments;
            process.StartInfo = startInfo;
            process.Start();
            string error = process.StandardError.ReadToEnd();
            if (!string.IsNullOrWhiteSpace(error))
                throw new Exception(error);

            return process.StandardOutput.ReadToEnd();
        }

        public static void SetPythonPath(string path)
        {
            Environment.SetEnvironmentVariable("PYTHON_PATH", path);
            Environment.SetEnvironmentVariable("PYTHON_HOME", path);

            Python.Runtime.PythonEngine.PythonHome = path;
            Python.Runtime.PythonEngine.Initialize();
        }
    }
}
