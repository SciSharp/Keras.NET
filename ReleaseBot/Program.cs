using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Xml;
using HtmlAgilityPack;

namespace ReleaseBot
{
    class Program
    {
        private const string V = "4.4"; // <--- Keras.net version!
        private const string NumpyNetVersion = "1.22";

        private const string ProjectPath = "../../../Keras";
        private const string ProjectName = "Keras.csproj";

        private const string Description = "C# bindings for Keras on {0} - Keras.NET is a high-level neural networks API, capable of running on top of TensorFlow, CNTK, or Theano. ";
        private const string Tags = "Data science, Machine Learning, ML, AI, Keras, Neural Network, deep learning";

        static void Main(string[] args)
        {
            //ProcessKeras();
            var specs = new ReleaseSpec[]
           {
                // linux                
                new ReleaseSpec() { CPythonVersion = "2.7", Platform="Linux",   },
                new ReleaseSpec() { CPythonVersion = "3.7", Platform="Linux",   },
                new ReleaseSpec() { CPythonVersion = "3.8", Platform="Linux",   },
                // mac
                new ReleaseSpec() { CPythonVersion = "2.7", Platform="OSX",  },
                new ReleaseSpec() { CPythonVersion = "3.7", Platform="OSX",  },
                new ReleaseSpec() { CPythonVersion = "3.8", Platform="OSX",  },
                // win
                new ReleaseSpec() { CPythonVersion = "2.7", Platform="Win64",   },
                new ReleaseSpec() { CPythonVersion = "3.7", Platform="Win64",   },
                new ReleaseSpec() { CPythonVersion = "3.8", Platform="Win64",   },
           };

            foreach (var spec in specs)
            {
                spec.Version = $"{spec.CPythonVersion}.{V}";
                spec.PythonNetVersion = $"{spec.CPythonVersion}.{NumpyNetVersion}";
                spec.Description = string.Format(Description, spec.Platform, spec.CPythonVersion);
                spec.PackageTags = Tags;
                spec.RelativeProjectPath = ProjectPath;
                spec.ProjectName = ProjectName;
                switch (spec.Platform)
                {
                    case "Linux":
                        spec.PackageId = "Keras.NET.Mono";
                        spec.NumpyNet = "Numpy.Bare.Mono";
                        break;
                    case "OSX":
                        spec.PackageId = "Keras.NET.OSX";
                        spec.NumpyNet = "Numpy.Bare.OSX";
                        break;
                    case "Win64":
                        spec.PackageId = "Keras.NET";
                        spec.NumpyNet = "Numpy.Bare";
                        break;
                }
                spec.Process();
            }

            var key = File.ReadAllText(@"C:\Tools\nuget.key").Trim();
            foreach (var nuget in Directory.GetFiles(Path.Combine(ProjectPath, "bin", "Release"), "*.nupkg"))
            {
                Console.WriteLine("Push " + nuget);
                var arg = $"push -Source https://api.nuget.org/v3/index.json -ApiKey {key} {nuget}";
                var p = new Process() { StartInfo = new ProcessStartInfo("nuget.exe", arg) { RedirectStandardOutput = true, RedirectStandardError = true, UseShellExecute = false } };
                p.OutputDataReceived += (x, data) => Console.WriteLine(data.Data);
                p.ErrorDataReceived += (x, data) => Console.WriteLine("Error: " + data.Data);
                p.Start();
                p.WaitForExit();
                Console.WriteLine("... pushed");
            }
        }

        private static void ProcessKeras()
        {
            var spec = new ReleaseSpec()
            {
                Version = $"3.8.{V}",
                ProjectName = ProjectName,
                RelativeProjectPath = ProjectPath,
                PackageId = "Keras.NET",
                Description = @"C# bindings for Keras on {0} - Keras.NET is a high-level neural networks API, capable of running on top of TensorFlow, CNTK, or Theano. ",
                PackageTags = "Data science, Machine Learning, ML, AI, Keras, Neural Network, deep learning",
                UsePythonIncluded = true,
            };
            spec.Process();
            // nuget
            var key = File.ReadAllText("../../nuget.key").Trim();
            foreach (var nuget in Directory.GetFiles(Path.Combine(ProjectPath, "bin", "Release"), "*.nupkg"))
            {
                Console.WriteLine("Push " + nuget);
                var arg = $"push -Source https://api.nuget.org/v3/index.json -ApiKey {key} {nuget}";
                var p = new Process() { StartInfo = new ProcessStartInfo("nuget.exe", arg) { RedirectStandardOutput = true, RedirectStandardError = true, UseShellExecute = false } };
                p.OutputDataReceived += (x, data) => Console.WriteLine(data.Data);
                p.ErrorDataReceived += (x, data) => Console.WriteLine("Error: " + data.Data);
                p.Start();
                p.WaitForExit();
                Console.WriteLine("... pushed");
            }
        }
    }

    public class ReleaseSpec
    {
        /// <summary>
        /// The assembly / nuget package version
        /// </summary>
        public string Version;

        public string CPythonVersion;
        public string Platform;

        /// <summary>
        /// Project description
        /// </summary>
        public string Description;

        /// <summary>
        /// Project description
        /// </summary>
        public string PackageTags;

        /// <summary>
        /// Nuget package id
        /// </summary>
        public string PackageId;

        /// <summary>
        /// PythonNet package name
        /// </summary>
        public string NumpyNet;

        /// <summary>
        /// PythonNet Version
        /// </summary>
        public string PythonNetVersion;

        /// <summary>
        /// Name of the csproj file
        /// </summary>
        public string ProjectName;

        /// <summary>
        /// Path to the csproj file, relative to the execution directory of ReleaseBot
        /// </summary>
        public string RelativeProjectPath;

        public string FullProjectPath => Path.Combine(RelativeProjectPath, ProjectName);

        /// <summary>
        /// Uses Python.Included
        /// </summary>
        public bool UsePythonIncluded { get; set; }

        public void Process()
        {
            if (!File.Exists(FullProjectPath))
                throw new InvalidOperationException("Project not found at: " + FullProjectPath);
            // modify csproj
            var doc = new HtmlDocument() { OptionOutputOriginalCase = true, OptionWriteEmptyNodes = true };
            doc.Load(FullProjectPath);
            var group0 = doc.DocumentNode.Descendants("propertygroup").FirstOrDefault();
            SetInnerText(group0.Element("version"), Version);
            Console.WriteLine("Version: " + group0.Element("version").InnerText);
            SetInnerText(group0.Element("description"), Description);
            Console.WriteLine("Description: " + group0.Element("description").InnerText);
            if (!UsePythonIncluded)
            {
                SetInnerText(group0.Element("packageid"), PackageId);
                var group1 = doc.DocumentNode.Descendants("itemgroup").FirstOrDefault(g => g.Element("packagereference") != null);
                var reference = group1.Descendants("packagereference").ToArray()[0];
                reference.Attributes["Include"].Value = NumpyNet;
                reference.Attributes["Version"].Value = PythonNetVersion;
            }
            doc.Save(FullProjectPath);
            // now build in release mode
            RestoreNugetDependencies();
            Build();
            //Pack();
        }

        private void RestoreNugetDependencies()
        {
            Console.WriteLine("Fetch Nugets " + Description);
            var p = new Process()
            {
                StartInfo = new ProcessStartInfo("dotnet", "restore")
                { WorkingDirectory = Path.GetFullPath(RelativeProjectPath) }
            };
            p.Start();
            p.WaitForExit();
        }

        private void Build()
        {
            Console.WriteLine("Build " + Description);
            var p = new Process()
            {
                StartInfo = new ProcessStartInfo("dotnet", "msbuild -p:Configuration=Release")
                { WorkingDirectory = Path.GetFullPath(RelativeProjectPath) }
            };
            p.Start();
            p.WaitForExit();
        }

        private void Pack()
        {
            Console.WriteLine("Build " + Description);
            var p = new Process()
            {
                StartInfo = new ProcessStartInfo("dotnet", "pack")
                { WorkingDirectory = Path.GetFullPath(RelativeProjectPath) }
            };
            p.Start();
            p.WaitForExit();
        }

        private void SetInnerText(HtmlNode node, string text)
        {
            node.ReplaceChild(HtmlTextNode.CreateNode(text), node.FirstChild);
        }
    }
}
