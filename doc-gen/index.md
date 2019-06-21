# An Intro to Amplifier.NET
Amplifier.NET is a high level wrapper to make your life easier which got OpenCL under the hood. it .NET developers to easily run complex applications with intensive mathematical computation on Intel CPU/GPU, NVIDIA, AMD without writing any additional C kernel code. Write your function in .NET and Amplifier will take care of running it on your favorite hardware.

“OpenCL (Open Computing Language) is a framework for writing programs that execute across heterogeneous platforms consisting of central processing units (CPUs), graphics processing units (GPUs), digital signal processors (DSPs), field-programmable gate arrays (FPGAs) and other processors or hardware accelerators. OpenCL specifies programming languages (based on C99 and C++11) for programming these devices and application programming interfaces (APIs) to control the platform and execute programs on the compute devices. OpenCL provides a standard interface for parallel computing using task- and data-based parallelism.”

Mode about OpenCL: https://en.wikipedia.org/wiki/OpenCL

For many years, GPUs have powered the display of images and motion on computer displays, but they are technically capable of doing more. Graphics processors are brought into play when massive calculations are needed on a single task. GPGPU (General-purpose computing on graphics processing units) is currently one of the hot topics you will see in the machine learning world which is used heavily to run matrix and vector computation.

GPU vs CPU cores
A normal CPU will have one or more cores, whereas a GPU has 1000’s of cores. Now think about the amount of parallelism you can achieve by executing the code across the cores, the execution of your algorithm simply multiplies. Although it’s not just simple to write a GPU based code since we have many programming languages which is well polished to execute in CPU, with little more effort you can very well develop your own GPU based programming. In this post, we will uncover the use of Amplifier.NET to write your GPU based program and execute it using C# .NET. Oh yes using C# and not C or C++ which will make the .NET developer life easier.


