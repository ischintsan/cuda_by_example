# cuda_by_example
《GPU高性能编程 CUDA实战》(《CUDA By Example an Introduction to General -Purpose GPU Programming》)随书代码

**IDE：** Visual Studio 2019

**CUDA Version：** 11.1

**Base on：**[CodedK/CUDA-by-Example-source-code-for-the-book-s-examples](https://github.com/CodedK/CUDA-by-Example-source-code-for-the-book-s-examples-.git)



**学习过程中遇到的一些坑：**

1. 编译报错：无法打开文件“glut64.lib“
   解决方法：项目->属性->VC++目录->库目录，包含lib文件夹

2. 运行报错：由于找不到glut64.dll，无法继续执行代码。重新安装程序可能会解决此问题。
   解决方法：将bin/glut64.dll文件复制到Debug中
