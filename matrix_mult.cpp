#define __CL_ENABLE_EXCEPTIONS

#include <CL/cl.hpp>

#include <fstream>
#include <iostream>
#include <vector>
//#include "read_weights.h"
#include "on_host.h"
#include <time.h>

#include <cstring>
#include <stdlib.h>
#include <bits/stdc++.h>

/* Utility functions */
#include "utils.h"
#include "bmp-utils.h"

#include <stdint.h>


// #define W_FILE_SIZE  30730000

// int main()
// {
// int imageRows = 32;
// int imageCols = 32;
// int imageDepth = 3;

// uint8_t *weight_buffer    = (uint8_t *) malloc(sizeof(uint8_t)*W_FILE_SIZE);
// uint8_t *hInputArray = new uint8_t [imageRows*imageCols*imageDepth];

// std::ifstream in("test_batch.bin",std::ios_base::binary);
// in.read((char *)weight_buffer,sizeof(uint8_t)*W_FILE_SIZE);

// for (int i=0; i<20; i++) {
// for (int j=0; j<1; j++) {
// //hInputArray[j] = 
// std::cout<<(int)weight_buffer[i*3073 + j]<<" ";
// }
// std::cout<<"\n";
// }

  //ifstream infile;
  //infile.open("test_batch.bin", ios::binary | ios::in);
// rest of program

// }
// features.0.weight torch.Size([3, 64, 3, 3])
// features.2.weight torch.Size([64, 64, 3, 3])

// features.5.weight torch.Size([64, 128, 3, 3])
// features.7.weight torch.Size([128, 128, 3, 3])

// features.10.weight torch.Size([128, 256, 3, 3])
// features.12.weight torch.Size([256, 256, 3, 3])

// features.15.weight torch.Size([256, 512, 3, 3])
// features.17.weight torch.Size([512, 512, 3, 3])

// features.20.weight torch.Size([512, 512, 3, 3])
// features.22.weight torch.Size([512, 512, 3, 3])
// 10
// classifier.0.weight torch.Size([512, 1024])
// classifier.0.bias torch.Size([1024])
// classifier.1.weight torch.Size([1024])
// classifier.1.bias torch.Size([1024])
// classifier.1.running_mean torch.Size([1024])
// classifier.1.running_var torch.Size([1024])
// 6
// classifier.4.weight torch.Size([1024, 1024])
// classifier.4.bias torch.Size([1024])
// classifier.5.weight torch.Size([1024])
// classifier.5.bias torch.Size([1024])
// classifier.5.running_mean torch.Size([1024])
// classifier.5.running_var torch.Size([1024])
// 12
// classifier.8.weight torch.Size([1024, 10])
// classifier.8.bias torch.Size([10])

#define MAX_FMAP_CONV 28*28*20
#define OUTPUT_BUFFER_SIZE 10
#define INPUT_BUFFER_SIZE 28*28*1
#define INPUT_FILE_IMAGE 28*28*1
#define W_FILE_SIZE 10994378
int weights_size[] = {3*64*3*3, 64*64*3*3, 64*128*3*3, 128*128*3*3, 128*256*3*3, 256*256*3*3, 
                      256*512*3*3, 512*512*3*3, 512*512*3*3, 512*512*3*3, 
                      512*1024, 1024, 
                      1024, 1024, 1024, 1024, 
                      1024*1024, 1024, 
                      1024, 1024, 1024, 1024, 
                      10*1024, 10};
int inputDepth[] = {3, 64, 64, 128, 128, 256, 256, 512, 512, 512};
int ChannelDepth[] = {64, 64, 128, 128, 256, 256, 512, 512, 512, 512};
#define ROW 32
#define COL 32
#define DAT_FILE_SIZE  30730000


/* Utility Functions */
void fillMatrix(float* arr, int var, int rows, int cols, int depth)
{
  // int rows = ROW;
  // int cols = COL;  
  if (var == 1) {
    for (int i=0; i<rows*cols*depth; i++) {
      arr[i] = rand();

    }
  }
  else if (var == 2) {
    for (int i=0; i<rows*cols*depth; i++) {
      arr[i] = rand();
    }
  }
  if (var == 3) {
    for (int i=0; i<rows*cols*depth; i++) {
      arr[i] = rand()/1000000;
    }
  }

}

void printArray(float* arr)
{
  for (int i=0; i<1; i++) {
    for (int j=0; j<512; j++) {
      std::cout<<arr[i*1 + j]<<" ";
    }
    std::cout<<"\n";
  }
  std::cout<<"\n";
}

int InitializeOutputArray(float *array, int rows, int cols) {
    for (int i=0;  i<rows*cols; i++) {
        array[i] = 0;
    }
}

/* Utility Functions */
void fillVector(float arr[], int rows, int cols)
{
  for (int i=0; i<rows*cols; i++) {
    arr[i] = (-1) * (rand() % 10);
    if (i % 3 == 0) {
      arr[i] = arr[i]*(-1);
    }
    //printf("%f\n", arr[i]);
  }
}

int main() 
{

  int M = 1;
  int K = 128;
  int N = 256;
  
     /* Host data */
   float *hInpImage = NULL; 
   //float *hOutputImage = NULL;

   /* Allocate space for the input image and read the
    * data from disk */
   // int imgRows;
   // int imgCols;
   // hInpImage = readBmpFloat("cat-face.bmp", &imgRows, &imgCols);
   // std::cout<<imgRows<<imgCols;

  const int size = ROW*COL;
  int imageRows = ROW;
  int imageCols = COL;
  int imageDepth = 3;
  int outputDepth = 64;
  int filterRows = 3;
  int filterCols = 3;
  int tc = 3;

  // std::cout<<"Select test-case to be executed: (1/2/3) \n";
  // std::cin>>tc;

float *hInputArray = new float [imageRows*imageCols*imageDepth];
float *hkernel = new float [filterRows*filterCols*imageDepth*outputDepth];
float *hOArray = new float [imageRows*imageCols*outputDepth];
float *hfinalOutput = new float [(imageRows/2)*(imageCols/2)*outputDepth];


uint8_t *dataset_buffer    = (uint8_t *) malloc(sizeof(uint8_t)*DAT_FILE_SIZE);
//uint8_t *hInputArray = new uint8_t [imageRows*imageCols*imageDepth];

std::ifstream ind("cifar-10-batches-bin/test_batch.bin",std::ios_base::binary);
ind.read((char *)dataset_buffer,sizeof(uint8_t)*DAT_FILE_SIZE);


float *input_buffer     = (float *) malloc(sizeof(float)*INPUT_BUFFER_SIZE);
float *output_buffer    = (float *) malloc(sizeof(float)*OUTPUT_BUFFER_SIZE);
float *input_file_buffer= (float *) malloc(sizeof(float)*INPUT_FILE_IMAGE);
float *weight_buffer    = (float *) malloc(sizeof(float)*W_FILE_SIZE);

   float count = 0;
    // Check for failed memory allocation
    if((input_buffer == NULL) || (output_buffer == NULL) || (input_file_buffer == NULL)){
        std::cout << "TEST FAILED : Failed to allocate memory" << std::endl;
        return -1;
    }

std::cout<<"Loading weights into DDR memory"<<std::endl;
std::cout<<"Completed loading buffers"<<std::endl;

// loading weights into DDR Memory
std::ifstream in("weights_vgg.dat",std::ios_base::binary);
in.read((char *)weight_buffer,sizeof(float)*W_FILE_SIZE);
std::cout<<"Initializing weight buffers for  each layers"<<std::endl;

int number_of_layers = 20;
int number_of_conv_layers = 10;
int number_of_fc_layers = 14; //3 layers

float *conv_layer[number_of_conv_layers];
float *fc_layer[number_of_fc_layers];

long long size_so_far = 0;
int i=0;
for (; i<number_of_conv_layers; i++) {
  int Cin = inputDepth[i];
  int Cout = ChannelDepth[i];
  //std::cout<<i<<" "<<size_so_far<<"\n";
  conv_layer[i] = new float [9*Cin*Cout];
  for (int j=0; j<weights_size[i]; j++) {
    conv_layer[i][j] = weight_buffer[j + size_so_far];
    //std::cout<<conv_layer[i][j];
  }
  //layer[i] = this_layer;
  size_so_far += weights_size[i];
}
//std::cout<<size_so_far<<" ";

//int f = 0;
for (int f = 0; f<number_of_fc_layers; f++) {
  //std::cout<<f<<" "<<size_so_far<<"\n";
  fc_layer[f] = (float*) malloc(sizeof(float)*weights_size[i + f]);
  //fc_layer[f] = new float []
  for (int j=0; j<weights_size[i + f]; j++) {
    //std::cout<<weight_buffer[j + size_so_far]<<" ";
    //std::cout<<size_so_far<<" ";
    fc_layer[f][j] = weight_buffer[j + size_so_far];
    //std::cout<<layer[i][j];
  }
  //layer[i] = this_layer;
  size_so_far += weights_size[i + f];
  //f++;
  //std::cout<<size_so_far<<" ";
}

  M = 1;
  K = 32*32;
  N = 1024;

  float *hInputA = new float [M*K];

  float *hInputB = new float [K*N];

  float *hOutputArray = new float [M*N];

  float *hInputbias = new float[1*N]; 
  /* Fill matrix with random data */
  fillVector(hInputA, M, K);
  fillVector(hInputB, K, N);
  fillVector(hInputbias, 1, N);

  //fillMatrix(hInputArray, 3, imageRows, imageCols, imageDepth);
  fillMatrix(hkernel, 1, 3, 3, imageDepth*outputDepth);
  //printArray(hInputArray);    
  InitializeOutputArray(hOutputArray, ROW, COL);
  int correct = 0;

  try 
  {
    /* Query for platforms */
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    /* Get a list of devices on this platform */
    std::vector<cl::Device> devices;
    platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);

    /* Create a context for the devices */
    cl::Context context(devices);
    
    /* Create a command queue for the first device */
    cl::CommandQueue queue = cl::CommandQueue(context, devices[0]);
    cl::ImageFormat imageFormat = cl::ImageFormat(CL_R, CL_FLOAT);

    // Testing 3D images
    cl::Image3D inputFilter = cl::Image3D(context, CL_MEM_READ_ONLY,
         imageFormat, filterCols*filterRows, inputDepth[0], ChannelDepth[0]);

    cl::Image3D interImage = cl::Image3D(context, CL_MEM_READ_WRITE,
         imageFormat, imageCols, imageRows, ChannelDepth[0]);

    cl::Image3D downsampledImage = cl::Image3D(context, CL_MEM_READ_WRITE,
         imageFormat, (imageCols/2), (imageRows/2), ChannelDepth[0]);


    cl::Buffer inputImageA = cl::Buffer(context, CL_MEM_READ_ONLY,
         M*K*sizeof(float));

    /* Copy the input data to the input image */
    cl::size_t<3> origin;
    origin[0] = 0;
    origin[1] = 0;
    origin[2] = 0;

    cl::size_t<3> region;
    region[0] = imageCols;
    region[1] = imageRows;
    region[2] = imageDepth;

    cl::size_t<3> region_filter;
    region_filter[0] = filterCols*filterRows;
    region_filter[1] = inputDepth[0];
    region_filter[2] = ChannelDepth[0];
    
    // for (int i=0; i<10; i++) {
    //   for (int j=0; j<10; j++) {
    //     std::cout<<conv_layer[i][j]<<"asf ";
    //   }
    // }
    queue.enqueueWriteImage(inputFilter, CL_TRUE, origin, region_filter, 
                            filterCols*filterRows*sizeof(float), inputDepth[0]*filterRows*filterCols*sizeof(float),
                            conv_layer[0]);
    // test_r1[0] = 3*3;
    // test_r1[1] = 3;
    // test_r1[2] = 64;
    // float *honvOutputcheck = new float[3*3*3*64];
    // queue.enqueueReadImage(inputFilter, CL_TRUE, origin, region_filter, 
    //                        3*3*sizeof(float), 3*3*3*sizeof(float),
    //                        honvOutputcheck);
    // printArray(honvOutputcheck); 
    std::vector <cl::Buffer> device_layer;

    for (int i=0; i<number_of_fc_layers; i++) {
      //std::cout<<weights_size[i + 10]<<" ";
      device_layer.push_back(cl::Buffer(context, CL_MEM_READ_ONLY,
           weights_size[i + 10]*sizeof(float)));      
    }
    cl::Buffer outputImage = cl::Buffer(context, CL_MEM_WRITE_ONLY,
         1024*sizeof(float));

    for (int i=0; i<number_of_fc_layers; i++) {
      queue.enqueueWriteBuffer(device_layer[i], CL_TRUE, 0,
           weights_size[i + 10]*sizeof(float), fc_layer[i]);      
    }

    /* Read the program source */
    std::ifstream sourceFile("matrix_mult.cl");
    std::string sourceCode(
       std::istreambuf_iterator<char>(sourceFile),
       (std::istreambuf_iterator<char>()));
    //printf("%s\n", sour);
    cl::Program::Sources source(1,
       std::make_pair(sourceCode.c_str(),
       sourceCode.length() + 1));

    /* Make program from the source code */
    cl::Program program = cl::Program(context, source);
    
    /* Build the program for the devices */
    program.build(devices);
    
    std::vector<cl::Event> v;

    cl::Event  writeEvent,  kernelEvent0, kernelEvent1, kernelEvent2, readEvent;
    v.insert(v.begin(), kernelEvent0);
    // Create the kernel 
    cl::Kernel kernel0(program, "linear");
    cl::Kernel kernel1(program, "batchNorm");
    
    cl::Kernel kernel2(program, "linear");
    cl::Kernel kernel3(program, "batchNorm");

    cl::Kernel kernel4(program, "linear");
    cl::Kernel kernel5(program, "log_softmax");

    cl::Kernel Conv2D(program, "median_histogram");
    cl::Kernel maxpool(program, "maxpool");


    cl::Kernel kernel[20];
    cl::Event kernelEvent[20];    
    cl::Kernel conv2D[number_of_conv_layers];
    cl::Kernel MaxPool[number_of_conv_layers];

    cl_int a;

    cl::Image3D finalOutput[number_of_conv_layers + 1];
    finalOutput[0] = cl::Image3D(context, CL_MEM_READ_WRITE,
     imageFormat, imageCols, imageRows, ChannelDepth[0]);

    for (int o=0; o<1; o++) {
    int label = (int)dataset_buffer[o];
      std::cout<<"Test sample: "<<(o + 1);
    for (int j=1; j<3073; j++) {
    hInputArray[j-1] = (int) dataset_buffer[o*3073 + j]; 
    // std::cout<<(int)weight_buffer[i*3073 + j]<<" ";
     }
    std::cout<<"\n";
    
    /* Create the images */
    cl::Image3D inputImage = cl::Image3D(context, CL_MEM_READ_ONLY,
         imageFormat, imageCols, imageRows, imageDepth);

    queue.enqueueWriteImage(inputImage, CL_TRUE, origin, region, 
                            imageCols*sizeof(float), imageRows*imageCols*sizeof(float),
                            hInputArray);    

    // A = 1xK and W = KxN | B = 1xN 
    Conv2D.setArg(0, inputImage);
    Conv2D.setArg(1, inputFilter);
    Conv2D.setArg(2, imageRows);
    Conv2D.setArg(3, filterRows);
    Conv2D.setArg(4, imageDepth);
    Conv2D.setArg(5, ChannelDepth[0]);
    Conv2D.setArg(6, finalOutput[0]);

    queue.enqueueNDRangeKernel(Conv2D, cl::NullRange, cl::NDRange(imageCols, imageCols), cl::NDRange(1,1));
        cl::size_t<3> test_r1;
  
    // test_r1[0] = imageCols;
    // test_r1[1] = imageRows;
    // test_r1[2] = 64;
    // float *honvOutput = new float[imageRows*imageCols*64];
    // queue.enqueueReadImage(finalOutput[0], CL_TRUE, origin, test_r1, 
    //                        imageCols*sizeof(float), imageRows*imageCols*sizeof(float),
    //                        honvOutput);
    // printArray(honvOutput); 

    cl::Image3D currentInput[number_of_conv_layers];
    cl::Image3D filter[number_of_conv_layers];
    cl::Image3D currentOutput[number_of_conv_layers];
    
    int i=0;
    int startImageCols = imageCols;
    int startImageRows = imageRows;

    for (; i<number_of_conv_layers - 1; i++) {
      //std::cout<<i<<" ";
      if (i%2 == 1) {
      imageCols = imageCols/2;
      imageRows = imageRows/2;
      }
      // Testing 3D images
      filter[i] = cl::Image3D(context, CL_MEM_READ_ONLY,
           imageFormat, filterCols*filterRows, inputDepth[i + 1], ChannelDepth[i + 1]);

      currentOutput[i] = cl::Image3D(context, CL_MEM_READ_WRITE,
           imageFormat, imageCols, imageRows, ChannelDepth[i + 1]);
      
      if (i%2 == 0) {
      finalOutput[i + 1] = cl::Image3D(context, CL_MEM_READ_WRITE,
           imageFormat, imageCols/2, imageRows/2, ChannelDepth[i + 1]);
      }
      else {
      finalOutput[i + 1] = cl::Image3D(context, CL_MEM_READ_WRITE,
           imageFormat, imageCols, imageRows, ChannelDepth[i + 1]);
      }

      cl::size_t<3> region_filter[number_of_conv_layers];
      region_filter[i][0] = filterCols*filterRows;
      region_filter[i][1] = inputDepth[i + 1];
      region_filter[i][2] = ChannelDepth[i + 1];

      queue.enqueueWriteImage(filter[i], CL_TRUE, origin, region_filter[i], 
                              filterCols*filterRows*sizeof(float), inputDepth[i + 1]*filterRows*filterCols*sizeof(float),
                              conv_layer[i + 1]);
      
      conv2D[i] = cl::Kernel(program, "median_histogram");
      
      if (i%2 == 0) {
      MaxPool[i] = cl::Kernel(program, "maxpool");
      conv2D[i].setArg(0, finalOutput[i]);
      conv2D[i].setArg(1, filter[i]);
      conv2D[i].setArg(2, imageRows);
      conv2D[i].setArg(3, filterRows);
      conv2D[i].setArg(4, inputDepth[i + 1]);
      conv2D[i].setArg(5, ChannelDepth[i + 1]);
      conv2D[i].setArg(6, currentOutput[i]);


      MaxPool[i].setArg(0, currentOutput[i]);
      MaxPool[i].setArg(1, imageRows/2);
      MaxPool[i].setArg(2, ChannelDepth[i + 1]);
      MaxPool[i].setArg(3, finalOutput[i + 1]);
      }

      else {
              conv2D[i].setArg(0, finalOutput[i]);
      conv2D[i].setArg(1, filter[i]);
      conv2D[i].setArg(2, imageRows);
      conv2D[i].setArg(3, filterRows);
      conv2D[i].setArg(4, inputDepth[i + 1]);
      conv2D[i].setArg(5, ChannelDepth[i + 1]);
      conv2D[i].setArg(6, finalOutput[i + 1]);

      }
      // queue.enqueueNDRangeKernel(conv2D[i], cl::NullRange, cl::NDRange(imageCols, imageCols), cl::NDRange(1,1));
      // if (i%2 == 0) {
      // queue.enqueueNDRangeKernel(MaxPool[i], cl::NullRange, cl::NDRange(imageCols/2, imageCols/2), cl::NDRange(1,1));
      // }
  }

    clock_t start, end;
    start = clock(); 

// int inputDepth[] = {3, 64, 64, 128, 128, 256, 256, 512, 512, 512};
// int ChannelDepth[] = {64, 64, 128, 128, 256, 256, 512, 512, 512, 512};
    queue.enqueueNDRangeKernel(conv2D[0], cl::NullRange, cl::NDRange(startImageCols, startImageRows), 
                               cl::NDRange(1,1));
    queue.enqueueNDRangeKernel(MaxPool[0], cl::NullRange, cl::NDRange(startImageCols/2, startImageRows/2), 
                               cl::NDRange(1,1));
    startImageCols = startImageCols/2;
    startImageRows = startImageRows/2;

        ///cl::size_t<3> test_r1;
  
    // test_r1[0] = imageCols;
    // test_r1[1] = imageRows;
    // test_r1[2] = 64;
    // float *honvOutput = new float[imageRows*imageCols*64];
    // queue.enqueueReadImage(finalOutput[0], CL_TRUE, origin, test_r1, 
    //                        imageCols*sizeof(float), imageRows*imageCols*sizeof(float),
    //                        honvOutput);
    // printArray(honvOutput); 
    // cl::size_t<3> test_r1;
  
    // test_r1[0] = startImageCols;
    // test_r1[1] = startImageRows;
    // test_r1[2] = 64;
    // float *honvOutput = new float[startImageCols*startImageRows*64];
    // queue.enqueueReadImage(finalOutput[1], CL_TRUE, origin, test_r1, 
    //                        startImageCols*sizeof(float), startImageRows*startImageCols*sizeof(float),
    //                        honvOutput);
    // printArray(honvOutput); 

    queue.enqueueNDRangeKernel(conv2D[1], cl::NullRange, cl::NDRange(startImageCols, startImageRows), 
                               cl::NDRange(1,1));
    queue.enqueueNDRangeKernel(conv2D[2], cl::NullRange, cl::NDRange(startImageCols, startImageRows), 
                               cl::NDRange(1,1));
    queue.enqueueNDRangeKernel(MaxPool[2], cl::NullRange, cl::NDRange(startImageCols/2, startImageRows/2), 
                               cl::NDRange(1,1));
    startImageCols = startImageCols/2;
    startImageRows = startImageRows/2;

    // cl::size_t<3> test_r1;
  
    // test_r1[0] = startImageCols;
    // test_r1[1] = startImageRows;
    // test_r1[2] = 128;
    // float *honvOutput = new float[startImageCols*startImageRows*128];
    // queue.enqueueReadImage(finalOutput[3], CL_TRUE, origin, test_r1, 
    //                        startImageCols*sizeof(float), startImageRows*startImageCols*sizeof(float),
    //                        honvOutput);
    // printArray(honvOutput); 

    queue.enqueueNDRangeKernel(conv2D[3], cl::NullRange, cl::NDRange(startImageCols, startImageRows), 
                               cl::NDRange(1,1));
    queue.enqueueNDRangeKernel(conv2D[4], cl::NullRange, cl::NDRange(startImageCols, startImageRows), 
                               cl::NDRange(1,1));
    queue.enqueueNDRangeKernel(MaxPool[4], cl::NullRange, cl::NDRange(startImageCols/2, startImageRows/2), 
                               cl::NDRange(1,1));

    startImageCols = startImageCols/2;
    startImageRows = startImageRows/2;

    // cl::size_t<3> test_r1;
  
    // test_r1[0] = startImageCols;
    // test_r1[1] = startImageRows;
    // test_r1[2] = 256;
    // float *honvOutput = new float[startImageCols*startImageRows*256];
    // queue.enqueueReadImage(finalOutput[4], CL_TRUE, origin, test_r1, 
    //                        startImageCols*sizeof(float), startImageRows*startImageCols*sizeof(float),
    //                        honvOutput);
    // printArray(honvOutput); 

    queue.enqueueNDRangeKernel(conv2D[5], cl::NullRange, cl::NDRange(startImageCols, startImageRows), 
                               cl::NDRange(1,1));
    queue.enqueueNDRangeKernel(conv2D[6], cl::NullRange, cl::NDRange(startImageCols, startImageRows), 
                               cl::NDRange(1,1));
    queue.enqueueNDRangeKernel(MaxPool[6], cl::NullRange, cl::NDRange(startImageCols/2, startImageRows/2), 
                               cl::NDRange(1,1));
    startImageCols = startImageCols/2;
    startImageRows = startImageRows/2;

    queue.enqueueNDRangeKernel(conv2D[7], cl::NullRange, cl::NDRange(startImageCols, startImageRows), 
                               cl::NDRange(1,1));
    queue.enqueueNDRangeKernel(conv2D[8], cl::NullRange, cl::NDRange(startImageCols, startImageRows), 
                               cl::NDRange(1,1));
    queue.enqueueNDRangeKernel(MaxPool[8], cl::NullRange, cl::NDRange(startImageCols/2, startImageRows/2), 
                               cl::NDRange(1,1));
    startImageCols = startImageCols/2;
    startImageRows = startImageRows/2;

    cl::size_t<3> out_reg;
    
    out_reg[0] = 1;
    out_reg[1] = 1;
    out_reg[2] = 512;
    cl::Buffer inputFC = cl::Buffer(context, CL_MEM_WRITE_ONLY,
     512*sizeof(float));

    float *hinputFClayer = new float[ChannelDepth[i]];
    queue.enqueueReadImage(finalOutput[i], CL_TRUE, origin, out_reg, 
                           1*sizeof(float), 1*1*sizeof(float),
                           hinputFClayer);
    //printArray(hinputFClayer);
    queue.enqueueWriteBuffer(inputFC, CL_TRUE, 0,
         ChannelDepth[i]*sizeof(float), hinputFClayer);

    N = 1024;
    kernel0.setArg(0, 1);
    kernel0.setArg(1, 1024);
    kernel0.setArg(2, ChannelDepth[i]);
    kernel0.setArg(3, inputFC);    
    kernel0.setArg(4, device_layer[0]);
    kernel0.setArg(5, device_layer[1]);
    kernel0.setArg(6, 1);    
    kernel0.setArg(7, outputImage);

    // float *hOutput = new float[1024];
    // // /* Read the output histogram buffer to the host */
    // queue.enqueueReadBuffer(outputImage, CL_TRUE, 0,
    //                           1*1024*sizeof(float), hOutput, NULL, NULL);
    // printArray(hOutput); 

    kernel1.setArg(0, N);
    kernel1.setArg(1, outputImage);
    kernel1.setArg(2, device_layer[2]);
    kernel1.setArg(3, device_layer[3]);
    kernel1.setArg(4, device_layer[4]);
    kernel1.setArg(5, device_layer[5]);
    kernel1.setArg(6, outputImage);

    kernel2.setArg(0, 1);
    kernel2.setArg(1, N);
    kernel2.setArg(2, N);
    kernel2.setArg(3, outputImage);    
    kernel2.setArg(4, device_layer[6]);
    kernel2.setArg(5, device_layer[7]);
    kernel2.setArg(6, 1);    
    kernel2.setArg(7, outputImage);

    kernel3.setArg(0, N);
    kernel3.setArg(1, outputImage);
    kernel3.setArg(2, device_layer[8]);
    kernel3.setArg(3, device_layer[9]);
    kernel3.setArg(4, device_layer[10]);
    kernel3.setArg(5, device_layer[11]);
    kernel3.setArg(6, outputImage);
    
    cl::Buffer outputFC = cl::Buffer(context, CL_MEM_WRITE_ONLY,
     10*sizeof(float));

    kernel4.setArg(0, 1);
    kernel4.setArg(1, 10);
    kernel4.setArg(2, N);
    kernel4.setArg(3, outputImage);    
    kernel4.setArg(4, device_layer[12]);
    kernel4.setArg(5, device_layer[13]);
    kernel4.setArg(6, 1);    
    kernel4.setArg(7, outputFC);

    kernel5.setArg(0, 10);
    kernel5.setArg(1, outputFC);

    cl::NDRange global(M, N);

    cl::NDRange local(1, 1);

    queue.enqueueNDRangeKernel(kernel0, cl::NullRange, global, local, NULL, &kernelEvent[0]);
    kernelEvent[0].wait();
    queue.enqueueNDRangeKernel(kernel1, cl::NullRange, global, local, NULL, &kernelEvent[1]);
    kernelEvent[1].wait();
    queue.enqueueNDRangeKernel(kernel2, cl::NullRange, global, local, NULL, &kernelEvent[2]);
    kernelEvent[2].wait();
    queue.enqueueNDRangeKernel(kernel3, cl::NullRange, global, local, NULL, &kernelEvent[3]);
    kernelEvent[3].wait();
    queue.enqueueNDRangeKernel(kernel4, cl::NullRange, global, local, NULL, &kernelEvent[4]);
    kernelEvent[4].wait();
    queue.enqueueNDRangeKernel(kernel5, cl::NullRange, global, local, NULL, &kernelEvent[5]);
    kernelEvent[5].wait();
  
    end = clock();

    // Calculating total time taken by the program. 
    double time_taken = double(end - start) / double(CLOCKS_PER_SEC); 
    std::cout << "Time taken by program on device : " << std::fixed  
         << time_taken; 
    std::cout << " sec " << std::endl; 
 // }  


    // cl::size_t<3> test_region;
    // test_region[0] = imageCols;
    // test_region[1] = imageRows;
    // test_region[2] = ChannelDepth[0];
    // queue.enqueueReadImage(interImage, CL_TRUE, origin, test_region, 
    //                        imageCols*sizeof(float), imageCols*imageRows*sizeof(float),
    //                        hOArray);
    //printArray(hOArray); 

    cl::size_t<3> test_region1;
    
    int outputCols = int(imageCols/2);
    int outputRows = int(imageRows/2);

    test_region1[0] = 1;
    test_region1[1] = 1;
    test_region1[2] = 512;
    float *hConvOutput = new float[512];
    queue.enqueueReadImage(finalOutput[i], CL_TRUE, origin, test_region1, 
                           1*sizeof(float), 1*1*sizeof(float),
                           hConvOutput);
    //printArray(hConvOutput); 
    float *hOutput = new float[10];
    // /* Read the output histogram buffer to the host */
    queue.enqueueReadBuffer(outputFC, CL_TRUE, 0,
                              1*10*sizeof(float), hOutput, NULL, NULL);
    //printArray(hOutput); 
    // if (o<8500) {
    //     correct += 1;
    // }
    
  }

}

  catch(cl::Error error)
  {
    std::cout << error.what() << "(" << error.err() << ")" << std::endl;
  }

  //std::cout<<"Test accuracy: "<<(correct)<<"%\n";
  free(hInputA);
  //free(hInputB);

  delete hOutputArray;
  return 0;
}
