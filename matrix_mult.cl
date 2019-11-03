#ifdef cl_khr_fp64
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
    #pragma OPENCL EXTENSION cl_amd_fp64 : enable
#else
    #error "Double precision floating point not supported by OpenCL implementation."
#endif
__kernel void log_softmax(int N, 
                          __global float* nodes) {
  
  const int i = get_global_id(0);

  float expsum=0;
  for (int j=0;j<N;j++)
    expsum+=exp(nodes[j]);
  
  nodes[i]=(exp(nodes[i])/expsum);  
}

__kernel void batchNorm(const int M,
                        const __global float* input_layer,
                        const __global float* bn_weight,
                        const __global float* bn_bias,
                        const __global float *running_mean,
                        const __global float *running_var,
                        __global float* output_layer) {
    
    float epsilon = 0.00001f;
    float var = epsilon + running_var[0];
    float sqrt =  pow(var, 2);
    // printf("%f\n", var);
    // printf("%f\n", sqrt);
    float temp = 0.0f;
    for (int i=0; i<M; i++) {
      temp = (input_layer[i] - running_mean[0])/sqrt;
      output_layer[i] = temp*bn_weight[i] + bn_bias[i];
    }
}


// __kernel void hardTanh(const int N, 
//               const __global float* input_layer,
//              __global float* output_layer) {

//   for (int i=0; i<N; i++) {
//     if (input_layer[i] > 1) {
//       output_layer[i] = 1;
//     }
//     else if (input_layer[i] < -1) {
//       output_layer[i] = -1;
//     }
//     else {
//       output_layer[i] = input_layer[i];
//     }
//   }
// }

__kernel void linear(const int M, const int N, const int K,
                    const __global float* input_layer,
                    const __global float* weights,
                    const __global float* bias,
                    const int flag,
                    __global float* output_layer) {

  const int gRow = get_global_id(0); 
  const int gCol = get_global_id(1);
  //printf("%s\n", "sdf");
  //Input: 1xK, Weight Matrix: KxN
  float temp = 0.0f;
  for (int k=0; k<K; k++) {
    temp += input_layer[k * M + gRow] * weights[gCol*K + k];
  }
    temp = temp + bias[gCol*M];
    if (flag == 1) {
      if (temp > 1) {
        output_layer[gCol*M + gRow] = 1;
      }
      else if (temp < -1) {
        output_layer[gCol*M + gRow] = -1;
      }
      else {
        output_layer[gCol*M + gRow] = temp;
      }      
    }
}

__kernel
void median_histogram(
  __read_only image3d_t inputImage,
  __read_only image3d_t inputFilter,
  int inputWidth,
  int filterWidth,
  int inputDepth,
  int outputDepth,
  __write_only image3d_t outputImage
)

{
  /* Store each work-item’s unique row and column */
  int column = get_global_id(0);
  int row = get_global_id(1);

  //printf("%d %d\n", column, row);
  //printf("%s\n", );
  int4 coords_img;
  int4 coords_fil;
  int4 coords_fil_start;
  int4 coords_fil_end;
  int4 coords;

  coords_fil_start.x = 0;
  coords_fil_start.y = 0;

  coords_fil_end.x = 2;
  coords_fil_end.y = 2;

  // Define start and end point in both input image and kernel 
  if (row == 0) {
    coords_img.y = 0;
    coords_fil_start.y = 1;    
  }
  else if (row == inputWidth - 1) {
    coords_img.y = inputWidth - 2;
    coords_fil_end.y = 1;    
  }
  else if (row != 0 && row != inputWidth - 1) {
    coords_img.y = row - 1;
  }

  // define start and end point along row
  if (column == 0) {
    coords_img.x = 0;
    coords_fil_start.x = 1;    
  }
  else if (column == inputWidth - 1) {
    coords_img.x = inputWidth - 2;
    coords_fil_end.x = 1;    
  }
  else if (column != 0 && column != column - 1) {
    coords_img.x = column - 1;
  }

for (int d=0; d<outputDepth; d++) {

  int4 temp_coords; 
  //temp_filter_coords.x = coords_fil.y;
  temp_coords.y = coords_img.y;
  temp_coords.z = d;

  coords_img.z = d;
  coords_fil.z = d;
  float4 temp = {0.0f, 0.0f, 0.0f, 0.0f};

  for (int k=coords_fil_start.y; k<=coords_fil_end.y; k++) {
    coords_fil.y = k;
    temp_coords.x = coords_img.x;
  
    for (int l=coords_fil_start.x; l<=coords_fil_end.x; l++) {
//     if (column == 2 && row == 1) {
//     printf("Value of k: %d %d | %d %d | Image coordinates: %d %d Start Filter coordinates: %d %d End Filter coordinates: %d %d\n", 
//          k, l, column, row, temp_coords.x, temp_coords.y, coords_fil_start.x, coords_fil_start.y, coords_fil_end.x, coords_fil_end.y);
// }
      coords_fil.x = l;
      
      int4 temp_filter_coords;
      temp_filter_coords.x = l + k*3;
      temp_filter_coords.z = d;

      for (int p=0; p<inputDepth; p++) {
      
      temp_filter_coords.y = p;

      float4 pixel = {0.0f, 0.0f, 0.0f, 0.0f};
      float4 filter_val = {0.0f, 0.0f, 0.0f, 0.0f};
      
      //printf("%d\n", coords_img.x);
      // printf("%d\n", coords_img.y);
      pixel = read_imagef(inputImage, temp_coords);
      //printf("%f\n", pixel.x);
      filter_val = read_imagef(inputFilter, temp_filter_coords);
      temp.x += pixel.x*filter_val.x;

      }
      // float4 pixel = {0.0f, 0.0f, 0.0f, 0.0f};
      // float4 filter_val = {0.0f, 0.0f, 0.0f, 0.0f};
      
      // //printf("%d\n", coords_img.x);
      // // printf("%d\n", coords_img.y);
      // pixel = read_imagef(inputImage, temp_coords);
      // //printf("%f\n", pixel.x);
      // filter_val = read_imagef(inputFilter, coords_fil);
      // temp.x += pixel.x*filter_val.x;

      temp_coords.x += 1;
    }

    temp_coords.y += 1;
  }
     // /* Copy the data to the output image */
   coords.x = column;
   coords.y = row;
   coords.z = d;

   if (temp.x < 0) {
    temp.x = 0;
   }
   write_imagef(outputImage, coords, temp.x);
 } 
}



__kernel
void maxpool(
  __read_only image3d_t inputImage,
  int inputWidth,
  int inputDepth,
  __write_only image3d_t outputImage
)

{
  /* Store each work-item’s unique row and column */
  int column = get_global_id(0);
  int row = get_global_id(1);

  //printf("%d %d\n", column, row);
  //printf("%s\n", );
  int4 coords_img;
  int4 coords_fil;
  int4 coords_fil_start;
  int4 coords_fil_end;
  int4 coords;

  coords_img.x = column*2;
  coords_img.y = row*2;

for (int d=0; d<inputDepth; d++) {

  int4 temp_coords; 
  //int4 temp_filter_coords;

  //temp_filter_coords.x = coords_fil.y;
  temp_coords.y = coords_img.y;
  temp_coords.z = d;

  //coords_img.z = d;
  //coords_fil.z = d;
  float4 temp = {0.0f, 0.0f, 0.0f, 0.0f};

  for (int k=0; k < 2; k++) {

  // printf("Value of k: %d | %d %d | Image coordinates: %d %d Start Filter coordinates: %d %d End Filter coordinates: %d %d\n", 
  //        k, column, row, coords_img.x, coords_img.y, coords_fil_start.x, coords_fil_start.y, coords_fil_end.x, coords_fil_end.y);

    coords_fil.y = k;
    temp_coords.x = coords_img.x;
  
    for (int l=0; l < 2; l++) {
//     if (column == 2 && row == 1) {
//     printf("Value of k: %d %d | %d %d | Image coordinates: %d %d Start Filter coordinates: %d %d End Filter coordinates: %d %d\n", 
//          k, l, column, row, temp_coords.x, temp_coords.y, coords_fil_start.x, coords_fil_start.y, coords_fil_end.x, coords_fil_end.y);
// }
      //coords_fil.x = l;
      float4 pixel = {0.0f, 0.0f, 0.0f, 0.0f};
      float4 filter_val = {0.0f, 0.0f, 0.0f, 0.0f};
      
      //printf("%d\n", coords_img.x);
      // printf("%d\n", coords_img.y);
      pixel = read_imagef(inputImage, temp_coords);
      //printf("%f\n", pixel.x);
      //filter_val = read_imagef(inputFilter, coords_fil);
      if (temp.x < pixel.x) {
        temp.x = pixel.x;
      }
      //temp.x += pixel.x*filter_val.x;

      temp_coords.x += 1;
    }

    temp_coords.y += 1;
  }
     // /* Copy the data to the output image */
   coords.x = column;
   coords.y = row;
   coords.z = d;
   write_imagef(outputImage, coords, temp.x);
 } 
}

// // b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
// // fusedconv.bias.copy_( b_conv + b_bn )
// __kernel void batchNorm(const int M,
//                        const __global float* running_mean,
//                        const __global float* running_var,
//                        const __global float* bn_weight,
//                        const __global float* bn_bias,
//                        const __global float* input_layer,
//                        __global float* output_layer) {
    
//     float epsilon = 0.00001f;
//     float sqrt =  sqrt(running_var + epsilon);
//     float temp = 0.0f;
//     for (int i=0; i<M; i++) {
//       temp = (input_layer[i] - running_mean)/sqrt;
//       output_layer = temp*bn_weight[i] + bn_bias[i];
//     }
// }