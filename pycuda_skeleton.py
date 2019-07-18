
# encoding: utf-8
#
# Created Sri Harsha  on 2019-02-1.
#
#

import pycuda.driver as cuda
import pycuda.compiler
import pycuda.autoinit
import numpy as np
from math import pi,cos,sin
from pycuda.compiler import SourceModule
from pycuda.autoinit import context
from skimage import data
from skimage import draw
import math
import PIL.ImageOps
import time
from skimage.morphology import skeletonize, medial_axis
import cv2

pix_equality = SourceModule("""

__global__ void pixel_equality(unsigned char* g_in_1, unsigned char* g_in_2, unsigned char* g_out, int g_width, int g_height) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int g_size = g_width * g_height;

    while (tid < g_size) {
        int g_row = (tid / g_width);
        int g_col = (tid % g_width);

        unsigned char value_1 = g_in_1[g_row * g_width + g_col];
        unsigned char value_2 = g_in_2[g_row * g_width + g_col];

        unsigned char write_data = (value_1 == value_2);
        g_out[g_row * g_width + g_col] = write_data;

        tid += (gridDim.x * blockDim.x);
    }
}

""")

reduction = SourceModule("""

__device__ unsigned char block_and_reduce(unsigned char* s_data) {
    for (int s = (blockDim.x / 2); s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_data[threadIdx.x] &= s_data[threadIdx.x + s];
        }
        __syncthreads();
    }

    return s_data[0];
}

__global__ void and_reduction(unsigned char* g_data, int g_size) {

    __shared__ unsigned char s_data[2000];

    int blockReductionIndex = blockIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    //printf("thread = %d , block = %d, dim = %d, grid = %d\\n",threadIdx.x, blockIdx.x, blockDim.x, gridDim.x);

    int num_iterations_needed = ceil(g_size / ((double) (blockDim.x * gridDim.x)));
    //printf("%d \\n", num_iterations_needed);
    for (int iteration = 0; iteration < num_iterations_needed ; iteration++) {
        //printf("gdata = %d  i = %d gsize = %d\\n",g_data[i],i, g_size);
        s_data[threadIdx.x] = (i < g_size) ? g_data[i] : 1;
        __syncthreads();

        // do reduction in shared memory
        block_and_reduce(s_data);

        // write result for this block to global memory
        if (threadIdx.x == 0) {
            g_data[blockReductionIndex] = s_data[0];
            //printf("%d\\n",g_data[blockReductionIndex]);
        }

        blockReductionIndex += gridDim.x;
        i += (gridDim.x * blockDim.x);

    }

    //printf("%d , threadIdx =  %d\\n", g_data[blockReductionIndex], threadIdx.x);

}

""")


mod = SourceModule("""
#include <stdint.h>

#define BINARY_BLACK (1)
#define BINARY_WHITE (0)


__device__ unsigned char is_outside_image(int g_row, int g_col, int g_width, int g_height) {
    return (g_row < 0) | (g_row > (g_height - 1)) | (g_col < 0) | (g_col > (g_width - 1));
}

__device__ unsigned char border_global_mem_read(unsigned char* g_data, int g_row, int g_col, int g_width, int g_height) {
    return is_outside_image(g_row, g_col, g_width, g_height) ? BINARY_WHITE : g_data[g_row * g_width + g_col];
}


__device__ unsigned char P2_f(unsigned char* g_data, int g_row, int g_col, int g_width, int g_height) {
    return border_global_mem_read(g_data, g_row - 1, g_col, g_width, g_height);
}

__device__ unsigned char P3_f(unsigned char* g_data, int g_row, int g_col, int g_width, int g_height) {
    return border_global_mem_read(g_data, g_row - 1, g_col - 1, g_width, g_height);
}

__device__ unsigned char P4_f(unsigned char* g_data, int g_row, int g_col, int g_width, int g_height) {
    return border_global_mem_read(g_data, g_row, g_col - 1, g_width, g_height);
}

__device__ unsigned char P5_f(unsigned char* g_data, int g_row, int g_col, int g_width, int g_height) {
    return border_global_mem_read(g_data, g_row + 1, g_col - 1, g_width, g_height);
}

__device__ unsigned char P6_f(unsigned char* g_data, int g_row, int g_col, int g_width, int g_height) {
    return border_global_mem_read(g_data, g_row + 1, g_col, g_width, g_height);
}

__device__ unsigned char P7_f(unsigned char* g_data, int g_row, int g_col, int g_width, int g_height) {
    return border_global_mem_read(g_data, g_row + 1, g_col + 1, g_width, g_height);
}

__device__ unsigned char P8_f(unsigned char* g_data, int g_row, int g_col, int g_width, int g_height) {
    return border_global_mem_read(g_data, g_row, g_col + 1, g_width, g_height);
}

__device__ unsigned char P9_f(unsigned char* g_data, int g_row, int g_col, int g_width, int g_height) {
    return border_global_mem_read(g_data, g_row - 1, g_col + 1, g_width, g_height);
}

__device__ unsigned char wb_transitions_around(unsigned char* g_data, int g_row, int g_col, int g_width, int g_height) {
    int count = 0;

    count += ((P2_f(g_data, g_row, g_col, g_width, g_height) == BINARY_WHITE) & (P3_f(g_data, g_row, g_col, g_width, g_height) == BINARY_BLACK));
    count += ((P3_f(g_data, g_row, g_col, g_width, g_height) == BINARY_WHITE) & (P4_f(g_data, g_row, g_col, g_width, g_height) == BINARY_BLACK));
    count += ((P4_f(g_data, g_row, g_col, g_width, g_height) == BINARY_WHITE) & (P5_f(g_data, g_row, g_col, g_width, g_height) == BINARY_BLACK));
    count += ((P5_f(g_data, g_row, g_col, g_width, g_height) == BINARY_WHITE) & (P6_f(g_data, g_row, g_col, g_width, g_height) == BINARY_BLACK));
    count += ((P6_f(g_data, g_row, g_col, g_width, g_height) == BINARY_WHITE) & (P7_f(g_data, g_row, g_col, g_width, g_height) == BINARY_BLACK));
    count += ((P7_f(g_data, g_row, g_col, g_width, g_height) == BINARY_WHITE) & (P8_f(g_data, g_row, g_col, g_width, g_height) == BINARY_BLACK));
    count += ((P8_f(g_data, g_row, g_col, g_width, g_height) == BINARY_WHITE) & (P9_f(g_data, g_row, g_col, g_width, g_height) == BINARY_BLACK));
    count += ((P9_f(g_data, g_row, g_col, g_width, g_height) == BINARY_WHITE) & (P2_f(g_data, g_row, g_col, g_width, g_height) == BINARY_BLACK));

    return count;
}

__device__ unsigned char black_neighbors_around(unsigned char* g_data, int g_row, int g_col, int g_width, int g_height) {
    int count = 0;

    count += (P2_f(g_data, g_row, g_col, g_width, g_height) == BINARY_BLACK);
    count += (P3_f(g_data, g_row, g_col, g_width, g_height) == BINARY_BLACK);
    count += (P4_f(g_data, g_row, g_col, g_width, g_height) == BINARY_BLACK);
    count += (P5_f(g_data, g_row, g_col, g_width, g_height) == BINARY_BLACK);
    count += (P6_f(g_data, g_row, g_col, g_width, g_height) == BINARY_BLACK);
    count += (P7_f(g_data, g_row, g_col, g_width, g_height) == BINARY_BLACK);
    count += (P8_f(g_data, g_row, g_col, g_width, g_height) == BINARY_BLACK);
    count += (P9_f(g_data, g_row, g_col, g_width, g_height) == BINARY_BLACK);

    return count;
}


__global__ void skeletonize_pass(int g_width, int g_height, unsigned char *g_src, unsigned char *g_dst) {


    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int g_size = g_width * g_height;


    //printf("tid = %d, bdim = %d, gridDim  = %d",threadIdx.x, blockDim.x , gridDim.x );
    while (tid < g_size) {

        int g_row = (tid / g_width);
        int g_col = (tid % g_width);

        unsigned char NZ = black_neighbors_around(g_src, g_row, g_col, g_width, g_height);
        unsigned char TR_P1 = wb_transitions_around(g_src, g_row, g_col, g_width, g_height);
        unsigned char TR_P2 = wb_transitions_around(g_src, g_row - 1, g_col, g_width, g_height);
        unsigned char TR_P4 = wb_transitions_around(g_src, g_row, g_col - 1, g_width, g_height);
        unsigned char P2 = P2_f(g_src, g_row, g_col, g_width, g_height);
        unsigned char P4 = P4_f(g_src, g_row, g_col, g_width, g_height);
        unsigned char P6 = P6_f(g_src, g_row, g_col, g_width, g_height);
        unsigned char P8 = P8_f(g_src, g_row, g_col, g_width, g_height);

        unsigned char thinning_cond_1 = ((2 <= NZ) & (NZ <= 6));
        unsigned char thinning_cond_2 = (TR_P1 == 1);
        unsigned char thinning_cond_3 = (((P2 & P4 & P8) == 0) | (TR_P2 != 1));
        unsigned char thinning_cond_4 = (((P2 & P4 & P6) == 0) | (TR_P4 != 1));
        unsigned char thinning_cond_ok = thinning_cond_1 & thinning_cond_2 & thinning_cond_3 & thinning_cond_4;

        unsigned char g_dst_next = (thinning_cond_ok * BINARY_WHITE) + ((1 - thinning_cond_ok) * g_src[g_row * g_width + g_col]);

        //printf("d_sdt_next = %d",g_src[g_row * g_width + g_col]);
        g_dst[g_row * g_width + g_col] = g_dst_next;

        tid += (gridDim.x * blockDim.x);
        //printf("Iterations, tid = %d", threadIdx.x);
        //printf("tid = %d",tid);
    }

}
""")


def and_reduction(sourceImage_gpu, destImage_gpu, equ_data_gpu, g_width,g_height):
    grids = 32
    pixel_equality_gpu = pix_equality.get_function("pixel_equality")
    and_reduction_gpu = reduction.get_function("and_reduction")

    pixel_equality_gpu(sourceImage_gpu, destImage_gpu, equ_data_gpu, np.int32(g_width), np.int32(g_height),block=(512,1,1), grid = (grids,1))
    cuda.Context.synchronize()

    g_size = g_width * g_height

    while(True):
        and_reduction_gpu(equ_data_gpu, np.int32(g_size), block=(512,1,1), grid = (grids,1))
        cuda.Context.synchronize()
        g_size = math.ceil(g_size/256)

        if g_size==1.0:
            break

       # value = math.ceil(g_size/256)

def swap_images(src, dst):
    temp = dst
    dst = src
    src = temp


def remove_salt_noise(cnt):
    rej_idx = []
    for e, c in enumerate(cnt):
        if cv2.contourArea(c) < 300:
            rej_idx.append(e)

    rej_idx.sort()

    for i in range(len(rej_idx)):
        cnt.pop(rej_idx[i])
        rej_idx = [x-1 for x in rej_idx]
    return cnt

def skeleton_image( src_img , interpolation = "linear"):


    src_img = src_img/255.
    src_img = src_img.astype(dtype = "uint8")

    #call kernels
    equ_size = np.asarray(src_img.shape[1] * src_img.shape[0])

    cuda_skeleton = mod.get_function("skeletonize_pass")

    src_malloc_size = np.random.randn(src_img.size * src_img.dtype.itemsize)


    mem_stat = time.time()
    sourceImage_gpu = cuda.mem_alloc(src_img.nbytes)
    destImage_gpu = cuda.mem_alloc(src_img.nbytes)
    equ_data_gpu = cuda.mem_alloc(src_img.nbytes)

    #print(sourceImage_gpu, destImage_gpu, equ_data_gpu)

    cuda.memcpy_htod(sourceImage_gpu, src_img)
    mem_end = time.time()

    #print("time for memory transfer "+ str(mem_end - mem_stat))
    identical_bits = np.empty_like(src_img)
    dst_img = np.empty_like(src_img)

    i = 0


    while(True):

        cuda_skeleton(np.int32(src_img.shape[1]), np.int32(src_img.shape[0]), sourceImage_gpu, destImage_gpu,block=(512,1,1), grid = (32,1))

        cuda.Context.synchronize()
        and_reduction(sourceImage_gpu, destImage_gpu, equ_data_gpu, np.int(src_img.shape[1]), np.int32(src_img.shape[0]))

        cuda.memcpy_dtoh(identical_bits, equ_data_gpu)

        temp = destImage_gpu
        destImage_gpu = sourceImage_gpu
        sourceImage_gpu = temp

        i = i+1
        if (int(identical_bits[0][0])):
            break

    cuda.memcpy_dtoh(dst_img,  destImage_gpu)

    dst_img = (255 * dst_img).astype(dtype = "uint8")


    return dst_img


if __name__ == '__main__':
    from PIL import Image
    import sys

    def main( ):
        if len(sys.argv) != 2:
            print "You should really read the source...\n\nUsage: rotate.py <Imagename>\n"
            sys.exit(-1)

        # Open, convert to grayscale, convert to numpy array
        img = Image.open(sys.argv[1]).convert("L")
        i = np.fromstring(img.tobytes(),dtype="uint8").reshape(img.size[1],img.size[0])

        cv_img = cv2.imread(sys.argv[1])
        cv_img= cv_img[:,:,0]
        cv_img= np.uint8(cv_img>100)

        raw_contours, yyy = cv2.findContours(cv_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        contours_no_sn = remove_salt_noise(raw_contours)

        start_1 = time.time()
        final_skel=  np.zeros_like(cv_img)
        for cnt in contours_no_sn:
            skel_canvas = np.zeros_like(cv_img)

            gamma = 0.017 * cv2.arcLength(cnt , True)
            approx_skeleton = cv2.approxPolyDP(cnt , gamma, True)

            cv2.drawContours(skel_canvas, [approx_skeleton], -1, 255, cv2.FILLED)
            skel = skeleton_image(skel_canvas)
            final_skel = final_skel + skel

        end_1 = time.time()
        print("Time excecution for pycuda skeleton = " + str(end_1-start_1))

        # Rotate & convert back to PIL Image

        cnt_canvas = np.zeros_like(cv_img)
        for cnt in contours_no_sn:
            gamma = 0.05 * cv2.arcLength(cnt , True)
            approx_skeleton1 = cv2.approxPolyDP(cnt , gamma, True)
            cv2.drawContours(cnt_canvas, [approx_skeleton1], -1, 255, cv2.FILLED)

        start_3 = time.time()
        skel1 = skeleton_image(cnt_canvas)
        end_3 = time.time()
        print("Time excecution for pycuda entire image = " + str(end_3 - start_3))


        _skel = _skel *255
        rotimg = Image.fromarray(skel1 ,mode="L")

        rotimg.save("rotated.png")



    main()
