/*
 * This code is released into the public domain.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <cfloat>

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include <assert.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cublas_v2.h>
#include <cudnn.h>
#include <curand.h>
#include <cuda_profiler_api.h>

using namespace std;

//////////////////////////////////////////////////////////////////////////////
// Error handling
// Adapted from the CUDNN classification code 
// sample: https://developer.nvidia.com/cuDNN

#define FatalError(s) do {                                             \
    std::stringstream _where, _message;                                \
    _where << __FILE__ << ':' << __LINE__;                             \
    _message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__;  \
    std::cerr << _message.str() << "\nAborting...\n";                  \
    cudaDeviceReset();                                                 \
    exit(1);                                                           \
} while(0)

#define checkCUDNN(status) do {                                        \
    std::stringstream _error;                                          \
    if (status != CUDNN_STATUS_SUCCESS) {                              \
      _error << "CUDNN failure: " << cudnnGetErrorString(status);      \
      FatalError(_error.str());                                        \
    }                                                                  \
} while(0)

#define checkCudaErrors(status) do {                                   \
    std::stringstream _error;                                          \
    if (status != 0) {                                                 \
      _error << "Cuda failure: " << status;                            \
			_error << cudaGetErrorString(status) ;                           \
      FatalError(_error.str());                                        \
    }                                                                  \
} while(0)

#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
				    printf("Error at %s:%d\n",__FILE__,__LINE__);\
				    assert(0);}} while(0)


#define ITERATION (1024*256)
#define PAGESIZEINBYTE (1024*4)
#define READSIZEINBYTE (ITERATION*PAGESIZEINBYTE)
#define RANDSIZEINBYTE (ITERATION*sizeof(unsigned int))
#define SHRINK (READSIZEINBYTE/sizeof(unsigned int))

__global__ void initMEM(unsigned int n,unsigned int * pd)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
    pd[i]=0;
}

void allocAdviseMemory(void** ppout,size_t outbuf_size,int gpuid) {
	assert((outbuf_size%sizeof(unsigned int)) == 0);
	checkCudaErrors(cudaSetDevice(gpuid));
  checkCudaErrors (cudaMallocManaged (ppout, outbuf_size));
	checkCudaErrors (cudaMemAdvise (*ppout,outbuf_size ,cudaMemAdviseSetPreferredLocation,gpuid));
	int blockSize = 256;
	int numBlocks = (outbuf_size/sizeof(unsigned int) + blockSize - 1) / blockSize;
	initMEM<<<numBlocks, blockSize>>>(outbuf_size/sizeof(unsigned int),(unsigned int*)(*ppout));
}

void syncAllGPU(int num_gpus) {
  for (int gpuid = 0; gpuid < num_gpus; gpuid++)
  {
    checkCudaErrors (cudaSetDevice (gpuid));
    checkCudaErrors (cudaDeviceSynchronize ());
  }
}

/*__global__ void readlat(int src, void * pout)
{
	
}
*/
__global__ void remainSSY(unsigned int n,unsigned int * pd,unsigned int remssy)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
    pd[i]=pd[i]%remssy;
}


void genrandVector(unsigned int ** ppranddev,int src,bool genrand) {
	allocAdviseMemory((void**)ppranddev,RANDSIZEINBYTE,src);

    if(genrand) {
	//rangen
	curandGenerator_t gen;
	CURAND_CALL(curandCreateGenerator(&gen,CURAND_RNG_PSEUDO_DEFAULT));
	//set seed
	CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));
	CURAND_CALL(curandGenerate(gen, *ppranddev, ITERATION));
	int blockSize = 256;
	int numBlocks = (ITERATION + blockSize - 1) / blockSize;
	remainSSY<<<numBlocks, blockSize>>>(ITERATION,*ppranddev,SHRINK);
    CURAND_CALL(curandDestroyGenerator(gen));
    }

}

unsigned int rep;
void  checkRandHost(unsigned int *pranddev) {
	unsigned int * prandhost;
	prandhost = (unsigned int *) malloc(RANDSIZEINBYTE);
	//copy back to host
	checkCudaErrors(cudaMemcpy(prandhost, pranddev, RANDSIZEINBYTE, cudaMemcpyDeviceToHost));
	for(int i=0;i<ITERATION;i++) 
			cout<<i<<" "<<prandhost[i]<<endl;
	free(prandhost);
}

__global__ void access(unsigned int rep,unsigned int * pranddev,unsigned int * pout) {
	for(int j=0;j<rep;j++)
	for(int i=0;i<ITERATION;i++) {
		unsigned int idx=pranddev[i];
		unsigned int xx=pout[idx];
	}
}

long testlat(int src, int dest,int num_gpus,bool genrand) {
    if(genrand) 
    	cout<<"random address from "<<src<<" to "<<dest<<endl;
    else
        cout<<"zero address from "<<src<<" to "<<dest<<endl;
	//allocate memory on dest
	unsigned int *pout;
	allocAdviseMemory((void **)&pout,READSIZEINBYTE,dest);

	//generating randome on src
	checkCudaErrors(cudaSetDevice(src));
	unsigned int * pranddev;
	genrandVector(&pranddev,src,genrand);
	
	//check the geenrated random vector on host
//	if(src==0 && dest==0) 
//		checkRandHost(pranddev);
	syncAllGPU(num_gpus);
	checkCudaErrors(cudaSetDevice(src));
  auto t1 = chrono::high_resolution_clock::now ();
	unsigned int total=rep*ITERATION;
	access<<<1,1>>>(rep,pranddev,pout);
	syncAllGPU(num_gpus);
  auto t2 = chrono::high_resolution_clock::now ();

	auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = t2-t1;
	std::chrono::duration<double> diff1 = diff/total;
  std::cout << "accessing "<< total  <<" need " << diff.count() << "s each need " << diff1.count()*1000*1000*1000<<"ns"<<endl;
	checkCudaErrors(cudaFree(pranddev));
	checkCudaErrors(cudaFree(pout));
	return 0;
}

///////////////////////////////////////////////////////////////////////////////////////////
// Main function
//#define WIDTH 280
int
main (int argc, char **argv)
{


	if(argc!=2) {
			cout<<"Usage : nvlinklatency.exe <repreat time>"<<endl;
			return 0;
	}
	rep=atoi(argv[1]);
	cout<<"repeat "<<rep<<" times"<<endl;

  // Choose GPU
  int num_gpus;
  checkCudaErrors (cudaGetDeviceCount (&num_gpus));
	cout<<"num_gpus "<<num_gpus<<endl;

	for(int i=0;i<num_gpus;i++) {
					for(int j=0;j<num_gpus;j++) {
//									if(i==0 && j==1) cudaProfilerStart();
									testlat(i,j,num_gpus,false);
									testlat(i,j,num_gpus,true);
//									if(i==0 && j==1) cudaProfilerStop();
					}
	}
}
