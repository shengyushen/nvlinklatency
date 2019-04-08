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

bool advise;

void allocAdviseMemory(void** ppout,size_t outbuf_size,int gpuid) {
	checkCudaErrors(cudaSetDevice(gpuid));
  checkCudaErrors (cudaMallocManaged (ppout, outbuf_size));
	if(advise)
	  checkCudaErrors (cudaMemAdvise (*ppout,outbuf_size ,cudaMemAdviseSetPreferredLocation,gpuid));
}

//seems to be a class like struct
class baseModule{
	public : 
	float alpha;
	float beta;

	string name;
  int in_channels, out_channels ;
  int in_width, in_height, out_width, out_height;

	float * pin; // this is pass in from outside
	float * pinDiff;
	long inbuf_size;
	float * pout; // pout should be alloc by child class
	float * poutDiff;
	long outbuf_size;

	int minibatch;

	int gpuid;

  cudnnHandle_t cudnnHandle;
  cublasHandle_t cublasHandle;

	size_t m_workspaceSizeByte;
	void * p_workspace;

	bool bNeedSyncInTensor;


	void print() {
					cout<<"name "<<name<<endl;
					cout<<"in_channels "<<in_channels<<endl;
					cout<<"out_channels "<<out_channels<<endl;
					cout<<"in_width "<<in_width<<endl;
					cout<<"in_height "<<in_height<<endl;
					cout<<"out_width "<<out_width<<endl;
					cout<<"out_height "<<out_height<<endl;
					cout<<"inbuf_size "<<inbuf_size<<endl;
					cout<<"outbuf_size "<<outbuf_size<<endl;
					cout<<"minibatch "<<minibatch<<endl;
					cout<<"gpuid "<<gpuid<<endl;
	}
	baseModule(
									float alpha_,
									float beta_,
									string name_,
  								cudnnHandle_t cudnnHandle_,
								  cublasHandle_t cublasHandle_,
									int gpuid_,
									int minibatch_,
									int in_c_,
									int out_c_,
									int in_h_,
									int in_w_,
									int out_h_,
									int out_w_,
									float * pin_
									) 
	{
		alpha=alpha_;
		beta=beta_;
		name=name_;
		cudnnHandle = cudnnHandle_;
		cublasHandle = cublasHandle_;
		gpuid = gpuid_;
		minibatch = minibatch_;
		in_channels = in_c_;
		out_channels = out_c_;
		in_width = in_w_;
		in_height = in_h_;
		out_width = out_w_;
		out_height = out_h_;
		pin = pin_;
		inbuf_size = minibatch_*in_c_*in_w_*in_h_;
		m_workspaceSizeByte=0;
		p_workspace=NULL;
		bNeedSyncInTensor=true;

		assert(gpuid>=0);
		assert(minibatch >0);
		assert(in_channels >0);
		assert(out_channels >0);
		assert(in_width >0);
		assert(in_height >0);
		assert(out_width >0);
		assert(out_height >0);
		assert(pin );
	}

	void allocPout(int minibatch_,int out_c_,int out_h_,int out_w_) {
//		minibatch = minibatch_;
//		out_channels = out_c_;
//		out_width = out_w_;
//		out_height = out_h_;
					assert(minibatch_==minibatch);
					assert(out_channels==out_c_);
					assert(out_width==out_w_);
					assert(out_height==out_h_);
		outbuf_size = minibatch_*out_c_*out_w_*out_h_;
		allocAdviseMemory((void**)&pout,sizeof (float) *outbuf_size,gpuid);
	}

	void allocPinDiff() {
		allocAdviseMemory((void**)&pinDiff,sizeof (float) *inbuf_size,gpuid);
	}

	virtual void fw1step() {};
	virtual void bw1step() {};

	~baseModule  () {
			checkCudaErrors(cudaSetDevice(gpuid));
			cudaFree(pout);
			cudaFree(pinDiff);
	}
	size_t getOutputFloatNumber() {
					return outbuf_size;
	}
	size_t getInputFloatNumber() {
					return inbuf_size;
	}

};

class MaxPoolLayer: public baseModule {
	public :
	int size, stride;
	cudnnTensorDescriptor_t srcTensorDesc;
  cudnnPoolingDescriptor_t poolDesc;
	cudnnTensorDescriptor_t  dstTensorDesc; //this out already have pout in baseModule
	MaxPoolLayer(
			float alpha_,
			float beta_,
			string name_,
			cudnnHandle_t cudnnHandle_,
			cublasHandle_t cublasHandle_,
			int gpuid_,
			int minibatch_,
			int in_channels_, 
			int in_h_, int in_w_, 
			int kernel_size_, int stride_, //it seems pooling just remain the same number of channel as input
			int paddingH_, int paddingW_,float * pin_) 
					: baseModule(
									alpha_,
									beta_,
									name_,
									cudnnHandle_,
									cublasHandle_,
									gpuid_,
									minibatch_,
									in_channels_,
									in_channels_, // it seems the output channle is the same as input
									in_h_,
									in_w_,
									(in_h_+paddingH_*2-kernel_size_)/stride_+1,
									(in_w_+paddingW_*2-kernel_size_)/stride_+1,
									pin_
								)
	{
				printf("MaxPoolLayer gpuid %d minibatch %d in_channels_ %d in_h_ %d in_w_ %d kernel_size_ %d stride_ %d paddingH_ %d paddingW_ %d\n",
						                      gpuid_ ,  minibatch_ ,  in_channels_ ,  in_h_ ,  in_w_ ,   kernel_size_ ,  stride_ ,  paddingH_ ,  paddingW_ );
		size= kernel_size_;
		stride = stride_;
		assert(size > 0);
		assert(stride > 0);
//		assert((in_w_+paddingW_*2-kernel_size_)%stride_ == 0);
//		assert((in_h_+paddingH_*2-kernel_size_)%stride_ == 0);

		allocPout(minibatch,in_channels_,(in_h_+paddingH_*2-kernel_size_)/stride_+1,(in_w_+paddingW_*2-kernel_size_)/stride_+1);
		allocPinDiff();
		//all layer follow this pattern
		// 1 set the source tensor
		// 2 set the operator tensor
		// 3 set the dest tensor

		// 1 set the source tensor
    checkCUDNN (cudnnCreateTensorDescriptor (&srcTensorDesc));
    checkCUDNN(cudnnSetTensor4dDescriptor(srcTensorDesc,
                                              CUDNN_TENSOR_NCHW,
                                              CUDNN_DATA_FLOAT,
																							minibatch, in_channels, in_height, in_width
                                              ));

		// 2 set the operator tensor
    checkCUDNN(cudnnCreatePoolingDescriptor(&poolDesc));
    checkCUDNN(cudnnSetPooling2dDescriptor(poolDesc,
                                               CUDNN_POOLING_MAX,
                                               CUDNN_PROPAGATE_NAN,
                                               size, size,
                                               paddingH_,paddingW_ ,
                                               stride, stride));

		// 3 set the dest tensor
    checkCUDNN (cudnnCreateTensorDescriptor (&dstTensorDesc));
    checkCUDNN (cudnnSetTensor4dDescriptor (dstTensorDesc,
					    CUDNN_TENSOR_NCHW,
					    CUDNN_DATA_FLOAT, minibatch_, in_channels_, (in_h_+paddingH_*2-kernel_size_+1)/stride_, (in_w_+paddingW_*2-kernel_size_+1)/stride_));
		


		// should not be sync
		bNeedSyncInTensor = false;
		m_workspaceSizeByte =0;

	}
	void fw1step () {
				//pooling layer dont need workspace
				//assert(p_workspace!=NULL);
				//assert(m_workspaceSizeByte!=0);
        checkCudaErrors(cudaSetDevice(gpuid));
        checkCUDNN(cudnnPoolingForward(cudnnHandle, poolDesc, &alpha, srcTensorDesc,
                                       pin, &beta, dstTensorDesc,pout));

	}
	void bw1step() {
					// pooling layer have no weight
    checkCudaErrors(cudaSetDevice(gpuid));
		checkCUDNN(cudnnPoolingBackward(cudnnHandle, poolDesc, &alpha, 
                                        dstTensorDesc, pout,dstTensorDesc, poutDiff,
                                        srcTensorDesc, pin, &beta, srcTensorDesc, pinDiff));
	}
	~MaxPoolLayer() {
		checkCUDNN (cudnnDestroyTensorDescriptor(srcTensorDesc));
		checkCUDNN (cudnnDestroyPoolingDescriptor(poolDesc));
    checkCUDNN (cudnnDestroyTensorDescriptor (dstTensorDesc));
	}
};

class ConvBiasLayer: public baseModule
{
		public :
		int kernel_size,stride;

		cudnnTensorDescriptor_t biasTensor;
		float * pconvbias;
		float * pconvbiasGradient;

		cudnnTensorDescriptor_t srcTensorDesc;

		cudnnFilterDescriptor_t filterDesc;
		float * pconvWeigth;
		float * pconvWeigthGradient;

		cudnnConvolutionDescriptor_t convDesc;

		cudnnTensorDescriptor_t  dstTensorDesc; //this out already have pout in baseModule

		cudnnConvolutionFwdAlgo_t fwalgo;
		cudnnConvolutionBwdFilterAlgo_t bwfalgo;
		cudnnConvolutionBwdDataAlgo_t bwdalgo;

    ConvBiasLayer (
				float alpha_,
				float beta_,
				string name_,
				cudnnHandle_t cudnnHandle_,
			  cublasHandle_t cublasHandle_,
				int gpuid_,
				int minibatch_,
				int in_channels_, 
				int in_h_, int in_w_, 
				int numFilter_, int kernel_size_, int stride_, 
				int paddingH_, int paddingW_,
				float * pin_)  // pin pass from outside
						: baseModule(
									alpha_,
									beta_,
									name_,
									cudnnHandle_,
									cublasHandle_,
									gpuid_,
									minibatch_,
									in_channels_,
									numFilter_,
									in_h_,
									in_w_,
									(in_h_+paddingH_*2-kernel_size_)/stride_+1,
									(in_w_+paddingW_*2-kernel_size_)/stride_+1,
									pin_
								)
		{
				printf("ConvBiasLayer gpuid %d minibatch %d in_channels_ %d in_h_ %d in_w_ %d numFilter_ %d kernel_size_ %d stride_ %d paddingH_ %d paddingW_ %d out_height %d out_width %d\n",
						                      gpuid ,  minibatch ,  in_channels_ ,  in_h_ ,  in_w_ ,  numFilter_ ,  kernel_size_ ,  stride_ ,  paddingH_ ,  paddingW_ , (in_h_+paddingH_*2-kernel_size_)/stride_+1, (in_w_+paddingW_*2-kernel_size_)/stride_+1);
				//assert((in_w_+paddingW_*2-kernel_size_)%stride_ == 0);
				//assert((in_h_+paddingH_*2-kernel_size_)%stride_ == 0);

				kernel_size = kernel_size_;
				assert(kernel_size<16); //this is not strict, just to prevent unreasonable large kernel
				stride=stride_;
				assert(stride < 16);//also not strict

				//bias descriptor
				checkCUDNN (cudnnCreateTensorDescriptor (&biasTensor));
    		checkCUDNN (cudnnSetTensor4dDescriptor (biasTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, out_channels, 1, 1));
    		allocAdviseMemory((void**)&pconvbias,         sizeof(float) * out_channels,gpuid );
    		allocAdviseMemory((void**)&pconvbiasGradient, sizeof(float) * out_channels,gpuid );

				//set the source tensor
    		checkCUDNN (cudnnCreateTensorDescriptor (&srcTensorDesc));
				//this may fail in 800 pixel because the cudnnSetTensor4dDescriptor require the tensor smaller than 2GB, so I may need 700
    		checkCUDNN (cudnnSetTensor4dDescriptor (srcTensorDesc,
					    CUDNN_TENSOR_NCHW,
					    CUDNN_DATA_FLOAT, minibatch, in_channels, in_height, in_width));

				//set the filter desc
    		checkCUDNN (cudnnCreateFilterDescriptor (&filterDesc));
    		checkCUDNN (cudnnSetFilter4dDescriptor (filterDesc,
					    CUDNN_DATA_FLOAT,
					    CUDNN_TENSOR_NCHW,
					    out_channels,
					    in_channels,
					    kernel_size,
					    kernel_size));
				allocAdviseMemory((void**)&pconvWeigth,        sizeof(float)*in_channels_*kernel_size_*kernel_size_*numFilter_,gpuid);
				allocAdviseMemory((void**)&pconvWeigthGradient,sizeof(float)*in_channels_*kernel_size_*kernel_size_*numFilter_,gpuid);

    		checkCUDNN (cudnnCreateConvolutionDescriptor (&convDesc));
    		checkCUDNN (cudnnSetConvolution2dDescriptor (convDesc,
						 paddingH_, paddingW_,
						 stride, stride,
						 1, 1, // we currently dont support dilation
						 CUDNN_CROSS_CORRELATION,
						 CUDNN_DATA_FLOAT));

				int n,c,h,w;
    		checkCUDNN (cudnnGetConvolution2dForwardOutputDim (convDesc,
						       srcTensorDesc,
						       filterDesc,
						       &n, &c, &h, &w));
				assert(n==minibatch);
				assert(c=out_channels);
				assert(h==out_height);
				assert(w==out_width);
				cout<<"minibatch "<<minibatch<<endl;
				cout<<"out_channels "<<out_channels<<endl;
				cout<<"out_height "<<out_height<<endl;
				cout<<"out_width "<<out_width<<endl;

				allocPout(n,c,h,w);
				allocPinDiff();

    		checkCUDNN (cudnnCreateTensorDescriptor (&dstTensorDesc));
    		checkCUDNN (cudnnSetTensor4dDescriptor (dstTensorDesc,
					    CUDNN_TENSOR_NCHW,
					    CUDNN_DATA_FLOAT, n, c, h, w));
		    checkCUDNN (cudnnGetConvolutionForwardAlgorithm (cudnnHandle,
						     srcTensorDesc,
						     filterDesc,
						     convDesc,
						     dstTensorDesc,
						     CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
						     0, &fwalgo));

		    checkCUDNN (cudnnGetConvolutionForwardWorkspaceSize (cudnnHandle,
							 srcTensorDesc,
							 filterDesc,
							 convDesc,
							 dstTensorDesc,
							 fwalgo, &m_workspaceSizeByte));
				//assert(m_workspaceSizeByte >0);

				bNeedSyncInTensor = kernel_size_ >1; // bigger than 1 need to consider data from neighbour

				//handling backward algo
        // If backprop filter algorithm was requested
				size_t tmpsize=0;
        checkCUDNN(cudnnGetConvolutionBackwardFilterAlgorithm(
            cudnnHandle, srcTensorDesc, dstTensorDesc, convDesc, filterDesc,
            CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0, &bwfalgo));

        checkCUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(
            cudnnHandle, srcTensorDesc, dstTensorDesc, convDesc, filterDesc, 
            bwfalgo, &tmpsize));

        m_workspaceSizeByte = std::max(m_workspaceSizeByte, tmpsize);

        // If backprop data algorithm was requested
        checkCUDNN(cudnnGetConvolutionBackwardDataAlgorithm(
            cudnnHandle, filterDesc, dstTensorDesc, convDesc, srcTensorDesc,
            CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0, &bwdalgo));

        checkCUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(
            cudnnHandle, filterDesc, dstTensorDesc, convDesc, srcTensorDesc, 
            bwdalgo, &tmpsize));

        m_workspaceSizeByte = std::max(m_workspaceSizeByte, tmpsize);
        
		}
	void fw1step () {
					//this may not be neccessary
//				if(p_workspace==NULL) 
//					cout<<"WARNING : gpuid "<< gpuid<<" p_workspace is empty"<<endl;
//				if(m_workspaceSizeByte==0) 
//					cout<<"WARNING : gpuid "<< gpuid<<" m_workspaceSizeByte is zero"<<endl;
        float alpha = 1.0f, beta = 0.0f;
        checkCudaErrors(cudaSetDevice(gpuid));
        checkCUDNN(cudnnConvolutionForward(cudnnHandle, 
																					&alpha, 
																					srcTensorDesc, pin, 
																					filterDesc, pconvWeigth, 
																					convDesc, 
                                           fwalgo, p_workspace, m_workspaceSizeByte, &beta,
                                           dstTensorDesc, pout));
        checkCUDNN(cudnnAddTensor(cudnnHandle, &alpha, biasTensor,
                                  pconvbias, &alpha, dstTensorDesc, pout));

	}

	void bw1step () {
        checkCUDNN(cudnnConvolutionBackwardBias(cudnnHandle, &alpha, dstTensorDesc,
                                                poutDiff, &beta, biasTensor, pconvbias));

        
        checkCUDNN(cudnnConvolutionBackwardFilter(cudnnHandle, &alpha, srcTensorDesc,
                                                  pin, dstTensorDesc, poutDiff, convDesc,
                                                  bwfalgo, p_workspace, m_workspaceSizeByte,
                                                  &beta, filterDesc, pconvWeigthGradient));
    
        checkCUDNN(cudnnConvolutionBackwardData(cudnnHandle, &alpha, filterDesc,
                                                pconvWeigth, dstTensorDesc, poutDiff, convDesc, 
                                                bwdalgo, p_workspace, m_workspaceSizeByte,
                                                &beta, srcTensorDesc, pinDiff));
			
	}
	~ConvBiasLayer() {
		checkCUDNN (cudnnDestroyTensorDescriptor(biasTensor));
		checkCUDNN (cudnnDestroyTensorDescriptor(srcTensorDesc));
		checkCUDNN (cudnnDestroyFilterDescriptor(filterDesc));
		checkCudaErrors(cudaFree(pconvWeigth));
		checkCudaErrors(cudaFree(pconvWeigthGradient));
		checkCudaErrors(cudaFree(pconvbias));
		checkCudaErrors(cudaFree(pconvbiasGradient));
		checkCUDNN (cudnnDestroyConvolutionDescriptor(convDesc));
    checkCUDNN (cudnnDestroyTensorDescriptor (dstTensorDesc));
	}

};


///////////////////////////////////////////////////////////////////////////////////////////
// CUDNN/CUBLAS training context

class TrainingContext
{
	public :
	float alpha,beta; 
  cudnnHandle_t cudnnHandle;
  cublasHandle_t cublasHandle;
  int m_gpuid;
  int m_batchSize;
	//only the first tensor is need to store here
  cudnnTensorDescriptor_t dataTensor;

	vector<class baseModule *> vmod;
	int currentlayer;

	void * pworkspace;

	void print() {
					printf("TrainingContext m_gpuid %d m_batchSize %d\n",m_gpuid,m_batchSize);
					for(int i =0;i<vmod.size();i++) {
						vmod[i]->print();
					}
	}

  TrainingContext (int gpuid, int batch_size,float alpha_,float beta_)
  {
    m_batchSize = batch_size;
		m_gpuid =gpuid;
    printf ("gpuid %d batch_size %d\n", gpuid,batch_size);
		currentlayer=0;
		alpha=alpha_;
		beta=beta_;

    // Create CUBLAS and CUDNN handles
    checkCudaErrors (cudaSetDevice (gpuid));
    /*checkCudaErrors (*/cublasCreate (&cublasHandle)/*)*/;
    checkCUDNN (cudnnCreate (&cudnnHandle));

    // Create tensor descriptors
    checkCUDNN (cudnnCreateTensorDescriptor (&dataTensor));
  }

	void addMod(class baseModule * pmod) {
		vmod.push_back(pmod);
	}

	baseModule * getCurrentLayer() {
			assert(currentlayer >=0);
			if(currentlayer >= vmod.size()) {
					cout<<"currentlayer "<<currentlayer<<"vmod.size "<<vmod.size()<<endl;
					assert(0);
			}
			return vmod[currentlayer];
	}

	baseModule * getLastLayer() {
		return vmod[vmod.size()-1];
	}

	void reset() {
					currentlayer=0;
	}

	bool isForwardFinished() {
		 if(currentlayer>=vmod.size()) return true;
		 else return false;
	}
	bool isBackwardFinished() {
		 if(currentlayer==0) return true;
		 else return false;
	}

	void finishAddMod () {
		size_t maxsize=0;
		for(int i=0;i<vmod.size();i++) {
			maxsize = max(maxsize,vmod[i]->m_workspaceSizeByte);
//			cout<<"maxsize "<<vmod[i]->m_workspaceSizeByte <<endl;
		}
		//alloc new size
		if(maxsize>0) {
			allocAdviseMemory(&pworkspace,maxsize,m_gpuid);
		} else {
						maxsize = 0;
						pworkspace=NULL;
		}
		for(int i=0;i<vmod.size();i++) {
				vmod[i]->p_workspace = pworkspace;
				vmod[i]->m_workspaceSizeByte=maxsize;
		}
	}

   ~TrainingContext ()
  {
		for(int i=0;i<vmod.size();i++) {
			delete vmod[i];
		}
    checkCudaErrors (cudaSetDevice (m_gpuid));

    checkCUDNN (cudnnDestroyTensorDescriptor (dataTensor));
    /*checkCudaErrors (*/cublasDestroy (cublasHandle)/*)*/;
    checkCUDNN (cudnnDestroy (cudnnHandle));
  }

	 void ForwardPropagation1() {
		 if(currentlayer>=vmod.size()) {
 		  cout<<"finished at layer "<<currentlayer<<endl;
			assert(0);
		 } else {
				cout<<"layer "<<currentlayer<<endl;
        checkCudaErrors(cudaSetDevice(m_gpuid));

        // Conv1 layer
				vmod[currentlayer]->fw1step();
				
				currentlayer++;
		 }
	 }
	 void BackwardPropagation() {
		currentlayer--;
		 if(currentlayer<0) {
 		  cout<<"finished at layer "<<currentlayer<<endl;
			assert(0);
		 } else {
				cout<<"layer "<<currentlayer<<endl;
        checkCudaErrors(cudaSetDevice(m_gpuid));

        // Conv1 layer
				vmod[currentlayer]->bw1step();
		 }
	 }
};//end of TrainingContext

__global__ void
ssyinitfloat (float *p, size_t n)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;
  for (size_t idx = index; idx < n; idx += stride)
    {
      p[idx] = 0.0;
    }
}

struct runingConfig {
	size_t width, height;
	int iterations;
	int minib;
//	int chnl;
	bool copy;
	float fract;
	int num_gpus;
  vector < float *> * pd_dataV;
	vector <TrainingContext * > * pcontextV;
};

void construct_Lenet(struct runingConfig * prc ){
  float alpha = 1.0f, beta = 0.0f;
	size_t width = prc->width;
	size_t height = prc->height;
//	int iterations = prc->iterations;
	int minib = prc->minib;
	int chnl = 3;
//	bool copy = prc->copy;
//	float fract = prc->fract;
	int num_gpus = prc->num_gpus;
	vector < float *> * pd_dataV=prc->pd_dataV;
	vector <TrainingContext * > * pcontextV = prc->pcontextV;
  for (int gpuid = 0; gpuid < num_gpus; gpuid++)
    {
      checkCudaErrors (cudaSetDevice (gpuid));
			//alloc the input data
			float * pdata;
			size_t input_sz = minib*chnl*width*height;
			checkCudaErrors(cudaMallocManaged(&pdata,sizeof(float)*input_sz));
			//the context for this gpu
      TrainingContext * pcontext = new TrainingContext (gpuid, minib,alpha,beta);

      class ConvBiasLayer * pconv1=new ConvBiasLayer (
											alpha,
											beta,
											"conv1",
											pcontext->cudnnHandle,
											pcontext->cublasHandle,
											gpuid,
											minib,
											chnl,
											height,width, 
											20,5,1,
											0,0,
											pdata
											);
			class MaxPoolLayer * ppool1=new MaxPoolLayer (
											alpha,
											beta,
											"pool1",
											pcontext->cudnnHandle,
											pcontext->cublasHandle,
											gpuid,
											minib,
											pconv1->out_channels,
											pconv1->out_height, pconv1->out_width,
											2,2,
											0,0,
											pconv1->pout
											);
			class ConvBiasLayer * pconv2=new ConvBiasLayer (
											alpha,
											beta,
											"conv2",
											pcontext->cudnnHandle,
											pcontext->cublasHandle,
											gpuid,
											minib,
											ppool1->out_channels,
											ppool1->out_height, ppool1 ->out_width,
											50,5,1,
											0,0,
											ppool1->pout
											);
			class MaxPoolLayer * ppool2=new MaxPoolLayer (
											alpha,
											beta,
											"pool2",
											pcontext->cudnnHandle,
											pcontext->cublasHandle,
											gpuid,
											minib,
											pconv2->out_channels,
											pconv2->out_height, pconv2->out_width,
											2,2,
											0,0,
											pconv2->pout
											);

			pcontext -> addMod(pconv1);
			pcontext -> addMod(ppool1);
			pcontext -> addMod(pconv2);
			pcontext -> addMod(ppool2);
			pcontext -> finishAddMod();

      pcontextV->push_back (pcontext);
			pd_dataV->push_back(pdata);
	}

	for (int gpuid = 0; gpuid < num_gpus; gpuid++) {
					(*pcontextV)[gpuid]-> print();
	}
}
void construct_Resnet(struct runingConfig * prc ){
  float alpha = 1.0f, beta = 0.0f;
	size_t width = prc->width;
	size_t height = prc->height;
//	int iterations = prc->iterations;
	int minib = prc->minib;
	int chnl = 3;
//	bool copy = prc->copy;
//	float fract = prc->fract;
	int num_gpus = prc->num_gpus;
	vector < float *> * pd_dataV=prc->pd_dataV;
	vector <TrainingContext * > * pcontextV = prc->pcontextV;
  for (int gpuid = 0; gpuid < num_gpus; gpuid++)
    {
      checkCudaErrors (cudaSetDevice (gpuid));
			//alloc the input data
			float * pdata;
			size_t input_sz = minib*chnl*width*height;
			checkCudaErrors(cudaMallocManaged(&pdata,sizeof(float)*input_sz));
			//the context for this gpu
      TrainingContext * pcontext = new TrainingContext (gpuid, minib,alpha,beta);

      class ConvBiasLayer * pconv11=new ConvBiasLayer (
											alpha,
											beta,
											"conv1",
											pcontext->cudnnHandle,
											pcontext->cublasHandle,
											gpuid,
											minib,
											chnl,
											height,width, 
											64,7,2, //output filters number, kernel size,stride
											0,0,//padding
											pdata
											);
			pcontext -> addMod(pconv11);
			class MaxPoolLayer * ppool1=new MaxPoolLayer (
											alpha,
											beta,
											"conv2_pool",
											pcontext->cudnnHandle,
											pcontext->cublasHandle,
											gpuid,
											minib,
											pcontext->getLastLayer()->out_channels,
											pcontext->getLastLayer()->out_height, pcontext->getLastLayer()->out_width,
											3,2,//kernel size, stride
											0,0,//padding
											pcontext->getLastLayer()->pout
											);
			pcontext -> addMod(ppool1);
			//conv2
			for(int i =0;i<3;i++) {
	      class ConvBiasLayer * pconv1=new ConvBiasLayer (
											alpha,
											beta,
											"conv2_"+to_string(i)+"_1",
											pcontext->cudnnHandle,
											pcontext->cublasHandle,
											gpuid,
											minib,
											pcontext->getLastLayer()->out_channels,
											pcontext->getLastLayer()->out_height, pcontext->getLastLayer()->out_width,
											64,1,1, //output filters number, kernel size,stride
											0,0,//padding
											pcontext->getLastLayer()->pout
											);
				pcontext -> addMod(pconv1);

				class ConvBiasLayer * pconv2=new ConvBiasLayer (
											alpha,
											beta,
											"conv2_"+to_string(i)+"_2",
											pcontext->cudnnHandle,
											pcontext->cublasHandle,
											gpuid,
											minib,
											pcontext->getLastLayer()->out_channels,
											pcontext->getLastLayer()->out_height, pcontext->getLastLayer()->out_width,
											64,3,1,
											1,1,
											pcontext->getLastLayer()->pout
											);
				pcontext -> addMod(pconv2);
	      class ConvBiasLayer * pconv3=new ConvBiasLayer (
											alpha,
											beta,
											"conv2_"+to_string(i)+"_3",
											pcontext->cudnnHandle,
											pcontext->cublasHandle,
											gpuid,
											minib,
											pcontext->getLastLayer()->out_channels,
											pcontext->getLastLayer()->out_height, pcontext->getLastLayer()->out_width,
											256,1,1, //output filters number, kernel size,stride
											0,0,//padding
											pcontext->getLastLayer()->pout
											);
				pcontext -> addMod(pconv3);
			}
			//conv3
			for(int i =0;i<4;i++) {
	      class ConvBiasLayer * pconv1=new ConvBiasLayer (
											alpha,
											beta,
											"conv3_"+to_string(i)+"_1",
											pcontext->cudnnHandle,
											pcontext->cublasHandle,
											gpuid,
											minib,
											pcontext->getLastLayer()->out_channels,
											pcontext->getLastLayer()->out_height, pcontext->getLastLayer()->out_width,
											128,1,(i==0)?2:1, //output filters number, kernel size,stride
											0,0,//padding
											pcontext->getLastLayer()->pout
											);
				pcontext -> addMod(pconv1);

				class ConvBiasLayer * pconv2=new ConvBiasLayer (
											alpha,
											beta,
											"conv3_"+to_string(i)+"_2",
											pcontext->cudnnHandle,
											pcontext->cublasHandle,
											gpuid,
											minib,
											pcontext->getLastLayer()->out_channels,
											pcontext->getLastLayer()->out_height, pcontext->getLastLayer()->out_width,
											128,3,1,
											1,1,
											pcontext->getLastLayer()->pout
											);
				pcontext -> addMod(pconv2);
	      class ConvBiasLayer * pconv3=new ConvBiasLayer (
											alpha,
											beta,
											"conv3_"+to_string(i)+"_3",
											pcontext->cudnnHandle,
											pcontext->cublasHandle,
											gpuid,
											minib,
											pcontext->getLastLayer()->out_channels,
											pcontext->getLastLayer()->out_height, pcontext->getLastLayer()->out_width,
											512,1,1, //output filters number, kernel size,stride
											0,0,//padding
											pcontext->getLastLayer()->pout
											);
				pcontext -> addMod(pconv3);
			}
			//conv4
			for(int i =0;i<23;i++) {
	      class ConvBiasLayer * pconv1=new ConvBiasLayer (
											alpha,
											beta,
											"conv4_"+to_string(i)+"_1",
											pcontext->cudnnHandle,
											pcontext->cublasHandle,
											gpuid,
											minib,
											pcontext->getLastLayer()->out_channels,
											pcontext->getLastLayer()->out_height, pcontext->getLastLayer()->out_width,
											256,1,(i==0)?2:1, //output filters number, kernel size,stride
											0,0,//padding
											pcontext->getLastLayer()->pout
											);
				pcontext -> addMod(pconv1);

				class ConvBiasLayer * pconv2=new ConvBiasLayer (
											alpha,
											beta,
											"conv4_"+to_string(i)+"_2",
											pcontext->cudnnHandle,
											pcontext->cublasHandle,
											gpuid,
											minib,
											pcontext->getLastLayer()->out_channels,
											pcontext->getLastLayer()->out_height, pcontext->getLastLayer()->out_width,
											256,3,1,
											1,1,
											pcontext->getLastLayer()->pout
											);
				pcontext -> addMod(pconv2);
	      class ConvBiasLayer * pconv3=new ConvBiasLayer (
											alpha,
											beta,
											"conv4_"+to_string(i)+"_3",
											pcontext->cudnnHandle,
											pcontext->cublasHandle,
											gpuid,
											minib,
											pcontext->getLastLayer()->out_channels,
											pcontext->getLastLayer()->out_height, pcontext->getLastLayer()->out_width,
											1024,1,1, //output filters number, kernel size,stride
											0,0,//padding
											pcontext->getLastLayer()->pout
											);
				pcontext -> addMod(pconv3);
			}
			//conv5
			for(int i =0;i<3;i++) {
	      class ConvBiasLayer * pconv1=new ConvBiasLayer (
											alpha,
											beta,
											"conv5_"+to_string(i)+"_1",
											pcontext->cudnnHandle,
											pcontext->cublasHandle,
											gpuid,
											minib,
											pcontext->getLastLayer()->out_channels,
											pcontext->getLastLayer()->out_height, pcontext->getLastLayer()->out_width,
											512,1,(i==0)?2:1, //output filters number, kernel size,stride
											0,0,//padding
											pcontext->getLastLayer()->pout
											);
				pcontext -> addMod(pconv1);

				class ConvBiasLayer * pconv2=new ConvBiasLayer (
											alpha,
											beta,
											"conv5_"+to_string(i)+"_2",
											pcontext->cudnnHandle,
											pcontext->cublasHandle,
											gpuid,
											minib,
											pcontext->getLastLayer()->out_channels,
											pcontext->getLastLayer()->out_height, pcontext->getLastLayer()->out_width,
											512,3,1,
											1,1,
											pcontext->getLastLayer()->pout
											);
				pcontext -> addMod(pconv2);
	      class ConvBiasLayer * pconv3=new ConvBiasLayer (
											alpha,
											beta,
											"conv5_"+to_string(i)+"_3",
											pcontext->cudnnHandle,
											pcontext->cublasHandle,
											gpuid,
											minib,
											pcontext->getLastLayer()->out_channels,
											pcontext->getLastLayer()->out_height, pcontext->getLastLayer()->out_width,
											2048,1,1, //output filters number, kernel size,stride
											0,0,//padding
											pcontext->getLastLayer()->pout
											);
				pcontext -> addMod(pconv3);
			}

			pcontext -> finishAddMod();

			//setting the diff buffer pointer
			int lastl=(pcontext->vmod).size();
			size_t lastSizeInFloat = (pcontext->vmod)[lastl-1]->getOutputFloatNumber();
			float * plast;
    	checkCudaErrors (cudaMallocManaged (&plast, sizeof (float) * lastSizeInFloat));
			for(int i=(pcontext->vmod).size()-1;i>=0;i--) {
				(pcontext->vmod)[i]->poutDiff=plast;
				plast=(pcontext->vmod)[i]->pinDiff;
			}

      pcontextV->push_back (pcontext);
			pd_dataV->push_back(pdata);
	}

	for (int gpuid = 0; gpuid < num_gpus; gpuid++) {
					(*pcontextV)[gpuid]-> print();
	}
}

void syncAllGPU(int num_gpus) {
  for (int gpuid = 0; gpuid < num_gpus; gpuid++)
  {
    checkCudaErrors (cudaSetDevice (gpuid));
    checkCudaErrors (cudaDeviceSynchronize ());
  }
}

///////////////////////////////////////////////////////////////////////////////////////////
// Main function
//#define WIDTH 280
int
main (int argc, char **argv)
{
  if (argc != 8) {
		printf("Usage : cudnnModelParallel.exe <nettype> <iteration> <minbatch> <width>  <copy or not> <fract to copy> <advise or not>");
		assert(0);
	}

	cout<<"argc "<<argc<<endl;
	char * nettype =argv[1];
  int iterations = atoi (argv[2]);
	int minib = atoi(argv[3]);

  size_t width, height;
  width = atoi (argv[4]);
  height = width;
	cout<<"width "<<width<<endl;

  bool copy = (atoi (argv[5]) > 0);
  float fract = (atof (argv[6]));

  // Choose GPU
  int num_gpus;
  checkCudaErrors (cudaGetDeviceCount (&num_gpus));
	cout<<"num_gpus "<<num_gpus<<endl;

	int deviceId;
//  int numberOfSMs;
	checkCudaErrors(cudaSetDevice(0));
	cudaGetDevice(&deviceId);
	cout<<"deviceId "<<deviceId<<endl;

	advise= (atoi(argv[7]) >0);
//	cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);
	//printf("numberOfSMs %s\n",numberOfSMs);
//  int threadsPerBlock = 256;
//  int numberOfBlocks = 32*80 ;


  vector < float *>d_dataV;
	vector <TrainingContext * >contextV;
	
	//construct the resnet
	struct runingConfig rc;
	rc.width =width;
	rc.height = height;
	rc.iterations = iterations;
	rc.minib = minib;
//	rc.chnl = chnl;
	rc.copy = copy;
	rc.fract = fract;
	rc.num_gpus=num_gpus;
	rc.pd_dataV = & d_dataV;
	rc.pcontextV= & contextV;

	if(strcmp(nettype,"resnet")==0) {
		cout<<"resnet"<<endl;
		construct_Resnet(&rc);
	} else if(strcmp(nettype,"lenet")==0) {
		cout<<"lenet"<<endl;
		construct_Lenet(&rc);
	}

	syncAllGPU(num_gpus);
  // Use SGD to train the network
	size_t totalsize=0;
  auto t1 = chrono::high_resolution_clock::now ();
  for (int iter = 0; iter < iterations; ++iter)
  {
		//reset
		for(int gpuid=0;gpuid<num_gpus;gpuid++)     {
						contextV[gpuid]->reset();
		}
		//forward propage
		while(true) {
			//run one layer
		  for (int gpuid = 0; gpuid < num_gpus; gpuid++)
			{
				assert(contextV[gpuid]->m_gpuid == gpuid);
			  checkCudaErrors (cudaSetDevice (gpuid));
			  contextV[gpuid]->ForwardPropagation1 ();
				assert(contextV[gpuid]->currentlayer >0);
			}
			if(contextV[0]->isForwardFinished()) break;

			syncAllGPU(num_gpus);	

		  if (copy)
			{
			  for (int gpuid = 0; gpuid < num_gpus; gpuid++)
			  {
			      //sync n+1 to n
			      checkCudaErrors (cudaSetDevice (gpuid));
						baseModule * pcurrent =contextV[gpuid]->getCurrentLayer();
			      size_t sz = sizeof (float) * (pcurrent->getInputFloatNumber() );
						assert(sz>0);
						cout<<"sz "<<sz<<endl;

			      if (gpuid > 0 && pcurrent-> bNeedSyncInTensor) {
							baseModule * pPrev =contextV[gpuid-1]->getCurrentLayer();
							size_t szPrev = sizeof (float) * (pPrev->getInputFloatNumber() );
							assert(sz==szPrev);
							size_t tobetransfered = int (fract * sz / 2);
							if(gpuid == num_gpus-1)
								totalsize = totalsize+tobetransfered;
						  checkCudaErrors (cudaMemcpyAsync (pcurrent->pin + sz / (2 * sizeof (float)), pPrev->pin, tobetransfered, cudaMemcpyDefault));
						} else {
							cout <<"No need to sync : gpuid "<<gpuid << "layer "<<pcurrent->name<<endl;
						}
			  }
		
				syncAllGPU(num_gpus);
			}
		}
		//backward propage
		while(true) {
			//run one layer
		  for (int gpuid = 0; gpuid < num_gpus; gpuid++)
			{
				assert(contextV[gpuid]->m_gpuid == gpuid);
			  checkCudaErrors (cudaSetDevice (gpuid));
			  contextV[gpuid]->BackwardPropagation();
				assert(contextV[gpuid]->currentlayer >=0);
			}
			if(contextV[0]->isBackwardFinished()) break;

			syncAllGPU(num_gpus);	

		  if (copy)
			{
			  for (int gpuid = 0; gpuid < num_gpus; gpuid++)
			  {
			      //sync n+1 to n
			      checkCudaErrors (cudaSetDevice (gpuid));
						baseModule * pcurrent =contextV[gpuid]->getCurrentLayer();
			      size_t sz = sizeof (float) * (pcurrent->getOutputFloatNumber() );
						assert(sz>0);
						cout<<"sz "<<sz<<endl;

			      if (gpuid > 0 && pcurrent-> bNeedSyncInTensor) {
							baseModule * pPrev =contextV[gpuid-1]->getCurrentLayer();
							size_t szPrev = sizeof (float) * (pPrev->getOutputFloatNumber() );
							assert(sz==szPrev);
							size_t tobetransfered = int (fract * sz / 2);
							if(gpuid == num_gpus-1)
								totalsize = totalsize+tobetransfered;
						  checkCudaErrors (cudaMemcpyAsync (pcurrent->poutDiff + sz / (2 * sizeof (float)), pPrev->poutDiff, tobetransfered, cudaMemcpyDefault));
						} else {
							cout <<"No need to sync : gpuid "<<gpuid << "layer "<<pcurrent->name<<endl;
						}
			  }
		
				syncAllGPU(num_gpus);
			}
		}
  }				// end of iteration
	syncAllGPU(num_gpus);
  auto t2 = chrono::high_resolution_clock::now ();

  cout<<"Iteration time: "
			<<" num_gpus "<< num_gpus
			<<" batch_size "<< minib
			<<" width "<<width
			<<" fract "<< (copy?fract:0.0)
			<<" totalsize "<< totalsize/iterations
			<<" advise "<< (advise?1:0)
			<<" time " << chrono::duration_cast < chrono::microseconds > (t2 - t1).count () / 1000.0f / iterations
		<<" ms"<<endl;
  return 0;
}
