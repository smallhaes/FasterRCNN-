TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
NSYNC_INC=$TF_INC"/external/nsync/public"

CUDA_PATH=/usr/local/cuda/
CXXFLAGS='-D_GLIBCXX_USE_CXX11_ABI=0'

if [[ "$OSTYPE" =~ ^darwin ]]; then
		CXXFLAGS+='-undefined dynamic_lookup'
	fi
	
	cd roi_pooling_layer
	
	if [ -d "$CUDA_PATH" ]; then
			nvcc -std=c++11 -c -o roi_pooling_op.cu.o roi_pooling_op_gpu.cu.cc \
						-I $TF_INC -I $NSYNC_INC -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC $CXXFLAGS \
								-arch=sm_37 -L $TF_LIB -ltensorflow_framework
			
				g++ -std=c++11 -shared -o roi_pooling.so roi_pooling_op.cc \
							roi_pooling_op.cu.o -I $TF_INC  -I $NSYNC_INC -D GOOGLE_CUDA=1 -fPIC $CXXFLAGS \
									-lcudart -L $CUDA_PATH/lib64 -L $TF_LIB -ltensorflow_framework
			else
					g++ -std=c++11 -shared -o roi_pooling.so roi_pooling_op.cc \
								-I $TF_INC -I $NSYNC_INC -fPIC $CXXFLAGS -L $TF_LIB -ltensorflow_framework
				fi
