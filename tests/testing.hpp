#ifdef MPS_TESTING
#include<iostream>
#undef CHECK_TN 
#define CHECK_TN \
if(epoch>model->get_sin()) {\
	std::cout<<"epoch:"<<epoch<<" ortoIndex:"<<model->get_ortoCenter()<<std::endl;\
	const auto overlap=model->get_overlap().squeeze_();\
	std::cout<<"overlap:\n"<<overlap<<std::endl;\
	const auto trace=torch::trace(overlap);\
	std::cout<<"trace:"<<trace<<std::endl;\
	TORCH_CHECK((torch::abs(trace-1).template item<double>()<0.0001), "Error: the trace of the overlap is different from 1"); }\

#endif
