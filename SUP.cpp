#include"custom_modules.hpp"
#include"custom_datasets.hpp"
#include"utils/yaml-torch.hpp"

using namespace custom_models;
using namespace custom_models::datasets;
using namespace torch::data::datasets;

template <typename DataLoader, typename module>
void train(size_t epoch, module& model,torch::Device device, DataLoader& data_loader,YAML::Node config) {

	model->train();
	int64_t correct = 0;
	double sumloss=0.;

	auto optim=yaml_interface::get_optimizer(config["Optimizer"],model->parameters());
	const static auto num_batches=(config["Train"])["Number of batches"].as<size_t>();
	size_t dataset_size=0;
	size_t iterator=0;

	for (auto& batch : data_loader) {
		if(iterator>=num_batches)break;
		auto data = batch.data.to(device).to(at::get_default_dtype_as_scalartype());
		dataset_size+=data.size(0);
		auto targets = batch.target.to(device).to(torch::kInt64);
		auto output = model(data);

		const auto pred = output.argmax(1);


		correct += pred.eq(targets).sum().template item<int64_t>();
		auto loss = torch::cross_entropy_loss(output, targets);
		AT_ASSERT(!std::isnan(loss.template item<float>()));

		optim->zero_grad();
		loss.backward();
		for(auto v:model->named_parameters())
		{
			std::cout<<v.key()<<"_grad:\n"<<v.value().grad().norm().template item<double>()<<std::endl;
		}
		optim->step();
		sumloss+=loss.template item<float>();
		iterator++;
	}
	model->update();
	std::cout<<"<Loss while training>:"<<sumloss/dataset_size<<std::endl;
	std::cout<<"<Accuracy in trainning>:"<<1.0*correct/dataset_size<<std::endl;
}

template <typename DataLoader, typename module>
void test( module& model, torch::Device device , DataLoader& data_loader,YAML::Node config) {

	torch::NoGradGuard no_grad;
	model->eval();

	const static auto num_batches=(config["Test"])["Number of batches"].as<size_t>();
	size_t dataset_size=0;
	int32_t correct = 0;
	size_t iterator=0;
	for (const auto& batch : data_loader) {
		if(iterator>=num_batches)break;
		auto data = batch.data.to(device);
		dataset_size+=data.size(0);
		auto targets = batch.target.to(device);
		auto output = model(data);

		const auto pred = output.argmax(1);
		correct += pred.eq(targets).sum().template item<int64_t>();
		iterator++;
	}
std::cout<<"<Accuracy in test>:"<<1.0*correct/dataset_size<<std::endl;
}

int main(int argc, char** argv)
{
	at::set_default_dtype(caffe2::TypeMeta::Make<double>());

	YAML::Node config = YAML::LoadFile(((argc>1)?argv[1]:"config.yaml"));
	const auto USE_GPU=config["USE_GPU"].as<bool>();
	torch::DeviceType device_type= torch::kCPU;
	if(USE_GPU)
	{
		if (torch::cuda::is_available()) {
			std::cout << "Device: GPU." << std::endl;
			device_type = torch::kCUDA;
		} else {
			std::cout << "Device: CPU." << std::endl;
		}
	}
	else
	{
		std::cout << "Device: CPU." << std::endl;
	}
	torch::Device device(device_type);

	auto model=MODEL(config["Model"]);


	if((config["Load and Save Module"])["Restart"].as<bool>())
	{
		std::cout<<"Loading module from file "<<(config["Load and Save Module"])["From"].as<std::string>()<<std::endl;
		torch::load(model,(config["Load and Save Module"])["From"].as<std::string>());
	}
	model->to(device);


#ifdef TRAIN
	auto train_dataset =DATASET((config["Dataset"])["From"].as<std::string>()).map(torch::data::transforms::Stack<>());
#endif
#ifdef TEST
	auto test_dataset = DATASET((config["Dataset"])["From"].as<std::string>(),DATASET::Mode::kTest).map(torch::data::transforms::Stack<>());
#endif



#ifdef TRAIN
	const size_t train_dataset_size = train_dataset.size().value();
	auto train_loader =
		torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
				std::move(train_dataset),(config["Train"])["Batch size"].as<size_t>() );
#endif
#ifdef TEST
	const size_t test_dataset_size = test_dataset.size().value();
	auto test_loader =
		torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
				std::move(test_dataset),(config["Test"])["Batch size"].as<size_t>() );
#endif



	for (size_t epoch = 1; epoch <= config["Number of epochs"].as<size_t>() ; ++epoch) {
		std::cout<<"Epoch:"<<epoch<<std::endl;
#ifdef TRAIN
		train(epoch, model,device, *train_loader,config);
#endif
#ifdef TEST
		test(model, device, *test_loader,config);
#endif
		if(epoch%((config["Load and Save Module"])["Save every"].as<size_t>())==0)
		{
			std::cout<<"Saving model to "<<(config["Load and Save Module"])["To"].as<std::string>()<<std::endl;
			torch::save(model, (config["Load and Save Module"])["To"].as<std::string>());
		}
		std::cout<<std::endl;
	}
	torch::save(model, (config["Load and Save Module"])["To"].as<std::string>());

}
