# Classification application based on libtorch

This repository is a "general" application for doing classification using libtorch.
It uses [yaml](https://github.com/jbeder/yaml-cpp) [input files](INPUT) for configuration of the classification process. 

# Dependencies 

* libtorch from [Pytorch](https://pytorch.org/)

#CMake variables for configuration

|name|values|description|
|----|------|-----------|
|MODEL|MPS,TTN,DGAN...|The torch module to use in the classification. One can create custom modules and add it to the namespace custom_models.|
|DATASET|FP2,IRIS,CMNIST,MNIST|The torch dataset to use in the classification. One can create custom datasets and add it to the namespace custom_models::datasets.|
|TRAIN|ON,OFF|Perform training on the model.|
|TEST|ON,OFF|Perform testing on the model.|

## Note on this

Custom modules and custom datasets have to have a constructor of the form like

|Object|Object constructor|
|-------------|------------------------------------|
|Custom_Module|Custom_ModuleImpl(YAML::Node config)|
|Custom_Dataset|Custom_DatasetImpl(const std::string& root, Mode mode = Mode::kTrain)|

# Install and Execute

### Build and install
```
git clone git@github.com:EddyTheCo/SUP.git SUP
cd SUP
cmake -DCMAKE_INSTALL_PREFIX=install -DTEST=ON -DTRAIN=ON -DMODEL=MPS -DDATASET=IRIS -DCUSTOM_MODULES="MPS" -DCUSTOM_DATASETS="IRIS" ../
cmake --build . --target install -- -j4
```

### Execute
```
./install/bin/SUP install/INPUT/mps_iris.yaml

```
