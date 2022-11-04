cmake_minimum_required(VERSION 3.11 FATAL_ERROR)

project(testing VERSION 0.1 DESCRIPTION "diferent tests for the library" LANGUAGES CXX)

if(MODEL STREQUAL MPS AND DATASET STREQUAL IRIS)
add_executable(mps_iris_test ../SUP.cpp testing.hpp)
target_include_directories(mps_iris_test PRIVATE .)


target_compile_definitions(mps_iris_test PRIVATE MODEL=MPS)
target_compile_definitions(mps_iris_test PRIVATE DATASET=IRIS)

target_compile_definitions(mps_iris_test PRIVATE TRAIN)
target_compile_definitions(mps_iris_test PRIVATE TEST)
target_compile_definitions(mps_iris_test PRIVATE BUILD_TESTING)
target_compile_definitions(mps_iris_test PRIVATE MPS_TESTING)

target_link_libraries(mps_iris_test PRIVATE  custom_datasets)
target_link_libraries(mps_iris_test PRIVATE  custom_modules)
target_link_libraries(mps_iris_test PRIVATE  "${TORCH_LIBRARIES}")
target_link_libraries(mps_iris_test PRIVATE  yaml-cpp)
target_link_libraries(mps_iris_test PRIVATE  yaml-torch)
add_test(NAME mps_iris_test COMMAND mps_iris_test ${PROJECT_SOURCE_DIR}/../INPUT/mps_iris.yaml)
endif(MODEL STREQUAL MPS AND DATASET STREQUAL IRIS)

if(MODEL STREQUAL MPS AND DATASET STREQUAL MNIST)
add_executable(mps_mnist_test ../SUP.cpp testing.hpp)
target_include_directories(mps_mnist_test PRIVATE .)


target_compile_definitions(mps_mnist_test PRIVATE MODEL=MPS)
target_compile_definitions(mps_mnist_test PRIVATE DATASET=MNIST)

target_compile_definitions(mps_mnist_test PRIVATE TRAIN)
target_compile_definitions(mps_mnist_test PRIVATE TEST)
target_compile_definitions(mps_mnist_test PRIVATE BUILD_TESTING)
target_compile_definitions(mps_mnist_test PRIVATE MPS_TESTING)

target_link_libraries(mps_mnist_test PRIVATE  custom_datasets)
target_link_libraries(mps_mnist_test PRIVATE  custom_modules)
target_link_libraries(mps_mnist_test PRIVATE  "${TORCH_LIBRARIES}")
target_link_libraries(mps_mnist_test PRIVATE  yaml-cpp)
target_link_libraries(mps_mnist_test PRIVATE  yaml-torch)
add_test(NAME mps_mnist_test COMMAND mps_mnist_test ${PROJECT_SOURCE_DIR}/../INPUT/mps_mnist.yaml)
endif(MODEL STREQUAL MPS AND DATASET STREQUAL MNIST)
