cmake_minimum_required(VERSION 3.11 FATAL_ERROR)
include(local_conf.cmake OPTIONAL)
project(Classification VERSION 0.1 DESCRIPTION "Torch Classification example" LANGUAGES CXX)


set(default_build_type "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
	set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
		STRING "Choose the type of build." FORCE)
	# Set the possible values of build type for cmake-gui
	set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
		"Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set(CMAKE_CXX_EXTENSIONS OFF)


add_executable(SUP SUP.cpp)
####################################CONFIG-VARIABLES#####################################
if(NOT MODEL)
	set(MODEL "MPS" CACHE STRING "MPS TTN DGAN ")
	message(STATUS "MODEL not set using default ${MODEL} ")
endif(NOT MODEL)
target_compile_definitions(SUP PRIVATE MODEL=${MODEL})
if(NOT DATASET)
	set(DATASET "FP2" CACHE STRING "FP2 IRIS CMNIST MNIST ")
	message(STATUS "DATASET not set using default ${DATASET} from options IRIS CMNIST MNIST FP2")
endif(NOT DATASET)
target_compile_definitions(SUP PRIVATE DATASET=${DATASET})

option(TRAIN "On or off" ON)
option(TEST "On or off" OFF)
option(USE_YAML "On or off" ON)
#########################################################################################
if(TRAIN)
	target_compile_definitions(SUP PRIVATE TRAIN)
endif(TRAIN)
if(TEST)
	target_compile_definitions(SUP PRIVATE TEST)
endif(TEST)


include(FetchContent)
find_package(Torch REQUIRED)

target_link_libraries(SUP PRIVATE  "${TORCH_LIBRARIES}")
if(USE_YAML)
	if(NOT TARGET yaml-cpp)
		find_package(YAML-CPP CONFIG QUIET)
		if(NOT YAML-CPP_FOUND)
			message(STATUS "YAML-CPP was not found. Downloading")
			FetchContent_Declare(
				yaml-cpp
				GIT_REPOSITORY git@github.com:jbeder/yaml-cpp.git
				GIT_TAG yaml-cpp-0.7.0
				)
			FetchContent_MakeAvailable(yaml-cpp)
		else()
			message(STATUS "YAML FOUND in ${YAML-CPP_DIR}")
		endif(NOT YAML-CPP_FOUND)
		target_link_libraries(SUP PRIVATE  yaml-cpp)
	endif(NOT TARGET yaml-cpp)
endif(USE_YAML)



FetchContent_Declare(
	custom_modules
	GIT_REPOSITORY git@github.com:EddyTheCo/custom_modules.git
	GIT_TAG main
	)
FetchContent_MakeAvailable(custom_modules)


target_link_libraries(SUP PRIVATE  custom_modules)


FetchContent_Declare(
	custom_datasets
	GIT_REPOSITORY git@github.com:EddyTheCo/custom_datasets.git
	GIT_TAG main
	)
FetchContent_MakeAvailable(custom_datasets)

target_link_libraries(SUP PRIVATE  custom_datasets)

FetchContent_Declare(
	yaml-torch
	GIT_REPOSITORY git@github.com:EddyTheCo/yaml-torch.git
	GIT_TAG main
	)
FetchContent_MakeAvailable(yaml-torch)
target_link_libraries(SUP PRIVATE  yaml-torch)

target_compile_features(SUP PUBLIC cxx_std_17)



install(TARGETS SUP DESTINATION bin)
install(DIRECTORY INPUT DESTINATION .)
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    include(CTest)
endif()
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME AND BUILD_TESTING)
    add_subdirectory(tests)
endif()
execute_process(COMMAND "git" "log" "--format='%H'" "-n 1"
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        OUTPUT_VARIABLE  commit)
execute_process(COMMAND "git" "rev-parse" "--abbrev-ref" "HEAD" 
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        OUTPUT_VARIABLE  branch)

FILE(WRITE ${CMAKE_CURRENT_BINARY_DIR}/.${PROJECT_NAME}.txt  ${PROJECT_NAME}:\n)
FILE(APPEND ${CMAKE_CURRENT_BINARY_DIR}/.${PROJECT_NAME}.txt  \t commit=${commit})
FILE(APPEND ${CMAKE_CURRENT_BINARY_DIR}/.${PROJECT_NAME}.txt  \t branch=${branch}\n)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/.${PROJECT_NAME}.txt DESTINATION .conf)
install(DIRECTORY gnuplot DESTINATION .)
