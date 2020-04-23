cmake_minimum_required(VERSION 2.8)

project(cse576-2020sp-HW3)

set(CMAKE_CXX_FLAGS "-fdiagnostics-color=always -std=c++11 -pthread -O3 -g -march=native -fPIC")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/)

add_library(nn SHARED
        src/activations.cpp # modify this file
        src/classifier.cpp # modify this file
        src/matrix.cpp
        src/data.cpp
        src/matrix.h
        src/neural.h
        src/utils.h
        )

link_libraries(nn)

add_executable(train src/train.cpp)

