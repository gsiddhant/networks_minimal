cmake_minimum_required(VERSION 3.10)
project(networks_minimal)

set(CMAKE_CXX_STANDARD 17)

# Library Generation Path
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)

# Eigen3
find_package(Eigen3 REQUIRED HINTS ${Eigen3_HINT})

include_directories(${EIGEN3_INCLUDE_DIRS} include)

# Make the shared library
file(GLOB SOURCES src/*.cpp)
file(GLOB HEADERS include/${PROJECT_NAME}/*.hpp)

add_library(${PROJECT_NAME} SHARED ${SOURCES} ${HEADERS})
