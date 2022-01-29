cmake_minimum_required(VERSION 3.10)
project(networks_minimal VERSION 0.1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(GNUInstallDirs)

find_package(Eigen3 REQUIRED)

include_directories(include ${EIGEN3_INCLUDE_DIRS})

add_library(${PROJECT_NAME}
            src/Activation.cpp
            src/GatedRecurrentUnit.cpp
            src/MultiLayerPerceptron.cpp)
target_include_directories(${PROJECT_NAME} PUBLIC 
                           "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
                           "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
                           )

install(DIRECTORY include/${PROJECT_NAME} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# export library
set(generated_dir "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(version_config "${generated_dir}/${PROJECT_NAME}ConfigVersion.cmake")
set(project_config "${generated_dir}/${PROJECT_NAME}Config.cmake")
set(targets_export_name "${PROJECT_NAME}Targets")
set(config_install_dir "share/${PROJECT_NAME}/cmake")

# install library
install(TARGETS ${PROJECT_NAME} EXPORT ${targets_export_name}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        )

install(EXPORT ${targets_export_name}
        DESTINATION ${config_install_dir})

install(FILES ${project_config} ${version_config} DESTINATION ${config_install_dir})

export(TARGETS ${PROJECT_NAME}
        FILE ${generated_dir}/${targets_export_name}.cmake)

# Include module with fuction 'write_basic_package_version_file'
include(CMakePackageConfigHelpers)

set(INCLUDE_INSTALL_DIR ${CMAKE_INSTALL_INCLUDEDIR})   
set(LIB_INSTALL_DIR ${CMAKE_INSTALL_LIBDIR})

# Configure '<PROJECT-NAME>Config.cmake'
# Use variables:
#   * targets_export_name
#   * PROJECT_NAME
configure_package_config_file(${CMAKE_CURRENT_LIST_DIR}/${PROJECT_NAME}Config.cmake.in
        "${project_config}"
        INSTALL_DESTINATION "${config_install_dir}"
        PATH_VARS INCLUDE_INSTALL_DIR LIB_INSTALL_DIR
        )

# Configure '<PROJECT-NAME>ConfigVersion.cmake'
# Note: PROJECT_VERSION is used as a VERSION
write_basic_package_version_file("${version_config}" COMPATIBILITY SameMajorVersion)

# add_library(${PROJECT_NAME} SHARED ${SOURCES} ${HEADERS})

# Test file
add_executable(boxy_actuation examples/boxy_actuation.cpp)
target_include_directories(boxy_actuation PUBLIC examples/helpers)
target_link_libraries(boxy_actuation PUBLIC ${PROJECT_NAME} stdc++fs)

add_executable(a1_actuation examples/a1_actuation.cpp)
target_include_directories(a1_actuation PUBLIC examples/helpers)
target_link_libraries(a1_actuation PUBLIC ${PROJECT_NAME} stdc++fs)

# Install catkin package.xml
install(FILES package.xml DESTINATION share/${PROJECT_NAME})

target_compile_definitions(${PROJECT_NAME} PUBLIC PATH_TO_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
