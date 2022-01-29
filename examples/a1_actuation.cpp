// Created by Siddhant Gangapurwala


#include <iostream>
#include <experimental/filesystem>

#include "networks_minimal/Config.h"
#include "Actuation.hpp"


int main() {
    std::string currentPath(DIR_PATH);
    std::string parametersDirectory(currentPath + "/examples/parameters/");
    std::cout << parametersDirectory << std::endl;

    Eigen::MatrixXd networkInputScaling(2, 1);
    networkInputScaling.col(0) << 1., 0.05;

    double networkOutputScaling = 40.;

    Actuation actuation(parametersDirectory + "a1", networkInputScaling, networkOutputScaling, 12);
    actuation.reset();

    Eigen::MatrixXd jointPositionErrors, jointVelocities;
    jointPositionErrors.setOnes(12, 1);
    jointVelocities.setOnes(12, 1);

    for (auto s = 0; s < 10; ++s) {
        std::cout << actuation.getActuationTorques(jointPositionErrors, jointVelocities).transpose() << std::endl;
    }

    actuation.reset();
    std::cout << "\nRecurrent State Reset" << std::endl;

    for (auto s = 0; s < 10; ++s) {
        std::cout << actuation.getActuationTorques(jointPositionErrors, jointVelocities).transpose() << std::endl;
    }

    std::cout << "\nComputing Inference Latency" << std::endl;

    std::chrono::high_resolution_clock::time_point startTimePoint{}, endTimePoint{};

    jointPositionErrors.setRandom(12, 1000);
    jointVelocities.setRandom(12, 1000);

    std::array<long, 1000> latency{};

    for (auto c = 0; c < 1000; ++c) {
        startTimePoint = std::chrono::high_resolution_clock::now();
        actuation.getActuationTorques(jointPositionErrors.col(c), jointVelocities.col(c));
        endTimePoint = std::chrono::high_resolution_clock::now();

        latency[c] = std::chrono::duration_cast<std::chrono::microseconds>(endTimePoint - startTimePoint).count();
    }

    double mean = 0;

    for (auto c = 0; c < 1000; ++c) {
        mean += static_cast<double>(latency[c]);
    }

    mean = mean / 1000.;

    std::cout << "Latency Mean: " << mean << " us" << std::endl;

    double std = 0;

    for (auto c = 0; c < 1000; ++c) {
        std += std::pow(static_cast<double>(latency[c]) - mean, 2);
    }

    std = std::pow(std / 1000., 0.5);

    std::cout << "Latency Std: " << std << " us" << std::endl;

    return 0;
}