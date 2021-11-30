// Created by Siddhant Gangapurwala

#ifndef MULTI_LAYER_PERCEPTRON_HPP
#define MULTI_LAYER_PERCEPTRON_HPP

#include <vector>
#include <fstream>

#include "Activation.hpp"

class MultiLayerPerceptron {
public:
    MultiLayerPerceptron() = delete;

    MultiLayerPerceptron(const std::vector<unsigned int> &networkLayers,
                         const std::reference_wrapper <Activation> &networkActivation,
                         bool outputActivation = false);

    MultiLayerPerceptron(const std::vector<unsigned int> &networkLayers,
                         const std::reference_wrapper <Activation> &networkActivation,
                         const std::string &networkParametersPath, bool outputActivation = false);

    MultiLayerPerceptron(const std::vector<unsigned int> &networkLayers,
                         const std::vector <std::reference_wrapper<Activation>> &networkActivations);

    MultiLayerPerceptron(const std::vector<unsigned int> &networkLayers,
                         const std::vector <std::reference_wrapper<Activation>> &networkActivations,
                         const std::string &networkParametersPath);

    void loadNetworkParametersFromFile(const std::string &networkParametersPath);

    const Eigen::MatrixXd &forward(const Eigen::MatrixXd &networkInput);

    const Eigen::MatrixXd &gradient(const Eigen::MatrixXd &networkInput);

    const Eigen::MatrixXd &latentLayerOutput(const int &layer);

private:
    void initializeNetworkVariables();

    void resetNetworkParameters();

private:
    // Network Parameters
    std::string networkParametersPath_;
    Eigen::MatrixXd networkParameters_;

    std::vector <Eigen::MatrixXd> networkWeights_;
    std::vector <Eigen::MatrixXd> networkBiases_;

    // Network Layers and Activations
    std::vector<unsigned int> networkLayers_;
    std::vector <std::reference_wrapper<Activation>> networkActivations_;

    // Forward Pass Variable
    std::vector <Eigen::MatrixXd> latentOutput_;

    // Gradient Variables
    std::vector <Eigen::MatrixXd> latentLinearOutput_;
    Eigen::MatrixXd networkDerivative_;

    // Flags
    bool outputActivation_;

};

#endif // MULTI_LAYER_PERCEPTRON_HPP
