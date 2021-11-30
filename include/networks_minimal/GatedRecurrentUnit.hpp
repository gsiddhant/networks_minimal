// Created by Siddhant Gangapurwala

#ifndef GATED_RECURRENT_UNIT_HPP
#define GATED_RECURRENT_UNIT_HPP

#include <vector>
#include <fstream>

#include "Activation.hpp"

class GatedRecurrentUnit {
public:
    GatedRecurrentUnit() = delete;

    GatedRecurrentUnit(const unsigned int &inputDimension, const unsigned int &hiddenStateDimension);

    GatedRecurrentUnit(const unsigned int &inputDimension, const unsigned int &hiddenStateDimension,
                       const std::string &networkParametersPath);

    void loadNetworkParametersFromFile(const std::string &networkParametersPath);

    const Eigen::MatrixXd &forward(const Eigen::MatrixXd &networkInput);

    const Eigen::MatrixXd &forward(const Eigen::MatrixXd &networkInput, const Eigen::MatrixXd &networkHiddenState);

    void resetHiddenState();

    void resetHiddenState(const Eigen::MatrixXd &networkHiddenState);

    const Eigen::MatrixXd &getNetworkHiddenState();

private:
    void resetNetworkParameters();

private:
    // Network Input and Hidden State Dimensions
    unsigned int inputDimension_;
    unsigned int hiddenStateDimension_, tripledHiddenStateDimension_;

    // Network Parameters
    std::string networkParametersPath_;
    Eigen::MatrixXd networkParameters_;

    std::vector <Eigen::MatrixXd> networkWeights_;
    std::vector <Eigen::MatrixXd> networkBiases_;

    Eigen::MatrixXd networkHiddenState_;

    // Forward pass containers
    Eigen::MatrixXd inputGatesLatent_, hiddenGatesLatent_;
    Eigen::MatrixXd rInputGatesLatent_, zInputGatesLatent_, nInputGatesLatent_;
    Eigen::MatrixXd rHiddenGatesLatent_, zHiddenGatesLatent_, nHiddenGatesLatent_;
    Eigen::MatrixXd rLatent_, zLatent_, nLatent_;
};

#endif // GATED_RECURRENT_UNIT_HPP
