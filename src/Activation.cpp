// Created by Siddhant Gangapurwala

#include "networks_minimal/Activation.hpp"

namespace networks_minimal {

Eigen::MatrixXd Activation::forward(const Eigen::MatrixXd &input) {
    return input;
}

Eigen::MatrixXd Activation::gradient(const Eigen::MatrixXd &input) {
    return Eigen::MatrixXd::Ones(input.rows(), input.cols());
}

Eigen::MatrixXd ReLU::forward(const Eigen::MatrixXd &input) {
    return input.cwiseMax(0.);
}

Eigen::MatrixXd ReLU::gradient(const Eigen::MatrixXd &input) {
    return (input.array() > 0.).cast<double>();
}

Eigen::MatrixXd TanH::forward(const Eigen::MatrixXd &input) {
    return input.array().tanh();
}

Eigen::MatrixXd TanH::gradient(const Eigen::MatrixXd &input) {
    return 1. - input.array().tanh().pow(2);
}

Eigen::MatrixXd SoftSign::forward(const Eigen::MatrixXd &input) {
    return input.array() / (input.array().abs() + 1.);
}

Eigen::MatrixXd SoftSign::gradient(const Eigen::MatrixXd &input) {
    return 1. / (input.array().abs() + 1.).pow(2);
}

Eigen::MatrixXd Sigmoid::forward(const Eigen::MatrixXd &input) {
    return 1. / ((input.array() * -1.).exp() + 1.);
}

Eigen::MatrixXd Sigmoid::gradient(const Eigen::MatrixXd &input) {
    Eigen::MatrixXd sigmoid = Sigmoid::forward(input);
    return sigmoid.array() * (1 - sigmoid.array());
}

Eigen::MatrixXd LeakyReLU::forward(const Eigen::MatrixXd &input) {
    return input.cwiseMax(0.) + (0.01 * input.cwiseMin(0.));
}

Eigen::MatrixXd LeakyReLU::gradient(const Eigen::MatrixXd &input) {
    Eigen::MatrixXd leakyReLUGradient = input;

    for (auto r = 0; r < leakyReLUGradient.rows(); ++r) {
        for (auto c = 0; c < leakyReLUGradient.cols(); ++c) {
            if (leakyReLUGradient(r, c) > 0.) leakyReLUGradient(r, c) = 1.0;
            else if (leakyReLUGradient(r, c) < 0.) leakyReLUGradient(r, c) = 0.01;
        }
    }

    return leakyReLUGradient;
}

ActivationHandler activation;

};
