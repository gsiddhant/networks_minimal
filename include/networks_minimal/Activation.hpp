// Created by Siddhant Gangapurwala

#ifndef ACTIVATION_HPP
#define ACTIVATION_HPP

#include <Eigen/Dense>

namespace networks_minimal {

class Activation {
public:
    virtual Eigen::MatrixXd forward(const Eigen::MatrixXd &input);

    virtual Eigen::MatrixXd gradient(const Eigen::MatrixXd &input);
};

class ReLU : public Activation {
public:
    Eigen::MatrixXd forward(const Eigen::MatrixXd &input) override;

    Eigen::MatrixXd gradient(const Eigen::MatrixXd &input) override;
};

class TanH : public Activation {
public:
    Eigen::MatrixXd forward(const Eigen::MatrixXd &input) override;

    Eigen::MatrixXd gradient(const Eigen::MatrixXd &input) override;
};

class SoftSign : public Activation {
public:
    Eigen::MatrixXd forward(const Eigen::MatrixXd &input) override;

    Eigen::MatrixXd gradient(const Eigen::MatrixXd &input) override;
};

class Sigmoid : public Activation {
public:
    Eigen::MatrixXd forward(const Eigen::MatrixXd &input) override;

    Eigen::MatrixXd gradient(const Eigen::MatrixXd &input) override;
};

class LeakyReLU : public Activation {
public:
    Eigen::MatrixXd forward(const Eigen::MatrixXd &input) override;

    Eigen::MatrixXd gradient(const Eigen::MatrixXd &input) override;
};

extern struct ActivationHandler {
    ReLU relu;
    TanH tanh;
    SoftSign softsign;
    Sigmoid sigmoid;
    LeakyReLU leakyReLu;
} activation;

};

#endif // ACTIVATION_HPP
