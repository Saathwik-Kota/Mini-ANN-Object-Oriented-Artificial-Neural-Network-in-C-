#pragma once
#include <cmath>
#include "MathVector.hpp"

using namespace std;

class IActivation {
public:
    virtual double activate(double x) = 0;
    virtual double derivative(double x) = 0;
    virtual ~IActivation() = default;
};

class Sigmoid : public IActivation {
public:
    double activate(double x) override { return 1.0 / (1.0 + exp(-x)); }
    double derivative(double x) override { double sig = activate(x); return sig * (1.0 - sig); }
};

class ReLU : public IActivation {
public:
    double activate(double x) override { return (x > 0.0) ? x : 0.0; }
    double derivative(double x) override { return (x > 0.0) ? 1.0 : 0.0; }
};

class Tanh : public IActivation {
public:
    double activate(double x) override { return tanh(x); }
    double derivative(double x) override { double t = tanh(x); return 1.0 - (t * t); }
};

class ILossFunction {
public:
    virtual double calculate(double prediction, double target) = 0;
    virtual double derivative(double prediction, double target) = 0;
    virtual ~ILossFunction() = default;
};

class MSE : public ILossFunction {
public:
    double calculate(double prediction, double target) override { return 0.5 * pow((prediction - target), 2); }
    double derivative(double prediction, double target) override { return (prediction - target); }
};

class BinaryCrossEntropy : public ILossFunction {
public:

    double calculate(double prediction, double target) override {
        double epsilon = 1e-15; 
        prediction = max(epsilon, min(1.0 - epsilon, prediction));
        return -(target * log(prediction) + (1.0 - target) * log(1.0 - prediction));
    }

    double derivative(double prediction, double target) override {
        double epsilon = 1e-15;
        prediction = max(epsilon, min(1.0 - epsilon, prediction));
        return -(target / prediction) + ((1.0 - target) / (1.0 - prediction));
    }
};

class IOptimizer {
protected:
    double learningRate;
public:
    IOptimizer(double lr) : learningRate(lr) {}
    virtual MathVector calculateUpdate(const MathVector& gradients, MathVector& neuron_velocity) = 0;
    virtual double calculateUpdate(double gradient, double& neuron_bias_velocity) = 0;
    virtual ~IOptimizer() = default;
};

class SGD : public IOptimizer {
public:
    SGD(double lr) : IOptimizer(lr) {}
    MathVector calculateUpdate(const MathVector& gradients, MathVector& neuron_velocity) override { return gradients * learningRate; }
    double calculateUpdate(double gradient, double& neuron_bias_velocity) override { return gradient * learningRate; }
};

class Momentum : public IOptimizer {
private:
    double beta;
public:
    Momentum(double lr, double b = 0.9) : IOptimizer(lr), beta(b) {}
    MathVector calculateUpdate(const MathVector& gradients, MathVector& neuron_velocity) override {
        neuron_velocity = (neuron_velocity * beta) + (gradients * learningRate);
        return neuron_velocity;
    }
    double calculateUpdate(double gradient, double& neuron_bias_velocity) override {
        neuron_bias_velocity = (neuron_bias_velocity * beta) + (gradient * learningRate);
        return neuron_bias_velocity;
    }
};
