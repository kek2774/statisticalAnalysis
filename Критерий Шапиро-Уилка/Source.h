#pragma once

#include <vector>
#include <utility>
#include <random>
#include <string>
#include <functional>
#include <stdexcept>

class Sample {
private:
    std::vector<double> sample;
    double mean = 0.0;
    double stdDev = 0.0;

public:
    Sample(const std::vector<double>& sample);

    double calculateMean() const;
    double calculateStandardDeviation() const;

    double getMean() const;
    double getStdDev() const;
    int getSampleSize() const;
    const std::vector<double>& getSample() const;
};

std::mt19937& globalGenerator();

int generateIntNumberFromAToB(std::mt19937& generator, int a = 20, int b = 45);

Sample generateSample(double mean, double stdDev);

double standardNormalCdf(double x);
double standardNormalQuantile(double p);
double standardNormalPdf(double x);
std::vector<double> computeStandardNormalOrderStatMeans(int n);
std::vector<std::vector<double>> computeStandardNormalOrderStatCovariance(int n);
std::vector<double> solveLinearSystemGaussJordan(std::vector<std::vector<double>> V, std::vector<double> b);
std::vector<double> computeShapiroWilkAiCoefficients(int n);

double calculateBCoef(const std::vector<double>& a_i, const Sample& sample);
double calculateWilkStat(double b, double s);
double calculateWAlpha(int n);

Sample generateInputFileWithNewSample(const std::string& filePath, double mean, double stdDev);

void writeShapiroWilkResultToFile(const std::string& filePath, const Sample& sample, const std::vector<double>& a_i, double alpha);
