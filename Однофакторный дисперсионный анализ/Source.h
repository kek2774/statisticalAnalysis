#pragma once
#include <vector>
#include <random>
#include <string>
#include <utility>
#include <iomanip>

class Sample {
private:
    std::vector<double> sample;
    double mean;
    double stdDev;
    double calculateMean() const;
    double calculateStandardDeviation() const;
public:
    explicit Sample(const std::vector<double>& sample);
    double getMean() const;
    double getStdDev() const;
    int getSampleSize() const;
    const std::vector<double>& getSample() const;
};

std::mt19937& globalGenerator();

int generateIntNumberFromAToB(std::mt19937& generator, int a = 20, int b = 45);

std::vector<double> generateSample(double mean, double stdDev);

std::vector<Sample> generateListOfSamples(const int numberOfSamples);

double calculateGeneralMean(const std::vector<Sample>& samples);

double calculateOuterDispersion(const std::vector<Sample>& samples, double generalMean);

double calculateInnerDispersion(const std::vector<Sample>& samples);

double calculateFisherStat(double sOut, double sIn);

std::pair<int, int> calculateFreedomDegreees(const std::vector<Sample>& samples);

double calculateFAlpha(const std::pair<int, int>& freedomDegrees, double alpha);

void generateInputFileIfEmpty(const std::string& fileName, int numberOfSamples, std::vector<Sample>& samples);

void writeAnovaToFile(const std::string& fileName, double alpha, const std::vector<Sample>& samples);
