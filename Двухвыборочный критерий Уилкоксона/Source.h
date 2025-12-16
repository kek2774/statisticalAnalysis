#pragma once
#include <vector>
#include <random>
#include <string>
#include <utility>

class Sample {
private:
    std::vector<double> sample;
    double mean;
    double stdDev;
    double calculateMean() const;
    double calculateStandardDeviation() const;
public:
    Sample();
    explicit Sample(const std::vector<double>& sample);
    double getMean() const;
    double getStdDev() const;
    int getSampleSize() const;
    const std::vector<double>& getSample() const;
};

std::mt19937& globalGenerator();
int generateIntNumberFromAToB(std::mt19937& generator, int a, int b);
double normPPF(double alpha);
Sample generateSampleNormal(double mean, double stdDev, int n, bool shift = false);

double calculateWilcoxonStat(const Sample& sample1, const Sample& sample2);
std::pair<double, double> calculateApproximately(int m, int n, double alpha, bool oneSided);
long double binomial_coefficient(int n, int k);
void buildExactWilcoxonDistribution(int m, int n, int& minW, int& maxW, int& maxU, std::vector<long double>& pU, std::vector<long double>& cdfU);
int findQuantileU(const std::vector<long double>& cdfU, int maxU, long double alpha);
int convertUtoW(int minW, int U_quantile);
std::pair<double, double> calculatePrecisely(int m, int n, double alpha, bool oneSided);
std::pair<double, double> calculateWCrit(const Sample& sample1, const Sample& sample2, double alpha, bool oneSided);

enum class WilcoxonDecision { AcceptH0, RejectH0 };
WilcoxonDecision wilcoxonTwoSidedDecision(const Sample& sample1, const Sample& sample2, double alpha);

void generateTwoSidedWilcoxonInp(double alpha);
void runTwoSidedWilcoxonFromFile();
