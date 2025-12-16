#include "Source.h"

#include <iostream>

using namespace std;

const string inputFilePath = "Inp/shapirowilktest.inp";
const string outputFilePath = "Out/shapirowilktest.out";

const double alpha = 0.05;
const double generateMean = 1.0;
const double generateStdDev = 0.03;

int main(const int argc, const char** argv) {
    setlocale(LC_ALL, "Russian");

    try {
        Sample sample = generateInputFileWithNewSample(inputFilePath, generateMean, generateStdDev);
        cout << "Входной файл с выборкой записан в: " << inputFilePath << endl;

        std::vector<double> a_i = computeShapiroWilkAiCoefficients(sample.getSampleSize());
        cout << "Коэффициенты a_i посчитаны для n = " << sample.getSampleSize() << endl;

        writeShapiroWilkResultToFile(outputFilePath, sample, a_i, alpha);
        cout << "Результаты критерия записаны в: " << outputFilePath << endl;
    }
    catch (const std::exception& ex) {
        cerr << "Ошибка: " << ex.what() << endl;
    }

    return 0;
}
