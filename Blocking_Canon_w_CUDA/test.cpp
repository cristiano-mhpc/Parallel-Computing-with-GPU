#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <iomanip> // For precision in output

// Function to read a matrix from a file
std::vector<std::vector<double>> readMatrix(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    std::vector<std::vector<double>> matrix;
    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::vector<double> row;
        double value;
        while (ss >> value) {
            row.push_back(value);
        }
        matrix.push_back(row);
    }
    file.close();
    return matrix;
}

// Function to multiply two matrices
std::vector<std::vector<double>> multiplyMatrices(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B) {
    size_t rowsA = A.size();
    size_t colsA = A[0].size();
    size_t colsB = B[0].size();

    std::vector<std::vector<double>> result(rowsA, std::vector<double>(colsB, 0.0));
    for (size_t i = 0; i < rowsA; ++i) {
        for (size_t j = 0; j < colsB; ++j) {
            for (size_t k = 0; k < colsA; ++k) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return result;
}

// Function to compare two matrices
int compareMatrices(const std::vector<std::vector<double>>& M1, const std::vector<std::vector<double>>& M2, double tolerance = 1) {
    int flag = 1;
    if (M1.size() != M2.size() || M1[0].size() != M2[0].size()) {
        return false;
    }
    for (size_t i = 0; i < M1.size(); ++i) {
        for (size_t j = 0; j < M1[0].size(); ++j) {
            if (std::abs(M1[i][j] - M2[i][j]) > tolerance) {
                std::cout<<"row: "<<i<<" col: "<<j<<" "<<M1[i][j]<<" "<<M2[i][j]<<" "<<std::abs(M1[i][j] - M2[i][j])<<std::endl;
		flag = 0;
//                return false;
            }
        }
    }
    return flag;
}

int main() {
    try {
        // Read matrices from files
        std::vector<std::vector<double>> A = readMatrix("A.txt");
        std::vector<std::vector<double>> B = readMatrix("B.txt");
        std::vector<std::vector<double>> C = readMatrix("C.txt");

        // Perform matrix multiplication
        std::vector<std::vector<double>> computedC = multiplyMatrices(A, B);

        // Check if computedC matches C
        if (compareMatrices(computedC, C)) {
            std::cout << "The multiplication result is correct!" << std::endl;
        } else {
            std::cout << "The multiplication result is incorrect!" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 0;
}
