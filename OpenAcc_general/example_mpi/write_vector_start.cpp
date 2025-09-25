#include <iostream>
#include <iomanip>
#include <vector>
using namespace std; 

void write_vector_start(vector<double> vec, size_t blocksize){
    cout << fixed << setprecision(8);
    for (int i=0; i< blocksize; ++i){
      cout << setw(12) << vec[i];
    }
    cout << endl;
}

