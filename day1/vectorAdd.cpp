#include <iostream>
#include <vector>
using namespace std;

void addVectors(float* vec1, float* vec2, float* sum_vec, int size){
    for (int i = 0; i < size; i++){
        sum_vec[i] = vec1[i] + vec2[i];
    }
    for (int i = 0; i < size; i++){
        cout << sum_vec[i] << " ";
    }
}

int main() {
    const int vec_size = 3;

    float vec1[vec_size] = {1.0, 2.0, 3.0};
    float vec2[vec_size] = {4.0, 5.0, 6.0};
    
    float sum_vec[vec_size];

    addVectors(vec1, vec2, sum_vec, vec_size);
    
    return 0;
}