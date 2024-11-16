#include <iostream> //for input and output (eg. printing to the console with std::cout)
#include <complex> //support for complex numbers
#include <vector> //allows us to use the std::vector type
#include <cmath> //mathematical functions like std::polar

const double PI = 3.141592653589793238460;

// Recursive function for 1D FFT
    //input is a vector of complex numbers, passed by reference &x, 
    //meaning that changes made to x inside the function affect the original data.
void fft(std::vector<std::complex<double>> &x) {
    int N = x.size();
    if (N <= 1) return; //If the array has only one element or is empty, the function stops (base case for recursion).

    // Divide: separate even and odd indices
    std::vector<std::complex<double>> even(N / 2), odd(N / 2);
    for (int i = 0; i < N / 2; ++i) {
        even[i] = x[i * 2]; //x[0] 2 4 ...N-2
        odd[i] = x[i * 2 + 1]; //1,3,5 ...N-1
    }

    // Recursively solve for even and odd parts
    fft(even); //call the fft function recursively on the even and odd parts. 
    fft(odd);  //This is the divide step of the divide-and-conquer algorithm.

    // Combine: FFT merge step
    for (int k = 0; k < N / 2; ++k) { //merges the results from the recursive calls for even and odd
        //generates a complex number on the unit circle with a given angle -2 * PI * k / N, 
        //corresponds to the FFT twiddle factor.
        std::complex<double> t = std::polar(1.0, -2 * PI * k / N) * odd[k];
        //The first half of x is filled with the sum of 
        //the corresponding even value and the transformed odd value.
        x[k] = even[k] + t; //0,1,2,...N/2-1
        x[k + N / 2] = even[k] - t; //N/2...N-1
    }
}

int main() { //main function that runs when the program starts.
    // Example input: 4-point complex data. {real part, imaginary part}
    std::vector<std::complex<double>> data = {
        {1.0, 0.0}, {2.0, 1.0}, {3.0, -1.0}, {4.0, 0.5},
    };

    // Perform FFT
    fft(data);

    // Output the results
    std::cout << "FFT output:\n";
    for (const auto& value : data) { //const makes the variable read-only,value will be a reference
        std::cout << value << "\n";
    }

    return 0;
}
