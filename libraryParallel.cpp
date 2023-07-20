//
// Created by ovelardo on 20/07/23.
//

#include <iostream>
#include <vector>
#include "libraryParallel.h"
#include <thread>
#include <mutex>
#include <algorithm>
#include <cmath>
#include <omp.h>
#include <fstream>
#include <climits>

#ifdef _WIN32
// Windows platform
    #define LIBRARY_API __declspec(dllexport)
#else
// Non-Windows platforms
#define LIBRARY_API
#endif

using namespace std;

///////Funciones en paralelo

void rotateP(unsigned short* src, unsigned short* dst, int width, int height, int direction)
{
    if (direction == 1) {
#pragma omp parallel for collapse(2)
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                dst[j * height + (height - i - 1)] = src[i * width + j];
            }
        }
    } else if (direction == 2) {
#pragma omp parallel for collapse(2)
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                dst[(height - i - 1) * width + (width - j - 1)] = src[i * width + j];
            }
        }
    } else if (direction == 3) {
#pragma omp parallel for collapse(2)
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                dst[(width - j - 1) * height + i] = src[i * width + j];
            }
        }
    }
}


void flipP(unsigned short* src, unsigned short* dst, int width, int height, int direction)
{
#pragma omp parallel for collapse(2)
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            if (direction == 1) {
                dst[i * width + (width - j - 1)] = src[i * width + j];
            } else if (direction == 2) {
                dst[(height - i - 1) * width + j] = src[i * width + j];
            }
        }
    }
}

