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

#include <omp.h>

void rotateP(unsigned short* src, unsigned short* dst, int width, int height, int direction)
{
    int numThreads = 8; // Puedes ajustar este valor según tus necesidades
    omp_set_num_threads(numThreads);

    if (direction == 1) {
#pragma omp parallel for
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                dst[j * height + (height - i - 1)] = src[i * width + j];
            }
        }
    } else if (direction == 2) {
#pragma omp parallel for
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                dst[(height - i - 1) * width + (width - j - 1)] = src[i * width + j];
            }
        }
    } else if (direction == 3) {
#pragma omp parallel for
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                dst[j * height + i] = src[i * width + (width - j - 1)];
            }
        }
    }
}

void rotateP2(unsigned short* src, unsigned short* dst, int width, int height, int direction)
{
    int numThreads = 8; // Puedes ajustar este valor según tus necesidades
    omp_set_num_threads(numThreads);

    if (direction == 1) {
#pragma omp parallel for collapse(2)
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                dst[j * height + (height - i - 1)] = src[i * width + j];
            }
        }
    } else if (direction == 2) {
#pragma omp parallel for
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                dst[(height - i - 1) * width + (width - j - 1)] = src[i * width + j];
            }
        }
    } else if (direction == 3) {
#pragma omp parallel for collapse(2)
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                dst[j * height + i] = src[i * width + (width - j - 1)];
            }
        }
    }
}



void flipP(unsigned short* src, unsigned short* dst, int width, int height, int direction)
{
    int numThreads = 8;
    omp_set_num_threads(numThreads);

#pragma omp parallel for
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


void adjustContrastP(unsigned short* src, unsigned short* dst, int rows, int cols, float contrastLevel, float amplificationFactor)
{
    float gaussFactor = std::exp(-0.5f * std::pow((contrastLevel - 128.0f) / 128.0f, 2));

    unsigned short minVal = std::numeric_limits<unsigned short>::max();
    unsigned short maxVal = std::numeric_limits<unsigned short>::min();

    int numThreads = 8;
    omp_set_num_threads(numThreads);

    // Calculamos los valores mínimo y máximo de la imagen original
#pragma omp parallel for reduction(min: minVal) reduction(max: maxVal)
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            unsigned short val = src[i * cols + j];
            minVal = std::min(minVal, val);
            maxVal = std::max(maxVal, val);
        }
    }

    // Calculamos el rango de valores de la imagen
    float range = maxVal - minVal;

    // Ajustamos los valores de contraste y amplificación
    float contrastScale = 65535.0f / range;
    float amplificationScale = amplificationFactor * contrastScale;

    // Ajuste para evitar valores negativos
    float adjustmentOffset = minVal * contrastScale * gaussFactor * amplificationScale;

#pragma omp parallel for
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            unsigned short val = src[i * cols + j];

            // Aplicamos el ajuste de contraste y amplificación
            float adjustedValue = (val * contrastScale * gaussFactor * amplificationScale) - adjustmentOffset;

            // Ajustamos los valores al rango ampliado
            adjustedValue = std::max(adjustedValue, -131070.0f);
            adjustedValue = std::min(adjustedValue, 327670.0f);

            // Mapeamos los valores al rango 0-65535
            adjustedValue = (adjustedValue + 131070.0f) * (65535.0f / 458740.0f);

            dst[i * cols + j] = static_cast<unsigned short>(adjustedValue);
        }
    }
}



void highPassContrastP(unsigned short* src, unsigned short* dst, int width, int height, float threshold, float contrastBoost)
{
    std::vector<int> kernelX = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };
    std::vector<int> kernelY = { -1, -2, -1, 0, 0, 0, 1, 2, 1 };
    const int kernelSize = 3;
    const int border = kernelSize / 2;

    int numThreads = 6;
    omp_set_num_threads(numThreads);

    // Aplica el filtro de paso alto utilizando el kernel de Sobel
#pragma omp parallel for
    for (int y = border; y < height - border; ++y) {
        for (int x = border; x < width - border; ++x) {
            int sumX = 0;
            int sumY = 0;

            // Calcula las sumas ponderadas en las direcciones horizontal y vertical
            for (int ky = 0; ky < kernelSize; ++ky) {
                for (int kx = 0; kx < kernelSize; ++kx) {
                    int index = (y + ky - border) * width + (x + kx - border);
                    sumX += kernelX[ky * kernelSize + kx] * src[index];
                    sumY += kernelY[ky * kernelSize + kx] * src[index];
                }
            }

            // Calcula la magnitud del gradiente
            float gradient = std::sqrt(static_cast<float>(sumX * sumX + sumY * sumY));

            // Aplica el contraste proporcionalmente al valor alto
            float result = src[y * width + x] + static_cast<float>(contrastBoost * gradient);

            // Ajustamos los valores al rango ampliado
            result = std::max(result, -131070.0f);
            result = std::min(result, 327670.0f);

            // Mapeamos los valores al rango 0-65535
            result = (result + 131070.0f) * (65535.0f / 458740.0f);

            // Aplica el umbral para resaltar los bordes
            if (std::abs(result - src[y * width + x]) > threshold) {
                dst[y * width + x] = static_cast<unsigned short>(result);
            } else {
                dst[y * width + x] = src[y * width + x];
            }
        }
    }
}


void deteccionBordesP(unsigned short* src, unsigned short* dst, int width, int height, float threshold, int amplificationFactor)
{
    std::vector<int> kernelX = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };
    std::vector<int> kernelY = { -1, -2, -1, 0, 0, 0, 1, 2, 1 };
    const int kernel_size = 3;
    const int border = kernel_size / 2;
    const float norm_factor = threshold / 65535.0f;

    int numThreads = 8;
    omp_set_num_threads(numThreads);

    // Iterar sobre los píxeles de la imagen
#pragma omp parallel for
    for (int i = border; i < height - border; ++i)
    {
        for (int j = border; j < width - border; ++j)
        {
            int sumX = 0;
            int sumY = 0;
            // Iterar sobre el kernel en X
            for (int k = 0; k < kernel_size; ++k)
            {
                for (int l = 0; l < kernel_size; ++l)
                {
                    sumX += kernelX[k * kernel_size + l] * src[(i - border + k) * width + (j - border + l)];
                    sumY += kernelY[k * kernel_size + l] * src[(i - border + k) * width + (j - border + l)];
                }
            }
            // Calcular la magnitud del gradiente
            int magnitude = std::abs(sumX) + std::abs(sumY);
            // Aplicar el umbral y la amplificación
            int result = static_cast<int>(src[i * width + j]) + static_cast<int>(norm_factor * magnitude * amplificationFactor);
            // Ajustar el resultado al rango 0-65535
            result = std::max(result, 0);
            result = std::min(result, 65535);
            dst[i * width + j] = static_cast<unsigned short>(result);
        }
    }
}


void boostLowContrastP(unsigned short* src, unsigned short* dst, int width, int height, float threshold, float contrastBoost)
{
    float maxValInThreshold = 0;
    int* iDst = new int[width * height];

    int numThreads = 8;
    omp_set_num_threads(numThreads);

    // Aplicamos el aumento de contraste a la zona dentro del threshold
#pragma omp parallel for
    for (int i = 0; i < width * height; ++i) {
        if (src[i] <= threshold) {
            // Aplicamos el contraste a los valores por debajo del threshold
            float val = static_cast<float>(src[i]);
            val = val * contrastBoost;
            iDst[i] = static_cast<int>(val);
            maxValInThreshold = std::max(maxValInThreshold, static_cast<float>(iDst[i]));
        }
    }

    // Ajustamos los valores por encima del threshold de manera lineal
#pragma omp parallel for
    for (int i = 0; i < width * height; ++i) {
        if (src[i] > threshold) {
            // Ajustamos los valores linealmente
            float val = src[i];
            val = maxValInThreshold + (val - threshold);
            iDst[i] = val;
        }
    }

    adjustToRangeP(iDst, dst, width * height);
}

#include <omp.h>

void adjustToRangeP(int* iDst, unsigned short* dst, int size)
{
    int minVal = std::numeric_limits<int>::max();
    int maxVal = std::numeric_limits<int>::min();

    int numThreads = 8;
    omp_set_num_threads(numThreads);

    // Encontramos el valor mínimo y máximo en la imagen
#pragma omp parallel for reduction(min:minVal) reduction(max:maxVal)
    for (int i = 0; i < size; ++i) {
        minVal = std::min(minVal, iDst[i]);
        maxVal = std::max(maxVal, iDst[i]);
    }

    // Ajustamos los valores al rango 0-65535
    float factor = 65535.0f / static_cast<float>(maxVal - minVal);

#pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        float val = static_cast<float>(iDst[i] - minVal) * factor;
        dst[i] = static_cast<unsigned short>(val);
    }
}


void perfiladoP(unsigned short* src, unsigned short* dst, int width, int height, float threshold, int amplificationFactor, int kernelSize)
{
    std::vector<int> kernel;

    if (kernelSize == 3) {
        kernel = { -1, -1, -1, -1, amplificationFactor, -1, -1, -1, -1 };
    } else if (kernelSize == 5) {
        kernel = { -1, -1, -1, -1, -1, -1, amplificationFactor, amplificationFactor, amplificationFactor, -1, -1, amplificationFactor, 9 * amplificationFactor, amplificationFactor, -1, -1, amplificationFactor, amplificationFactor, amplificationFactor, -1, -1, -1, -1, -1, -1 };
    } else if (kernelSize == 7) {
        kernel = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, amplificationFactor, amplificationFactor, amplificationFactor, amplificationFactor, amplificationFactor, -1, -1, amplificationFactor, 4 * amplificationFactor, 4 * amplificationFactor, 4 * amplificationFactor, amplificationFactor, -1, -1, amplificationFactor, 4 * amplificationFactor, 9 * amplificationFactor, 4 * amplificationFactor, amplificationFactor, -1, -1, amplificationFactor, 4 * amplificationFactor, 4 * amplificationFactor, 4 * amplificationFactor, amplificationFactor, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 };
    } else {
        // Valor por defecto: kernel de tamaño 3x3
        kernel = { -1, -1, -1, -1, amplificationFactor, -1, -1, -1, -1 };
        kernelSize = 3;
    }

    const int border = kernelSize / 2;
    const float norm_factor = threshold / 65535.0f;

    int numThreads = 8;
    omp_set_num_threads(numThreads);

    // Iterar sobre los píxeles de la imagen
#pragma omp parallel for
    for (int i = border; i < height - border; ++i)
    {
        for (int j = border; j < width - border; ++j)
        {
            int sum = 0;
            // Iterar sobre el kernel
            for (int k = 0; k < kernelSize; ++k)
            {
                for (int l = 0; l < kernelSize; ++l)
                {
                    sum += kernel[k * kernelSize + l] * src[(i - border + k) * width + (j - border + l)];
                }
            }
            // Aplicar el perfilado y el amplificación
            int result = static_cast<int>(src[i * width + j]) + static_cast<int>(norm_factor * sum);
            // Ajustar el resultado al rango 0-65535
            result = std::max(result, 0);
            result = std::min(result, 65535);
            dst[i * width + j] = static_cast<unsigned short>(result);
        }
    }
}


void adjustBrightnessP(unsigned short* src, unsigned short* dst, int width, int height, float contrastLevel)
{
    int numThreads = 8;
    omp_set_num_threads(numThreads);

    // Calcular el histograma en paralelo
    std::vector<int> histogram(65536, 0);
#pragma omp parallel for
    for (int i = 0; i < width * height; ++i) {
        ++histogram[src[i]];
    }

    // Calcular el número acumulativo de píxeles en el histograma en paralelo
    std::vector<int> cumulativeHistogram(65536, 0);
    cumulativeHistogram[0] = histogram[0];
#pragma omp parallel for
    for (int i = 1; i < 65536; ++i) {
        cumulativeHistogram[i] = cumulativeHistogram[i - 1] + histogram[i];
    }

    // Calcular el valor mínimo y máximo del histograma en paralelo
    int minValue = 0;
    int maxValue = 65535;
    while (histogram[minValue] == 0) {
        ++minValue;
    }
    while (histogram[maxValue] == 0) {
        --maxValue;
    }

    // Calcular el rango dinámico del histograma
    float dynamicRange = maxValue - minValue;

    // Calcular el valor objetivo de cada nivel de gris en paralelo
    std::vector<unsigned short> mappingTable(65536, 0);
#pragma omp parallel for
    for (int i = 0; i < 65536; ++i) {
        float normalizedValue = (i - minValue) / dynamicRange;
        float contrastAdjustedValue = std::pow(normalizedValue, contrastLevel);
        unsigned short mappedValue = static_cast<unsigned short>(contrastAdjustedValue * dynamicRange + minValue);
        mappingTable[i] = std::max(std::min(mappedValue, static_cast<unsigned short>(65535)), static_cast<unsigned short>(0));
    }

    // Aplicar la transformación de contraste a la imagen en paralelo
#pragma omp parallel for
    for (int i = 0; i < width * height; ++i) {
        dst[i] = mappingTable[src[i]];
    }
}

int backgroundSubtractionP(unsigned short* src, unsigned short* dst, int width, int height, float sigma)
{
    int result = 0;

    try {

        // Calcula el tamaño de la matriz gaussiana
        int size = 2 * static_cast<int>(std::ceil(3 * sigma)) + 1;

        // Calcula la matriz gaussiana
        std::vector<float> gaussianMatrix(size * size);
        float sum = 0.0f;

        result = 3;

#pragma omp parallel for reduction(+:sum)
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                float x = i - size / 2;
                float y = j - size / 2;
                //float value = std::exp(-(x * x + y * y) / (2 * sigma * sigma));
                float exponent = -(x * x + y * y) / (2 * sigma * sigma);
                float value = std::exp(exponent);

                gaussianMatrix[i * size + j] = value;
                sum += value;
            }
        }

        result = 2;

        // Normaliza la matriz gaussiana para que sume 1
#pragma omp parallel for
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                gaussianMatrix[i * size + j] /= sum;
            }
        }

        result = 1;

        // Realiza la convolución de la imagen con la matriz gaussiana
        int border = size / 2;
        //float threshold = 1000.0f; // Puedes ajustar este umbral según tus necesidades

#pragma omp parallel for
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                float sum = 0.0f;
                    for (int i = 0; i < size; ++i) {
                        for (int j = 0; j < size; ++j) {
                            if (i >= border && i <= height - border && j >= border && j <= width - border) {
                                sum += src[(y + i - border) * width + (x + j - border)] * gaussianMatrix[i * size + j];
                            } else {
                                sum += src[(y) * width + (x)] * gaussianMatrix[i * size + j];
                            }
                        }
                    }
                    float increm = (src[y * width + x] * 1.1);
                    float result = increm - sum;

                    // Verifica si la diferencia es mayor que el umbral para evitar el ruido horizontal
                    if (result < 0) result = 0;
                    if (result > 65535) result = 65535;

                    dst[y * width + x] = static_cast<unsigned short>(result);
            }
        }

        result = 0;

    } catch (...) {

        result = 999;
    }

    return result;
}

void smoothImageP(unsigned short* src, unsigned short* dst, int width, int height, int kernelSize)
{
    int halfKernel = kernelSize / 2;

#pragma omp parallel for
    for (int y = halfKernel; y < height - halfKernel; ++y) {
        for (int x = halfKernel; x < width - halfKernel; ++x) {
            int sum = 0;

            for (int i = -halfKernel; i <= halfKernel; ++i) {
                for (int j = -halfKernel; j <= halfKernel; ++j) {
                    sum += src[(y + i) * width + (x + j)];
                }
            }

            dst[y * width + x] = sum / (kernelSize * kernelSize);
        }
    }
}


void edgeDetectionP(unsigned short* src, unsigned short* dst, int width, int height, float edgeScale, int gradientThreshold)
{
    // Kernels de Sobel para detección de bordes en dirección X e Y
    std::vector<int> kernelX = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };
    std::vector<int> kernelY = { -1, -2, -1, 0, 0, 0, 1, 2, 1 };
    const int kernelSize = 3;
    const int border = kernelSize / 2;
    int* iDst = new int[width * height];

    // Aplica el filtro de Sobel para detección de bordes en paralelo
#pragma omp parallel for
    for (int y = border; y < height - border; ++y) {
        for (int x = border; x < width - border; ++x) {
            int sumX = 0;
            int sumY = 0;

            // Calcula las sumas ponderadas en las direcciones horizontal y vertical
            for (int ky = 0; ky < kernelSize; ++ky) {
                for (int kx = 0; kx < kernelSize; ++kx) {
                    int index = (y + ky - border) * width + (x + kx - border);
                    sumX += kernelX[ky * kernelSize + kx] * src[index];
                    sumY += kernelY[ky * kernelSize + kx] * src[index];
                }
            }

            // Calcula la magnitud del gradiente utilizando la fórmula de la norma euclidiana
            int gradient = static_cast<int>(std::sqrt(static_cast<float>(sumX * sumX + sumY * sumY)));

            // Asegurarse de que el valor está dentro del rango 0-65535
            gradient = std::max(gradient, 0);
            gradient = std::min(gradient, 65535);

            // Aplica el factor de escala al valor del gradiente
            gradient = static_cast<int>(gradient * edgeScale);

            // Asegurarse de que el valor escalado está dentro del rango 0-65535
            gradient = std::max(gradient, 0);
            gradient = std::min(gradient, 65535);

            // Aplica el umbral a la magnitud del gradiente para resaltar solo los bordes
            if (gradient >= gradientThreshold) {
                iDst[y * width + x] = (src[y * width + x] * 2) + static_cast<unsigned short>(gradient);
            } else {
                iDst[y * width + x] = (src[y * width + x] * 2);
            }
        }
    }

    adjustToRangeP(iDst, dst, width * height);
    delete[] iDst;
}


int processingAutoP(unsigned short* src, unsigned short* dst, int width, int height, float contrast, int smooth, float edgeScale,
                    int gradientThreshold, float lowThreshold, float lowContrastBoost)
{
    int result;
    if (contrast < 0) contrast = 0;
    if (contrast > 5.0) contrast = 5.0;
    if (smooth < 0) smooth = 0;
    if (smooth > 1) smooth = 1;
    if (edgeScale > 1.0) edgeScale = 1.0;
    if (edgeScale < 0.0) edgeScale = 0.0;
    if (gradientThreshold > 10000) gradientThreshold = 10000;
    if (gradientThreshold < 0) gradientThreshold = 0;
    if (lowThreshold > 10000.0) lowThreshold = 10000.0;
    if (lowThreshold < 0.0) lowThreshold = 0.0;
    if (lowContrastBoost > 5.0) lowContrastBoost = 5.0;
    if (lowContrastBoost < 0.0) lowContrastBoost = 0.0;

    unsigned short* tempDst = new unsigned short[width * height];

    try {
        boostLowContrastP(src, tempDst, width, height, lowThreshold, lowContrastBoost);

        for (int i = 0; i < width * height; ++i) {
            src[i] = tempDst[i];
        }
        result = 3;

        edgeDetectionP(src, tempDst, width, height, edgeScale, gradientThreshold);

        for (int i = 0; i < width * height; ++i) {
            src[i] = tempDst[i];
        }
        result = 2;

        smoothImageP(src, tempDst, width, height, smooth);

        for (int i = 0; i < width * height; ++i) {
            src[i] = tempDst[i];
        }
        result = 1;

        backgroundSubtractionP(src, dst, width, height, contrast);
        result = 0;
    } catch (...) {
        result = 999;
        // En caso de excepción, asegurarse de liberar la memoria asignada
        delete[] tempDst;
        throw; // Propagar la excepción nuevamente
    }

    delete[] tempDst;
    return result;
}
