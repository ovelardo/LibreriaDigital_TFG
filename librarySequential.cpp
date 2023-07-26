//
// Created by ovelardo on 02/06/23.
// We define some functions to apply to raw 16 bit image in order to change or modify it
//

#include <vector>
#include "librarySequential.h"
#include <mutex>
#include <cmath>

#ifdef _WIN32
// Windows platform
    #define LIBRARY_API __declspec(dllexport)
#else
// Non-Windows platforms
    #define LIBRARY_API
#endif

using namespace std;


//////Sequential functions

// Function to rotate 16 bit image
// direction = 1 -- 90º rotation clockwise
// direction = 2 -- 180º rotation clockwise
// direction = 3 -- 270º rotation clockwise
void rotate(unsigned short* src, unsigned short* dst, int width, int height, int direction)
{
    if (direction == 1) {
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                dst[j * height + (height - i - 1)] = src[i * width + j];
            }
        }
    } else if (direction == 2) {
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                dst[(height - i - 1) * width + (width - j - 1)] = src[i * width + j];
            }
        }
    } else if (direction == 3) {
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                dst[j * height + i] = src[i * width + (width - j - 1)];
            }
        }
    }
}


// Function to flip 16 bit image
// direction = 1 -- horizontal flip
// direction = 2 -- vertical flip
void flip(unsigned short* src, unsigned short* dst, int width, int height, int direction)
{
    if (direction == 1) {
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                dst[i * width + (width - j - 1)] = src[i * width + j];
            }
        }
    } else if (direction == 2) {
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                dst[(height - i - 1) * width + j] = src[i * width + j];
            }
        }
    }
}







void adjustContrast(unsigned short* src, unsigned short* dst, int rows, int cols, float contrastLevel, float amplificationFactor)
{
    float gaussFactor = std::exp(-0.5f * std::pow((contrastLevel - 128.0f) / 128.0f, 2));

    unsigned short minVal = std::numeric_limits<unsigned short>::max();
    unsigned short maxVal = std::numeric_limits<unsigned short>::min();

    int* iDst = new int[rows * cols];

    // Calculamos los valores mínimo y máximo de la imagen original
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

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            unsigned short val = src[i * cols + j];

            // Aplicamos el ajuste de contraste y amplificación
            float adjustedValue = (val * contrastScale * gaussFactor * amplificationScale) - adjustmentOffset;

            // Ajustamos los valores al rango ampliado
            //adjustedValue = std::max(adjustedValue, -131070.0f);
            //adjustedValue = std::min(adjustedValue, 327670.0f);

            // Mapeamos los valores al rango 0-65535
            //adjustedValue = (adjustedValue + 131070.0f) * (65535.0f / 458740.0f);

            //dst[i * cols + j] = static_cast<unsigned short>(adjustedValue);
            iDst[i * cols + j] = static_cast<int>(adjustedValue);
        }
    }

    adjustToRange(iDst, dst, rows * cols);
}

void highPassContrast(unsigned short* src, unsigned short* dst, int width, int height, float threshold, float contrastBoost)
{
    std::vector<int> kernelX = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };
    std::vector<int> kernelY = { -1, -2, -1, 0, 0, 0, 1, 2, 1 };
    const int kernelSize = 3;
    const int border = kernelSize / 2;

    // Aplica el filtro de paso alto utilizando el kernel de Sobel
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



void deteccionBordes(unsigned short* src, unsigned short* dst, int width, int height, float threshold, int amplificationFactor)
{
    std::vector<int> kernelX = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };
    std::vector<int> kernelY = { -1, -2, -1, 0, 0, 0, 1, 2, 1 };
    const int kernel_size = 3;
    const int border = kernel_size / 2;
    const float norm_factor = threshold / 65535.0f;

    // Iterar sobre los píxeles de la imagen
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

void boostLowContrast(unsigned short* src, unsigned short* dst, int width, int height, float threshold, float contrastBoost)
{
    float maxValInThreshold = 0;
    int* iDst = new int[width * height];

    // Aplicamos el aumento de contraste a la zona dentro del threshold
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
    for (int i = 0; i < width * height; ++i) {
        if (src[i] > threshold) {
            // Ajustamos los valores linealmente
            float val = src[i];
            val = maxValInThreshold + (val - threshold);
            iDst[i] = val;
        }
    }

    adjustToRange(iDst, dst, width * height);
}


void adjustToRange(int* iDst, unsigned short* dst, int size)
{
    int minVal = std::numeric_limits<int>::max();
    int maxVal = std::numeric_limits<int>::min();

    // Encontramos el valor mínimo y máximo en la imagen
    for (int i = 0; i < size; ++i) {
        minVal = std::min(minVal, static_cast<int>(iDst[i]));
        maxVal = std::max(maxVal, static_cast<int>(iDst[i]));
    }

    // Ajustamos los valores al rango 0-65535
    float factor = 65535.0f / static_cast<float>(maxVal - minVal);

    for (int i = 0; i < size; ++i) {
        float val = static_cast<float>(iDst[i] - minVal) * factor;
        dst[i] = static_cast<unsigned short>(val);
    }
}


void perfilado(unsigned short* src, unsigned short* dst, int width, int height, float threshold, int amplificationFactor, int kernelSize)
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

    // Iterar sobre los píxeles de la imagen
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

void adjustBrightness(unsigned short* src, unsigned short* dst, int width, int height, float contrastLevel)
{
    // Calcular el histograma
    std::vector<int> histogram(65536, 0);
    for (int i = 0; i < width * height; ++i) {
        ++histogram[src[i]];
    }

    // Calcular el número total de píxeles
    //int totalPixels = width * height;

    // Calcular el número acumulativo de píxeles en el histograma
    std::vector<int> cumulativeHistogram(65536, 0);
    cumulativeHistogram[0] = histogram[0];
    for (int i = 1; i < 65536; ++i) {
        cumulativeHistogram[i] = cumulativeHistogram[i - 1] + histogram[i];
    }

    // Calcular el valor mínimo y máximo del histograma
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

    // Calcular el valor objetivo de cada nivel de gris
    std::vector<unsigned short> mappingTable(65536, 0);
    for (int i = 0; i < 65536; ++i) {
        float normalizedValue = (i - minValue) / dynamicRange;
        float contrastAdjustedValue = std::pow(normalizedValue, contrastLevel);
        unsigned short mappedValue = static_cast<unsigned short>(contrastAdjustedValue * dynamicRange + minValue);
        mappingTable[i] = std::max(std::min(mappedValue, static_cast<unsigned short>(65535)), static_cast<unsigned short>(0));
    }

    // Aplicar la transformación de contraste a la imagen
    for (int i = 0; i < width * height; ++i) {
        dst[i] = mappingTable[src[i]];
    }
}

#include <cmath>

#include <cmath>

#include <cmath>

void backgroundSubtraction(unsigned short* src, unsigned short* dst, int width, int height, float sigma)
{
    // Calcula el tamaño de la matriz gaussiana
    int size = 2 * static_cast<int>(std::ceil(3 * sigma)) + 1;

    // Calcula la matriz gaussiana
    std::vector<float> gaussianMatrix(size * size);
    float sum = 0.0f;
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            float x = i - size / 2;
            float y = j - size / 2;
            float value = std::exp(-(x * x + y * y) / (2 * sigma * sigma));
            gaussianMatrix[i * size + j] = value;
            sum += value;
        }
    }

    // Normaliza la matriz gaussiana para que sume 1
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            gaussianMatrix[i * size + j] /= sum;
        }
    }

    // Realiza la convolución de la imagen con la matriz gaussiana
    int border = size / 2;
    //float threshold = 1000.0f; // Puedes ajustar este umbral según tus necesidades

    for (int y = border; y < height - border; ++y) {
        for (int x = border; x < width - border; ++x) {
            float sum = 0.0f;
            for (int i = 0; i < size; ++i) {
                for (int j = 0; j < size; ++j) {
                    sum += src[(y + i - border) * width + (x + j - border)] * gaussianMatrix[i * size + j];
                }
            }
            float increm = (src[y * width + x] * 1.1);
            float result = increm - sum;

            // Verifica si la diferencia es mayor que el umbral para evitar el ruido horizontal
            //if (std::abs(result - increm) > threshold) {
            if (result < 0) result=0;
            if (result > 65535) result = 65535;
                dst[y * width + x] = static_cast<unsigned short>(result);
            //} else {
            //    dst[y * width + x] = (src[y * width + x] * 1.1);
            //}
        }
    }
}






int main()
{
    // Ejemplo de uso
    const int width = 640;
    const int height = 480;
    unsigned short* srcImage = new unsigned short[width * height];
    unsigned short* dstImage = new unsigned short[width * height];

    // Rellenar la imagen de ejemplo con valores

    // Aplicar la mejora de contraste
    float contrastLevel = 1.5f;
    adjustBrightness(srcImage, dstImage, width, height, contrastLevel);

    // Realizar otras operaciones con la imagen mejorada

    delete[] srcImage;
    delete[] dstImage;

    return 0;
}






