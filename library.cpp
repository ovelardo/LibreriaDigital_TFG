#include <iostream>
#include <vector>
#include "library.h"
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

//////Funcion de prueba
void hello() {
    std::cout << "Hello, World!" << std::endl;
}

//////Funciones de cargado y guardado de imagen RaW 16 bit

bool loadRawImage(const char *filePath, unsigned short* image, int width, int height)
{
    // Convertir el filePath a std::string
    std::string filePathStr(filePath);

    // Abrir el archivo binario en modo de lectura
    std::ifstream file(filePathStr, std::ios::binary);
    if (!file.is_open()) {
        std::cout << "No se pudo abrir el archivo: " << filePath << std::endl;
        return false;
    }

    // Comprobar el tamaño del archivo
    file.seekg(0, std::ios::end);
    std::streampos fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    // Comprobar si el tamaño del archivo coincide con el tamaño de la imagen esperada
    if (fileSize != static_cast<std::streamoff>(width * height * sizeof(unsigned short))) {
        std::cout << "Tamaño del archivo incorrecto." << std::endl;
        return false;
    }

    // Leer los datos del archivo en el arreglo de imagen
    file.read(reinterpret_cast<char*>(image), fileSize);

    // Comprobar si ocurrieron errores durante la lectura
    if (!file) {
        std::cout << "Error de lectura del archivo." << std::endl;
        return false;
    }

    // Convertir los bytes leídos a valores unsigned short
    for (int i = 0; i < width * height; i++) {
        unsigned char byte1 = static_cast<unsigned char>(image[i] & 0xFF);
        unsigned char byte2 = static_cast<unsigned char>((image[i] >> 8) & 0xFF);
        image[i] = (byte2 << 8) | byte1;
    }

    // Cerrar el archivo
    file.close();

    return true;
}


bool saveRawImage(const char* filePath, const unsigned short* image, int width, int height)
{
    // Abrir el archivo binario en modo de escritura
    std::ofstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        std::cout << "No se pudo abrir el archivo para escribir: " << filePath << std::endl;
        return false;
    }

    // Escribir los datos de la imagen en el archivo
    file.write(reinterpret_cast<const char*>(image), width * height * sizeof(unsigned short));

    // Comprobar si ocurrieron errores durante la escritura
    if (!file) {
        std::cout << "Error de escritura en el archivo." << std::endl;
        return false;
    }

    // Cerrar el archivo
    file.close();

    return true;
}



//////Funciones en secuencial

// Función para rotar una imagen de 16 bits
// direction = 1 para rotación de 90 grados en sentido de las manecillas del reloj
// direction = 2 para rotación de 180 grados
// direction = 3 para rotación de 270 grados en sentido contrario a las manecillas del reloj
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
                dst[(width - j - 1) * height + i] = src[i * width + j];
            }
        }
    }
}

// Función para voltear una imagen de 16 bits
// direction = 1 para voltear horizontalmente
// direction = 2 para voltear verticalmente
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

void adjustContrast2(unsigned short* src, unsigned short* dst, int rows, int cols, float contrastLevel)
{
    float factor = (65535.0f * (contrastLevel + 255.0f)) / (255.0f * (65535.0f - contrastLevel));
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            float val = static_cast<float>(src[i * cols + j]);
            val = val * factor - 32768.0f * factor + 32768.0f;
            if (val < 0.0f) {
                val = 0.0f;
            } else if (val > 65535.0f) {
                val = 65535.0f;
            }
            dst[i * cols + j] = static_cast<unsigned short>(val);
        }
    }
}

void adjustContrast(unsigned short* src, unsigned short* dst, int rows, int cols, float contrastLevel, float amplificationFactor)
{
    float gaussFactor = std::exp(-0.5f * std::pow((contrastLevel - 128.0f) / 128.0f, 2));

    unsigned short minVal = std::numeric_limits<unsigned short>::max();
    unsigned short maxVal = std::numeric_limits<unsigned short>::min();

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
            adjustedValue = std::max(adjustedValue, -131070.0f);
            adjustedValue = std::min(adjustedValue, 327670.0f);

            // Mapeamos los valores al rango 0-65535
            adjustedValue = (adjustedValue + 131070.0f) * (65535.0f / 458740.0f);

            dst[i * cols + j] = static_cast<unsigned short>(adjustedValue);
        }
    }
}


void edgeDetection(unsigned short* src, unsigned short* dst, int rows, int cols, unsigned short threshold, int operation)
{
    int kernelX[3][3] = { {-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1} };
    int kernelY[3][3] = { {-1, -2, -1}, {0, 0, 0}, {1, 2, 1} };

    // Aplicar filtros de Sobel en X e Y
    for (int i = 1; i < rows - 1; i++) {
        for (int j = 1; j < cols - 1; j++) {
            int sumX = 0;
            int sumY = 0;

            // Convolución en X e Y
            for (int k = -1; k <= 1; k++) {
                for (int l = -1; l <= 1; l++) {
                    int pixel = src[(i + k) * cols + (j + l)];
                    sumX += kernelX[k + 1][l + 1] * pixel;
                    sumY += kernelY[k + 1][l + 1] * pixel;
                }
            }

            // Calcular magnitud aproximada del gradiente
            int magnitude = static_cast<unsigned short>(std::abs(sumX) + std::abs(sumY));

            // Operación de suma o resta
            int result;
            if (operation == 1) {
                result = magnitude + static_cast<int>(src[i * cols + j]);
            } else {
                result = magnitude - static_cast<int>(src[i * cols + j]);
            }

            // Ajustamos los valores al rango ampliado
            result = std::max(result, -131070);
            result = std::min(result, 327670);

            // Mapeamos los valores al rango 0-65535
            result = (result + 131070) * (65535 / 458740);

            dst[i * cols + j] = static_cast<unsigned short>(result);
        }
    }
}










void perfilado(unsigned short* src, unsigned short* dst, int width, int height, float threshold, int amplificationFactor)
{
    std::vector<int> kernel = { 0, amplificationFactor, 0, amplificationFactor, -5 * amplificationFactor, amplificationFactor, 0, amplificationFactor, 0 };
    const int kernel_size = 3;
    const int border = kernel_size / 2;
    const float norm_factor = threshold / 65535.0f;

    // Iterar sobre los píxeles de la imagen
    for (int i = border; i < height - border; ++i)
    {
        for (int j = border; j < width - border; ++j)
        {
            int sum = 0;
            // Iterar sobre el kernel
            for (int k = 0; k < kernel_size; ++k)
            {
                for (int l = 0; l < kernel_size; ++l)
                {
                    sum += kernel[k * kernel_size + l] * src[(i - border + k) * width + (j - border + l)];
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

#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

void contrastEnhancement(unsigned short* src, unsigned short* dst, int width, int height, float contrastLevel)
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
    contrastEnhancement(srcImage, dstImage, width, height, contrastLevel);

    // Realizar otras operaciones con la imagen mejorada

    delete[] srcImage;
    delete[] dstImage;

    return 0;
}






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
