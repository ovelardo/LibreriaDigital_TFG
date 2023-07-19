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
                dst[j * height + i] = src[i * width + (width - j - 1)];
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

void highContrast(unsigned short* src, unsigned short* dst, int width, int height, float threshold, float contrastBoost)
{
    std::cout << "Iniciamos funcion " << std::endl;

    unsigned short minVal = std::numeric_limits<unsigned short>::max();
    unsigned short maxVal = std::numeric_limits<unsigned short>::min();

    // Calcula el valor mínimo y máximo en la imagen
    for (int i = 0; i < width * height; ++i) {
        if (src[i] < minVal) {
            minVal = src[i];
        }
        if (src[i] > maxVal) {
            maxVal = src[i];
        }
    }

    unsigned short highThreshold = minVal + (unsigned short)((maxVal - minVal) * threshold);

    // Aplica el contraste proporcionalmente a los valores altos
    for (int i = 0; i < width * height; ++i) {
        if (src[i] > highThreshold) {
            float val = static_cast<float>(src[i]);

            // Calcula el factor de contraste en función del valor máximo
            float factor = contrastBoost * (65535.0f - static_cast<float>(src[i])) / (static_cast<float>(maxVal) - static_cast<float>(highThreshold));

            // Ajusta el valor proporcionalmente sin exceder el límite de 65535
            val = val + factor;
            // Ajustamos los valores al rango ampliado
            val = std::max(val, -131070.0f);
            val = std::min(val, 327670.0f);

            // Mapeamos los valores al rango 0-65535
            val = (val + 131070.0f) * (65535.0f / 458740.0f);
            dst[i] = static_cast<unsigned short>(val);
        } else {
            dst[i] = src[i];
        }
    }
}

void highPassContrast(unsigned short* src, unsigned short* dst, int width, int height, float contrastBoost)
{
    std::vector<int> kernel = { 0, -1, 0, -1, 4, -1, 0, -1, 0 };
    const int kernelSize = 3;
    const int border = kernelSize / 2;

    // Aplica el filtro de paso alto utilizando el operador Laplaciano
    for (int y = border; y < height - border; ++y) {
        for (int x = border; x < width - border; ++x) {
            int sum = 0;

            // Calcula la suma ponderada de los vecinos utilizando el kernel
            for (int ky = 0; ky < kernelSize; ++ky) {
                for (int kx = 0; kx < kernelSize; ++kx) {
                    int index = (y + ky - border) * width + (x + kx - border);
                    sum += kernel[ky * kernelSize + kx] * src[index];
                }
            }

            // Aplica el contraste proporcionalmente al valor alto
            float result = src[y * width + x] + static_cast<float>(contrastBoost * sum);

            // Ajustamos los valores al rango ampliado
            result = std::max(result, -131070.0f);
            result = std::min(result, 327670.0f);

            // Mapeamos los valores al rango 0-65535
            result = (result + 131070.0f) * (65535.0f / 458740.0f);

            //Sumamos la imagen inicial y volvemos a mapear
            result = src[y * width + x] + src[y * width + x] + result;
            result = (result + 131070.0f) * (65535.0f / 458740.0f);

            dst[y * width + x] = static_cast<unsigned short>(result);
        }
    }
}

void highPassContrast2(unsigned short* src, unsigned short* dst, int width, int height, float threshold, float contrastBoost)
{
    std::vector<int> kernel = { 0, -1, 0, -1, 4, -1, 0, -1, 0 };
    const int kernelSize = 3;
    const int border = kernelSize / 2;

    // Aplica el filtro de paso alto utilizando el operador Laplaciano
    for (int y = border; y < height - border; ++y) {
        for (int x = border; x < width - border; ++x) {
            int sum = 0;

            // Calcula la suma ponderada de los vecinos utilizando el kernel
            for (int ky = 0; ky < kernelSize; ++ky) {
                for (int kx = 0; kx < kernelSize; ++kx) {
                    int index = (y + ky - border) * width + (x + kx - border);
                    sum += kernel[ky * kernelSize + kx] * src[index];
                }
            }

            // Aplica el contraste proporcionalmente al valor alto
            float result = src[y * width + x] + static_cast<float>(contrastBoost * sum);

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

void highPassContrast3(unsigned short* src, unsigned short* dst, int width, int height, float threshold, float contrastBoost)
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

    // Aplicamos el aumento de contraste a la zona dentro del threshold
    for (int i = 0; i < width * height; ++i) {
        if (src[i] <= threshold) {
            // Aplicamos el contraste a los valores por debajo del threshold
            float val = static_cast<float>(src[i]);
            val = val * contrastBoost;
            dst[i] = static_cast<unsigned short>(val);
            maxValInThreshold = std::max(maxValInThreshold, static_cast<float>(dst[i]));
        }
    }

    // Ajustamos los valores por encima del threshold de manera lineal
    for (int i = 0; i < width * height; ++i) {
        if (src[i] > threshold) {
            // Ajustamos los valores linealmente
            float val = src[i];
            val = maxValInThreshold + (val - threshold);
            dst[i] = val;
        }
    }

    adjustToRange(dst, width * height);
}

void adjustToRange(unsigned short* dst, int size)
{
    float minVal = std::numeric_limits<unsigned short>::max();
    float maxVal = std::numeric_limits<unsigned short>::min();

    // Encontramos el valor mínimo y máximo en la imagen
    for (int i = 0; i < size; ++i) {
        minVal = std::min(minVal, static_cast<float>(dst[i]));
        maxVal = std::max(maxVal, static_cast<float>(dst[i]));
    }

    // Ajustamos los valores al rango 0-65535
    float factor = 65535.0f / static_cast<float>(maxVal - minVal);

    for (int i = 0; i < size; ++i) {
        float val = static_cast<float>(dst[i] - minVal) * factor;
        dst[i] = static_cast<unsigned short>(val);
    }
}



















void amplifyLowValues(unsigned short* src, unsigned short* dst, int width, int height, float threshold, float amplificationFactor)
{
    // Calcula el valor máximo en la imagen
    unsigned short maxVal = std::numeric_limits<unsigned short>::min();
    for (int i = 0; i < width * height; ++i) {
        if (src[i] > maxVal) {
            maxVal = src[i];
        }
    }

    unsigned short lowThreshold = static_cast<unsigned short>(maxVal * threshold);

    // Amplifica los valores bajos en función del factor de amplificación
    for (int i = 0; i < width * height; ++i) {
        if (src[i] < lowThreshold) {
            float val = static_cast<float>(src[i]);
            val = val * amplificationFactor;

            // Ajusta el valor proporcionalmente sin exceder el límite de 65535
            val = std::max(val, 0.0f);
            val = std::min(val, 65535.0f);
            dst[i] = static_cast<unsigned short>(val);
        } else {
            dst[i] = src[i];
        }
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




void perfilado1(unsigned short* src, unsigned short* dst, int width, int height, float threshold, int amplificationFactor)
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
