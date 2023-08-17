//
// Created by ovelardo on 02/06/23.
// Digital Proccesing functions (sequential)
// We define functions to apply to raw 16 bit image to proccess it
// We use as standard parameters in every function:
// Origin image as src that is a unsigned short*
// Destiny image as dst that is a unsigned short*
// Width image as width that is an int
// Height image as height that is an int
// Additional parameters are explained in every function

#include <vector>
#include "librarySequential.h"
#include <mutex>
#include <cmath>
#include <cstring>

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
    } else {
        memcpy(dst, src, width * height * sizeof(unsigned short));
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
    }  else {
        memcpy(dst, src, width * height * sizeof(unsigned short));
    }
}

// Function to adjust contrast histogram. Move contrast right or left or increase/decrease size
// contrastLevel: float type, moves contrast right or left (0.0f to 65535.0f)
// amplificationFactor: float type, increase/decrease histogram size (0.0f to 10.0f)
// Keep in mind that too higher values move image out of histogram
void adjustContrast(unsigned short* src, unsigned short* dst, int width, int height, float contrastLevel, float amplificationFactor)
{
    float gaussFactor = std::exp(-0.5f * std::pow((contrastLevel - 128.0f) / 128.0f, 2));

    unsigned short minVal = std::numeric_limits<unsigned short>::max();
    unsigned short maxVal = std::numeric_limits<unsigned short>::min();

    int* iDst = new int[width * height];

    // Min and Max calculation of original image
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            unsigned short val = src[i * height + j];
            minVal = std::min(minVal, val);
            maxVal = std::max(maxVal, val);
        }
    }

    // Range calculation of image values
    float range = maxVal - minVal;

    // Adjust contrast and amplification values
    float contrastScale = 65535.0f / range;
    float amplificationScale = amplificationFactor * contrastScale;

    // Adjust to avoid negative values
    float adjustmentOffset = minVal * contrastScale * gaussFactor * amplificationScale;

    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            unsigned short val = src[i * height + j];

            // Apply contrast and amplification
            float adjustedValue = (val * contrastScale * gaussFactor * amplificationScale) - adjustmentOffset;

            // Adjust values to range
            //adjustedValue = std::max(adjustedValue, -131070.0f);
            //adjustedValue = std::min(adjustedValue, 327670.0f);

            // Map values to range 0-65535
            //adjustedValue = (adjustedValue + 131070.0f) * (65535.0f / 458740.0f);

            iDst[i * height + j] = static_cast<int>(adjustedValue);
        }
    }

    adjustToRange(iDst, dst, width * height);
}


// Function to contrast high values. Create an emboss effect, but it is too strong. Only for use with images too smooth
// threshold: float type, low range to apply contrastboost (0.0f to 10.0f)
// contrastBoost: float type, increase boost (0.0f to 3.0f)
// Keep in mind that too higher values move image out of histogram
void highPassContrast(unsigned short* src, unsigned short* dst, int width, int height, float threshold, float contrastBoost)
{
    std::vector<int> kernelX = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };
    std::vector<int> kernelY = { -1, -2, -1, 0, 0, 0, 1, 2, 1 };
    const int kernelSize = 3;
    const int border = kernelSize / 2;

    // Apply high pass filter using Sobel kernel
    for (int y = border; y < height - border; ++y) {
        for (int x = border; x < width - border; ++x) {
            int sumX = 0;
            int sumY = 0;

            // Calculates the weighted sums in the horizontal and vertical directions
            for (int ky = 0; ky < kernelSize; ++ky) {
                for (int kx = 0; kx < kernelSize; ++kx) {
                    int index = (y + ky - border) * width + (x + kx - border);
                    sumX += kernelX[ky * kernelSize + kx] * src[index];
                    sumY += kernelY[ky * kernelSize + kx] * src[index];
                }
            }

            // Calculate the magnitude of the gradient
            float gradient = std::sqrt(static_cast<float>(sumX * sumX + sumY * sumY));

            // Applies contrast proportionally to the high value
            float result = src[y * width + x] + static_cast<float>(contrastBoost * gradient);

            // Adjust the values to the extended range
            result = std::max(result, -131070.0f);
            result = std::min(result, 327670.0f);

            // Map values to the range 0-65535
            result = (result + 131070.0f) * (65535.0f / 458740.0f);

            // Apply threshold to highlight the edges
            if (std::abs(result - src[y * width + x]) > threshold) {
                dst[y * width + x] = static_cast<unsigned short>(result);
            } else {
                dst[y * width + x] = src[y * width + x];
            }
        }
    }
}


// Function to increase edges. Increase edges using kernel
// threshold: float type, range to normalize (0.0f to 65535.0f)
// amplificationFactor: int type, amplification value (0 to 10)
// Keep in mind that too higher values move image out of histogram
void edgeIncrease(unsigned short* src, unsigned short* dst, int width, int height, float threshold, int amplificationFactor)
{
    std::vector<int> kernelX = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };
    std::vector<int> kernelY = { -1, -2, -1, 0, 0, 0, 1, 2, 1 };
    const int kernel_size = 3;
    const int border = kernel_size / 2;
    const float norm_factor = threshold / 65535.0f;

    // Iterate over the image pixels
    for (int i = border; i < height - border; ++i)
    {
        for (int j = border; j < width - border; ++j)
        {
            int sumX = 0;
            int sumY = 0;
            // Iterate over the kernel in X
            for (int k = 0; k < kernel_size; ++k)
            {
                for (int l = 0; l < kernel_size; ++l)
                {
                    sumX += kernelX[k * kernel_size + l] * src[(i - border + k) * width + (j - border + l)];
                    sumY += kernelY[k * kernel_size + l] * src[(i - border + k) * width + (j - border + l)];
                }
            }
            // Calculate the magnitude of the gradient
            int magnitude = std::abs(sumX) + std::abs(sumY);
            // Apply threshold and amplification
            int result = static_cast<int>(src[i * width + j]) + static_cast<int>(norm_factor * magnitude * amplificationFactor);
            // Fit the result to the range 0-65535
            result = std::max(result, 0);
            result = std::min(result, 65535);
            dst[i * width + j] = static_cast<unsigned short>(result);
        }
    }
}

// Function to increase contrast in low range.
// threshold: float type, range to normalize (0.0f to 65535.0f)
// contrastBoost: float type, amplification value (0.0f to 5.0f)
// Range adjusted to avoid values out of range
void boostLowContrast(unsigned short* src, unsigned short* dst, int width, int height, float threshold, float contrastBoost)
{
    float maxValInThreshold = 0;
    int* iDst = new int[width * height];

    // Apply the contrast increase to the zone inside the threshold
    for (int i = 0; i < width * height; ++i) {
        if (src[i] <= threshold) {
            // Apply the contrast to the values below the threshold
            float val = static_cast<float>(src[i]);
            val = val * contrastBoost;
            iDst[i] = static_cast<int>(val);
            maxValInThreshold = std::max(maxValInThreshold, static_cast<float>(iDst[i]));
        }
    }

    // Adjust the values above the threshold linearly
    for (int i = 0; i < width * height; ++i) {
        if (src[i] > threshold) {
            // Adjust the values linearly
            float val = src[i];
            val = maxValInThreshold + (val - threshold);
            iDst[i] = val;
        }
    }

    // Adjust range to avoid loosing values
    adjustToRange(iDst, dst, width * height);
}

// Function to adjust values to range 0-65535.
// size: int type, width * height
void adjustToRange(int* iDst, unsigned short* dst, int size)
{
    int minVal = std::numeric_limits<int>::max();
    int maxVal = std::numeric_limits<int>::min();

    // Find the minimum and maximum value in the image
    for (int i = 0; i < size; ++i) {
        minVal = std::min(minVal, static_cast<int>(iDst[i]));
        maxVal = std::max(maxVal, static_cast<int>(iDst[i]));
    }

    // We adjust the values to the range 0-65535
    float factor = 65535.0f / static_cast<float>(maxVal - minVal);

    for (int i = 0; i < size; ++i) {
        float val = static_cast<float>(iDst[i] - minVal) * factor;
        dst[i] = static_cast<unsigned short>(val);
    }
}

// Function to sharpness image.
// strength: float type, value to strength sharpness (1.0f to 10.0f)
// Range adjusted to avoid values out of range
void sharpnessImage(unsigned short* src, unsigned short* dst, int width, int height, float strength)
{
    // High Pass Filter
    std::vector<int> kernel = { -1, -1, -1, -1, 8, -1, -1, -1, -1 };
    const int kernelSize = 3;
    const int border = kernelSize / 2;
    int* iDst = new int[width * height];

    // Iterate over the image pixels
    for (int i = border; i < height - border; ++i)
    {
        for (int j = border; j < width - border; ++j)
        {
            int sum = 0;
            // Iterate over kernel
            for (int l = 0; l < kernelSize; ++l)
            {
                for (int k = 0; k < kernelSize; ++k)
                {
                    sum += kernel[k * kernelSize + l] * src[(i - border + k) * width + (j - border + l)];
                }
            }
            // Apply sharpen filter
            int result = static_cast<int>(src[i * width + j]) + static_cast<int>(strength * sum);
            iDst[i * width + j] = result;
        }
    }

    // Adjust range to avoid loosing values
    adjustToRange(iDst, dst, width * height);
    delete[] iDst;

    // Smooth image to avoid noise
    smoothImage(dst, dst, width, height, 1);
}

// Function to adjust brightness in image.
// contrastLevel: float type, increase (> 1) / decrease (< 1) brightness (0.0f to 2.0f)
// Range adjusted to avoid values out of range
void adjustBrightness(unsigned short* src, unsigned short* dst, int width, int height, float contrastLevel)
{
    // Calculate histogram
    std::vector<int> histogram(65536, 0);
    for (int i = 0; i < width * height; ++i) {
        ++histogram[src[i]];
    }

    // Calculate the cumulative number of pixels in the histogram
    std::vector<int> cumulativeHistogram(65536, 0);
    cumulativeHistogram[0] = histogram[0];
    for (int i = 1; i < 65536; ++i) {
        cumulativeHistogram[i] = cumulativeHistogram[i - 1] + histogram[i];
    }

    // Calculate the minimum and maximum value of the histogram
    int minValue = 0;
    int maxValue = 65535;
    while (histogram[minValue] == 0) {
        ++minValue;
    }
    while (histogram[maxValue] == 0) {
        --maxValue;
    }

    // Calculate the dynamic range of the histogram
    float dynamicRange = maxValue - minValue;

    // Calculate the target value of each gray level
    std::vector<unsigned short> mappingTable(65536, 0);
    for (int i = 0; i < 65536; ++i) {
        float normalizedValue = (i - minValue) / dynamicRange;
        float contrastAdjustedValue = std::pow(normalizedValue, contrastLevel);
        unsigned short mappedValue = static_cast<unsigned short>(contrastAdjustedValue * dynamicRange + minValue);
        mappingTable[i] = std::max(std::min(mappedValue, static_cast<unsigned short>(65535)), static_cast<unsigned short>(0));
    }

    // Apply the contrast transformation to the image
    for (int i = 0; i < width * height; ++i) {
        dst[i] = mappingTable[src[i]];
    }
}

// Function to increase contrast in advanced mode. Substraction of gaussian image
// sigma: float type, value to increase contrast (0.0f to 10.0f)
// Range adjusted to avoid values out of range
void backgroundSubtraction(unsigned short* src, unsigned short* dst, int width, int height, float sigma)
{
    // Calculate the size of the Gaussian matrix
    int size = 2 * static_cast<int>(std::ceil(3 * sigma)) + 1;

    // Calculate the Gaussian matrix
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

    // Normalize the Gaussian matrix so that it sums to 1
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            gaussianMatrix[i * size + j] /= sum;
        }
    }

    // Convolve the image with the Gaussian matrix
    int border = size / 2;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float sum = 0.0f;
            for (int i = 0; i < size; ++i) {
                for (int j = 0; j < size; ++j) {
                    sum += src[(y + i - border) * width + (x + j - border)] * gaussianMatrix[i * size + j];
                }
            }
            float increm = (src[y * width + x] * 1.1);
            float result = increm - sum;

            // Verifica si la diferencia es mayor que el umbral para evitar el ruido horizontal
            if (result < 0) result=0;
            if (result > 65535) result = 65535;
                dst[y * width + x] = static_cast<unsigned short>(result);
        }
    }
}

// Function to decrease noise in image. Using kernel media
// Parallel using OpenMP
// kernelSize: int type, 0 no reduction, 1 reduction
// Range adjusted to avoid values out of range
void smoothImage(unsigned short* src, unsigned short* dst, int width, int height, int kernelSize)
{
    int halfKernel = kernelSize / 2;

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

// Function to detect edges in image. Using kernel Sobel
// Parallel using OpenMP
// edgeScale: float type, power of edges (0.0f to 1.0f)
// gradientThreshold: int type, gradient value (1 to 5000)
// Range adjusted to avoid values out of range
void edgeDetection(unsigned short* src, unsigned short* dst, int width, int height, float edgeScale, int gradientThreshold)
{
    // Sobel kernels for edge detection in X and Y direction
    std::vector<int> kernelX = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };
    std::vector<int> kernelY = { -1, -2, -1, 0, 0, 0, 1, 2, 1 };
    const int kernelSize = 3;
    const int border = kernelSize / 2;
    int* iDst = new int[width * height];

    // Apply the Sobel filter for edge detection
    for (int y = border; y < height - border; ++y) {
        for (int x = border; x < width - border; ++x) {
            int sumX = 0;
            int sumY = 0;

            // Calculate the weighted sums in the horizontal and vertical directions
            for (int ky = 0; ky < kernelSize; ++ky) {
                for (int kx = 0; kx < kernelSize; ++kx) {
                    int index = (y + ky - border) * width + (x + kx - border);
                    sumX += kernelX[ky * kernelSize + kx] * src[index];
                    sumY += kernelY[ky * kernelSize + kx] * src[index];
                }
            }

            // Calculate the magnitude of the gradient using the Euclidean norm formula
            int gradient = static_cast<int>(std::sqrt(static_cast<float>(sumX * sumX + sumY * sumY)));

            // Make sure the value is in the range 0-65535
            gradient = std::max(gradient, 0);
            gradient = std::min(gradient, 65535);

            // Applies the scale factor to the gradient value
            gradient = static_cast<int>(gradient * edgeScale);

            // Make sure the scaled value is within the range 0-65535
            gradient = std::max(gradient, 0);
            gradient = std::min(gradient, 65535);

            // Applies the threshold to the magnitude of the gradient to highlight only the edges
            if (gradient >= gradientThreshold) {
                iDst[y * width + x] = (src[y * width + x]*2) + static_cast<unsigned short>(gradient);
            } else {
                iDst[y * width + x] = (src[y * width + x]*2);
            }
        }
    }

    // Adjust range to avoid loosing values
    adjustToRange(iDst, dst, width * height);
}










