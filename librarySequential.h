//
// Created by ovelardo on 02/06/23.
//

#ifndef LIBRERIADIGITAL_LIBRARYSEQUENTIAL_H
#define LIBRERIADIGITAL_LIBRARYSEQUENTIAL_H

#ifdef _WIN32
#ifdef LIBRARY_EXPORTS
        #define LIBRARY_API __declspec(dllexport)
    #else
        #define LIBRARY_API __declspec(dllimport)
    #endif
#else
#define LIBRARY_API
#endif

#ifdef __cplusplus
extern "C" {
#endif


LIBRARY_API void rotate(unsigned short* src, unsigned short* dst, int width, int height, int direction);
LIBRARY_API void flip(unsigned short* src, unsigned short* dst, int width, int height, int direction);
LIBRARY_API void adjustContrast(unsigned short* src, unsigned short* dst, int rows, int cols, float contrastLevel, float amplificationFactor);
LIBRARY_API void highPassContrast(unsigned short* src, unsigned short* dst, int width, int height, float threshold, float contrastBoost);
LIBRARY_API void adjustBrightness(unsigned short* src, unsigned short* dst, int width, int height, float contrastLevel);
LIBRARY_API void edgeIncrease(unsigned short* src, unsigned short* dst, int width, int height, float threshold, int amplificationFactor);
LIBRARY_API void adjustToRange(int* iDst, unsigned short* dst, int size);
LIBRARY_API void boostLowContrast(unsigned short* src, unsigned short* dst, int width, int height, float threshold, float contrastBoost);
LIBRARY_API void backgroundSubtraction(unsigned short* src, unsigned short* dst, int width, int height, float sigma);
LIBRARY_API void smoothImage(unsigned short* src, unsigned short* dst, int width, int height, int kernelSize);
LIBRARY_API void edgeDetection(unsigned short* src, unsigned short* dst, int width, int height, float edgeScale, int gradientThreshold);
LIBRARY_API void sharpnessImage(unsigned short* src, unsigned short* dst, int width, int height, float strength);


#ifdef __cplusplus
}
#endif

#endif // LIBRERIADIGITAL_LIBRARYSEQUENTIAL_H
