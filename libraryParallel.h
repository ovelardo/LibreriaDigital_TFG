//
// Created by ovelardo on 20/07/23.
//

#ifndef LIBRERIADIGITAL_LIBRARYPARALLEL_H
#define LIBRERIADIGITAL_LIBRARYPARALLEL_H

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

LIBRARY_API void rotateP(unsigned short* src, unsigned short* dst, int width, int height, int direction);
LIBRARY_API void flipP(unsigned short* src, unsigned short* dst, int width, int height, int direction);
LIBRARY_API void adjustContrastP(unsigned short* src, unsigned short* dst, int rows, int cols, float contrastLevel, float amplificationFactor);
LIBRARY_API void highPassContrastP(unsigned short* src, unsigned short* dst, int width, int height, float threshold, float contrastBoost);
LIBRARY_API void adjustBrightnessP(unsigned short* src, unsigned short* dst, int width, int height, float contrastLevel);
LIBRARY_API void edgeIncreaseP(unsigned short* src, unsigned short* dst, int width, int height, float threshold, int amplificationFactor);
LIBRARY_API void adjustToRangeP(int* iDst, unsigned short* dst, int size);
LIBRARY_API void boostLowContrastP(unsigned short* src, unsigned short* dst, int width, int height, float threshold, float contrastBoost);
LIBRARY_API int backgroundSubtractionP(unsigned short* src, unsigned short* dst, int width, int height, float sigma);
LIBRARY_API void smoothImageP(unsigned short* src, unsigned short* dst, int width, int height, int kernelSize);
LIBRARY_API void edgeDetectionP(unsigned short* src, unsigned short* dst, int width, int height, float edgeScale, int gradientThreshold);
LIBRARY_API void sharpnessImageP(unsigned short* src, unsigned short* dst, int width, int height, float strength);
LIBRARY_API int processingAutoP(unsigned short* src, unsigned short* dst, int width, int height, float contrast = 1.0, int smooth = 1, float edgeScale = 0.1,
                                int gradientThreshold = 1000, float lowThreshold = 1000.0, float lowContrastBoost = 1.0);

#ifdef __cplusplus
}
#endif

#endif // LIBRERIADIGITAL_LIBRARYPARALLEL_H