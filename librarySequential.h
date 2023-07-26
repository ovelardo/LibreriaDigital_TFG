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
LIBRARY_API void perfilado(unsigned short* src, unsigned short* dst, int width, int height, float threshold, int amplificationFactor,int kernel_size);
LIBRARY_API void adjustBrightness(unsigned short* src, unsigned short* dst, int width, int height, float contrastLevel);
LIBRARY_API void deteccionBordes(unsigned short* src, unsigned short* dst, int width, int height, float threshold, int amplificationFactor);
LIBRARY_API void boostLowContrast(unsigned short* src, unsigned short* dst, int width, int height, float threshold, float contrastBoost);
LIBRARY_API void adjustToRange(int* iDst, unsigned short* dst, int size);
LIBRARY_API void backgroundSubtraction(unsigned short* src, unsigned short* dst, int width, int height, float sigma);


#ifdef __cplusplus
}
#endif

#endif // LIBRERIADIGITAL_LIBRARYSEQUENTIAL_H
