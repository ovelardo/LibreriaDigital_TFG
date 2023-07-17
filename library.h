#ifndef LIBRERIADIGITAL_LIBRARY_H
#define LIBRERIADIGITAL_LIBRARY_H

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

LIBRARY_API void hello();

LIBRARY_API bool loadRawImage(const char *filePath, unsigned short* image, int width, int height);
LIBRARY_API bool saveRawImage(const char* filePath, const unsigned short* image, int width, int height);

LIBRARY_API void rotate(unsigned short* src, unsigned short* dst, int width, int height, int direction);
LIBRARY_API void flip(unsigned short* src, unsigned short* dst, int width, int height, int direction);
LIBRARY_API void adjustContrast(unsigned short* src, unsigned short* dst, int rows, int cols, float contrastLevel, float amplificationFactor);
LIBRARY_API void highContrast(unsigned short* src, unsigned short* dst, int width, int height, float threshold, float contrastBoost);
LIBRARY_API void highPassContrast(unsigned short* src, unsigned short* dst, int width, int height, float contrastBoost);
LIBRARY_API void highPassContrast2(unsigned short* src, unsigned short* dst, int width, int height, float threshold, float contrastBoost);
LIBRARY_API void highPassContrast3(unsigned short* src, unsigned short* dst, int width, int height, float threshold, float contrastBoost);
LIBRARY_API void perfilado(unsigned short* src, unsigned short* dst, int width, int height, float threshold, int amplificationFactor,int kernel_size);
LIBRARY_API void perfilado1(unsigned short* src, unsigned short* dst, int width, int height, float threshold, int amplificationFactor);
LIBRARY_API void contrastEnhancement(unsigned short* src, unsigned short* dst, int width, int height, float contrastLevel);
LIBRARY_API void deteccionBordes(unsigned short* src, unsigned short* dst, int width, int height, float threshold, int amplificationFactor);


LIBRARY_API void rotateP(unsigned short* src, unsigned short* dst, int width, int height, int direction);
LIBRARY_API void flipP(unsigned short* src, unsigned short* dst, int width, int height, int direction);

#ifdef __cplusplus
}
#endif

#endif // LIBRERIADIGITAL_LIBRARY_H
