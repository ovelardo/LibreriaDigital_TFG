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

#ifdef __cplusplus
}
#endif

#endif // LIBRERIADIGITAL_LIBRARYPARALLEL_H