//
// Created by ovelardo on 20/07/23.
//

#ifndef LIBRERIADIGITAL_LIBRARYBASIC_H
#define LIBRERIADIGITAL_LIBRARYBASIC_H

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

LIBRARY_API bool loadRawImage(const char *filePath, unsigned short* image, int width, int height);
LIBRARY_API bool saveRawImage(const char* filePath, const unsigned short* image, int width, int height);

#ifdef __cplusplus
}
#endif

#endif // LIBRERIADIGITAL_LIBRARYBASIC_H