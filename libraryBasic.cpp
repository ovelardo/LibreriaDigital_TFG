//
// Created by ovelardo on 20/07/23.
//

#include <iostream>
#include "libraryBasic.h"
#include <mutex>
#include <fstream>

#ifdef _WIN32
// Windows platform
    #define LIBRARY_API __declspec(dllexport)
#else
// Non-Windows platforms
#define LIBRARY_API
#endif

using namespace std;

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