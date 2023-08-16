//
// Created by ovelardo on 20/07/23.
// Open / save functions
// We define open and save functions to apply to raw 16 bit image
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

//////Load and save Raw images 16 bit

bool loadRawImage(const char *filePath, unsigned short* image, int width, int height)
{
    // Convert filePath to std::string
    std::string filePathStr(filePath);

    // Open binary file in read mode
    std::ifstream file(filePathStr, std::ios::binary);
    if (!file.is_open()) {
        std::cout << "File could not be opened: " << filePath << std::endl;
        return false;
    }

    // Check file size
    file.seekg(0, std::ios::end);
    std::streampos fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    // Check file size against expected image size
    if (fileSize != static_cast<std::streamoff>(width * height * sizeof(unsigned short))) {
        std::cout << "Wrong file size." << std::endl;
        return false;
    }

    // Read file data into image array
    file.read(reinterpret_cast<char*>(image), fileSize);

    // Check read errors
    if (!file) {
        std::cout << "Error reading file." << std::endl;
        return false;
    }

    // Convert bytes to unsigned short
    for (int i = 0; i < width * height; i++) {
        unsigned char byte1 = static_cast<unsigned char>(image[i] & 0xFF);
        unsigned char byte2 = static_cast<unsigned char>((image[i] >> 8) & 0xFF);
        image[i] = (byte2 << 8) | byte1;
    }

    // Close file
    file.close();

    return true;
}


bool saveRawImage(const char* filePath, const unsigned short* image, int width, int height)
{
    // Open binary file in write mode
    std::ofstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        std::cout << "No se pudo abrir el archivo para escribir: " << filePath << std::endl;
        return false;
    }

    // Write image data into image file
    file.write(reinterpret_cast<const char*>(image), width * height * sizeof(unsigned short));

    // Check write errors
    if (!file) {
        std::cout << "Error writing file." << std::endl;
        return false;
    }

    // Close file
    file.close();

    return true;
}