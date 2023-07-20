cmake_minimum_required(VERSION 3.20)
project(LibreriaDigital)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
set(BUILD_SHARED_LIBS TRUE)

# Agrega la opción -fopenmp a las opciones de compilación
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -m32")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m32")

# Incluye la ruta de MinGW
#include_directories(/usr/lib/gcc/x86_64-w64-mingw32/10-win32/include)
#include_directories(/usr/x86_64-w64-mingw32/include)
#include_directories(/usr/i686-w64-mingw32/include)

set(CMAKE_TOOLCHAIN_FILE ${CMAKE_CURRENT_LIST_DIR}/toolchain-mingw32.cmake)

# Configura la generación de la biblioteca compartida (.so)
add_library(LibreriaDigital SHARED librarySequential.cpp libraryBasic.cpp libraryParallel.cpp)

set_target_properties(LibreriaDigital PROPERTIES OUTPUT_NAME "LibreriaDigital")

# Configura las opciones de compilación
target_compile_options(LibreriaDigital PRIVATE -Wall -Wextra -Werror=attributes)

# Configura la salida de la biblioteca compartida
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/build/)

# Asegura que el directorio de salida exista
file(MAKE_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})

# Define la macro LIBRARY_API
if (WIN32)
    target_compile_definitions(LibreriaDigital PRIVATE LIBRARY_EXPORTS)
endif ()



