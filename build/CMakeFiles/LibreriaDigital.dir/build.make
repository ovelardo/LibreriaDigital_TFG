# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/ovelardo/CLionProjects/LibreriaDigital

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ovelardo/CLionProjects/LibreriaDigital/build

# Include any dependencies generated for this target.
include CMakeFiles/LibreriaDigital.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/LibreriaDigital.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/LibreriaDigital.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/LibreriaDigital.dir/flags.make

CMakeFiles/LibreriaDigital.dir/librarySequential.cpp.obj: CMakeFiles/LibreriaDigital.dir/flags.make
CMakeFiles/LibreriaDigital.dir/librarySequential.cpp.obj: ../librarySequential.cpp
CMakeFiles/LibreriaDigital.dir/librarySequential.cpp.obj: CMakeFiles/LibreriaDigital.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ovelardo/CLionProjects/LibreriaDigital/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/LibreriaDigital.dir/librarySequential.cpp.obj"
	/usr/bin/x86_64-w64-mingw32-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/LibreriaDigital.dir/librarySequential.cpp.obj -MF CMakeFiles/LibreriaDigital.dir/librarySequential.cpp.obj.d -o CMakeFiles/LibreriaDigital.dir/librarySequential.cpp.obj -c /home/ovelardo/CLionProjects/LibreriaDigital/librarySequential.cpp

CMakeFiles/LibreriaDigital.dir/librarySequential.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/LibreriaDigital.dir/librarySequential.cpp.i"
	/usr/bin/x86_64-w64-mingw32-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ovelardo/CLionProjects/LibreriaDigital/librarySequential.cpp > CMakeFiles/LibreriaDigital.dir/librarySequential.cpp.i

CMakeFiles/LibreriaDigital.dir/librarySequential.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/LibreriaDigital.dir/librarySequential.cpp.s"
	/usr/bin/x86_64-w64-mingw32-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ovelardo/CLionProjects/LibreriaDigital/librarySequential.cpp -o CMakeFiles/LibreriaDigital.dir/librarySequential.cpp.s

CMakeFiles/LibreriaDigital.dir/libraryBasic.cpp.obj: CMakeFiles/LibreriaDigital.dir/flags.make
CMakeFiles/LibreriaDigital.dir/libraryBasic.cpp.obj: ../libraryBasic.cpp
CMakeFiles/LibreriaDigital.dir/libraryBasic.cpp.obj: CMakeFiles/LibreriaDigital.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ovelardo/CLionProjects/LibreriaDigital/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/LibreriaDigital.dir/libraryBasic.cpp.obj"
	/usr/bin/x86_64-w64-mingw32-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/LibreriaDigital.dir/libraryBasic.cpp.obj -MF CMakeFiles/LibreriaDigital.dir/libraryBasic.cpp.obj.d -o CMakeFiles/LibreriaDigital.dir/libraryBasic.cpp.obj -c /home/ovelardo/CLionProjects/LibreriaDigital/libraryBasic.cpp

CMakeFiles/LibreriaDigital.dir/libraryBasic.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/LibreriaDigital.dir/libraryBasic.cpp.i"
	/usr/bin/x86_64-w64-mingw32-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ovelardo/CLionProjects/LibreriaDigital/libraryBasic.cpp > CMakeFiles/LibreriaDigital.dir/libraryBasic.cpp.i

CMakeFiles/LibreriaDigital.dir/libraryBasic.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/LibreriaDigital.dir/libraryBasic.cpp.s"
	/usr/bin/x86_64-w64-mingw32-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ovelardo/CLionProjects/LibreriaDigital/libraryBasic.cpp -o CMakeFiles/LibreriaDigital.dir/libraryBasic.cpp.s

CMakeFiles/LibreriaDigital.dir/libraryParallel.cpp.obj: CMakeFiles/LibreriaDigital.dir/flags.make
CMakeFiles/LibreriaDigital.dir/libraryParallel.cpp.obj: ../libraryParallel.cpp
CMakeFiles/LibreriaDigital.dir/libraryParallel.cpp.obj: CMakeFiles/LibreriaDigital.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ovelardo/CLionProjects/LibreriaDigital/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/LibreriaDigital.dir/libraryParallel.cpp.obj"
	/usr/bin/x86_64-w64-mingw32-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/LibreriaDigital.dir/libraryParallel.cpp.obj -MF CMakeFiles/LibreriaDigital.dir/libraryParallel.cpp.obj.d -o CMakeFiles/LibreriaDigital.dir/libraryParallel.cpp.obj -c /home/ovelardo/CLionProjects/LibreriaDigital/libraryParallel.cpp

CMakeFiles/LibreriaDigital.dir/libraryParallel.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/LibreriaDigital.dir/libraryParallel.cpp.i"
	/usr/bin/x86_64-w64-mingw32-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ovelardo/CLionProjects/LibreriaDigital/libraryParallel.cpp > CMakeFiles/LibreriaDigital.dir/libraryParallel.cpp.i

CMakeFiles/LibreriaDigital.dir/libraryParallel.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/LibreriaDigital.dir/libraryParallel.cpp.s"
	/usr/bin/x86_64-w64-mingw32-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ovelardo/CLionProjects/LibreriaDigital/libraryParallel.cpp -o CMakeFiles/LibreriaDigital.dir/libraryParallel.cpp.s

# Object files for target LibreriaDigital
LibreriaDigital_OBJECTS = \
"CMakeFiles/LibreriaDigital.dir/librarySequential.cpp.obj" \
"CMakeFiles/LibreriaDigital.dir/libraryBasic.cpp.obj" \
"CMakeFiles/LibreriaDigital.dir/libraryParallel.cpp.obj"

# External object files for target LibreriaDigital
LibreriaDigital_EXTERNAL_OBJECTS =

libLibreriaDigital.dll: CMakeFiles/LibreriaDigital.dir/librarySequential.cpp.obj
libLibreriaDigital.dll: CMakeFiles/LibreriaDigital.dir/libraryBasic.cpp.obj
libLibreriaDigital.dll: CMakeFiles/LibreriaDigital.dir/libraryParallel.cpp.obj
libLibreriaDigital.dll: CMakeFiles/LibreriaDigital.dir/build.make
libLibreriaDigital.dll: CMakeFiles/LibreriaDigital.dir/linklibs.rsp
libLibreriaDigital.dll: CMakeFiles/LibreriaDigital.dir/objects1.rsp
libLibreriaDigital.dll: CMakeFiles/LibreriaDigital.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/ovelardo/CLionProjects/LibreriaDigital/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX shared library libLibreriaDigital.dll"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/LibreriaDigital.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/LibreriaDigital.dir/build: libLibreriaDigital.dll
.PHONY : CMakeFiles/LibreriaDigital.dir/build

CMakeFiles/LibreriaDigital.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/LibreriaDigital.dir/cmake_clean.cmake
.PHONY : CMakeFiles/LibreriaDigital.dir/clean

CMakeFiles/LibreriaDigital.dir/depend:
	cd /home/ovelardo/CLionProjects/LibreriaDigital/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ovelardo/CLionProjects/LibreriaDigital /home/ovelardo/CLionProjects/LibreriaDigital /home/ovelardo/CLionProjects/LibreriaDigital/build /home/ovelardo/CLionProjects/LibreriaDigital/build /home/ovelardo/CLionProjects/LibreriaDigital/build/CMakeFiles/LibreriaDigital.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/LibreriaDigital.dir/depend

