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
CMAKE_SOURCE_DIR = /home/otooleat/slambook2/ch3/useGeometry

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/otooleat/slambook2/ch3/useGeometry/build

# Include any dependencies generated for this target.
include CMakeFiles/coordinateTransform.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/coordinateTransform.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/coordinateTransform.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/coordinateTransform.dir/flags.make

CMakeFiles/coordinateTransform.dir/coordinateTransform.o: CMakeFiles/coordinateTransform.dir/flags.make
CMakeFiles/coordinateTransform.dir/coordinateTransform.o: ../coordinateTransform.cpp
CMakeFiles/coordinateTransform.dir/coordinateTransform.o: CMakeFiles/coordinateTransform.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/otooleat/slambook2/ch3/useGeometry/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/coordinateTransform.dir/coordinateTransform.o"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/coordinateTransform.dir/coordinateTransform.o -MF CMakeFiles/coordinateTransform.dir/coordinateTransform.o.d -o CMakeFiles/coordinateTransform.dir/coordinateTransform.o -c /home/otooleat/slambook2/ch3/useGeometry/coordinateTransform.cpp

CMakeFiles/coordinateTransform.dir/coordinateTransform.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/coordinateTransform.dir/coordinateTransform.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/otooleat/slambook2/ch3/useGeometry/coordinateTransform.cpp > CMakeFiles/coordinateTransform.dir/coordinateTransform.i

CMakeFiles/coordinateTransform.dir/coordinateTransform.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/coordinateTransform.dir/coordinateTransform.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/otooleat/slambook2/ch3/useGeometry/coordinateTransform.cpp -o CMakeFiles/coordinateTransform.dir/coordinateTransform.s

# Object files for target coordinateTransform
coordinateTransform_OBJECTS = \
"CMakeFiles/coordinateTransform.dir/coordinateTransform.o"

# External object files for target coordinateTransform
coordinateTransform_EXTERNAL_OBJECTS =

coordinateTransform: CMakeFiles/coordinateTransform.dir/coordinateTransform.o
coordinateTransform: CMakeFiles/coordinateTransform.dir/build.make
coordinateTransform: CMakeFiles/coordinateTransform.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/otooleat/slambook2/ch3/useGeometry/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable coordinateTransform"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/coordinateTransform.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/coordinateTransform.dir/build: coordinateTransform
.PHONY : CMakeFiles/coordinateTransform.dir/build

CMakeFiles/coordinateTransform.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/coordinateTransform.dir/cmake_clean.cmake
.PHONY : CMakeFiles/coordinateTransform.dir/clean

CMakeFiles/coordinateTransform.dir/depend:
	cd /home/otooleat/slambook2/ch3/useGeometry/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/otooleat/slambook2/ch3/useGeometry /home/otooleat/slambook2/ch3/useGeometry /home/otooleat/slambook2/ch3/useGeometry/build /home/otooleat/slambook2/ch3/useGeometry/build /home/otooleat/slambook2/ch3/useGeometry/build/CMakeFiles/coordinateTransform.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/coordinateTransform.dir/depend

