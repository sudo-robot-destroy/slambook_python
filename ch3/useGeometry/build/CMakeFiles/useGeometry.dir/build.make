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
include CMakeFiles/useGeometry.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/useGeometry.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/useGeometry.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/useGeometry.dir/flags.make

CMakeFiles/useGeometry.dir/useGeometry.o: CMakeFiles/useGeometry.dir/flags.make
CMakeFiles/useGeometry.dir/useGeometry.o: ../useGeometry.cpp
CMakeFiles/useGeometry.dir/useGeometry.o: CMakeFiles/useGeometry.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/otooleat/slambook2/ch3/useGeometry/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/useGeometry.dir/useGeometry.o"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/useGeometry.dir/useGeometry.o -MF CMakeFiles/useGeometry.dir/useGeometry.o.d -o CMakeFiles/useGeometry.dir/useGeometry.o -c /home/otooleat/slambook2/ch3/useGeometry/useGeometry.cpp

CMakeFiles/useGeometry.dir/useGeometry.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/useGeometry.dir/useGeometry.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/otooleat/slambook2/ch3/useGeometry/useGeometry.cpp > CMakeFiles/useGeometry.dir/useGeometry.i

CMakeFiles/useGeometry.dir/useGeometry.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/useGeometry.dir/useGeometry.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/otooleat/slambook2/ch3/useGeometry/useGeometry.cpp -o CMakeFiles/useGeometry.dir/useGeometry.s

# Object files for target useGeometry
useGeometry_OBJECTS = \
"CMakeFiles/useGeometry.dir/useGeometry.o"

# External object files for target useGeometry
useGeometry_EXTERNAL_OBJECTS =

useGeometry: CMakeFiles/useGeometry.dir/useGeometry.o
useGeometry: CMakeFiles/useGeometry.dir/build.make
useGeometry: CMakeFiles/useGeometry.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/otooleat/slambook2/ch3/useGeometry/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable useGeometry"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/useGeometry.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/useGeometry.dir/build: useGeometry
.PHONY : CMakeFiles/useGeometry.dir/build

CMakeFiles/useGeometry.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/useGeometry.dir/cmake_clean.cmake
.PHONY : CMakeFiles/useGeometry.dir/clean

CMakeFiles/useGeometry.dir/depend:
	cd /home/otooleat/slambook2/ch3/useGeometry/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/otooleat/slambook2/ch3/useGeometry /home/otooleat/slambook2/ch3/useGeometry /home/otooleat/slambook2/ch3/useGeometry/build /home/otooleat/slambook2/ch3/useGeometry/build /home/otooleat/slambook2/ch3/useGeometry/build/CMakeFiles/useGeometry.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/useGeometry.dir/depend

