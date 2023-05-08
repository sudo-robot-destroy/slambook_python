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
CMAKE_SOURCE_DIR = /home/otooleat/slambook2/ch3/examples

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/otooleat/slambook2/ch3/examples/build

# Include any dependencies generated for this target.
include CMakeFiles/plotTrajectory.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/plotTrajectory.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/plotTrajectory.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/plotTrajectory.dir/flags.make

CMakeFiles/plotTrajectory.dir/plotTrajectory.cpp.o: CMakeFiles/plotTrajectory.dir/flags.make
CMakeFiles/plotTrajectory.dir/plotTrajectory.cpp.o: ../plotTrajectory.cpp
CMakeFiles/plotTrajectory.dir/plotTrajectory.cpp.o: CMakeFiles/plotTrajectory.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/otooleat/slambook2/ch3/examples/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/plotTrajectory.dir/plotTrajectory.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/plotTrajectory.dir/plotTrajectory.cpp.o -MF CMakeFiles/plotTrajectory.dir/plotTrajectory.cpp.o.d -o CMakeFiles/plotTrajectory.dir/plotTrajectory.cpp.o -c /home/otooleat/slambook2/ch3/examples/plotTrajectory.cpp

CMakeFiles/plotTrajectory.dir/plotTrajectory.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/plotTrajectory.dir/plotTrajectory.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/otooleat/slambook2/ch3/examples/plotTrajectory.cpp > CMakeFiles/plotTrajectory.dir/plotTrajectory.cpp.i

CMakeFiles/plotTrajectory.dir/plotTrajectory.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/plotTrajectory.dir/plotTrajectory.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/otooleat/slambook2/ch3/examples/plotTrajectory.cpp -o CMakeFiles/plotTrajectory.dir/plotTrajectory.cpp.s

# Object files for target plotTrajectory
plotTrajectory_OBJECTS = \
"CMakeFiles/plotTrajectory.dir/plotTrajectory.cpp.o"

# External object files for target plotTrajectory
plotTrajectory_EXTERNAL_OBJECTS =

plotTrajectory: CMakeFiles/plotTrajectory.dir/plotTrajectory.cpp.o
plotTrajectory: CMakeFiles/plotTrajectory.dir/build.make
plotTrajectory: /home/otooleat/slambook2/slambook2/3rdparty/Pangolin/build/libpango_glgeometry.so
plotTrajectory: /home/otooleat/slambook2/slambook2/3rdparty/Pangolin/build/libpango_python.so
plotTrajectory: /home/otooleat/slambook2/slambook2/3rdparty/Pangolin/build/libpango_scene.so
plotTrajectory: /home/otooleat/slambook2/slambook2/3rdparty/Pangolin/build/libpango_tools.so
plotTrajectory: /home/otooleat/slambook2/slambook2/3rdparty/Pangolin/build/libpango_video.so
plotTrajectory: /home/otooleat/slambook2/slambook2/3rdparty/Pangolin/build/libpango_geometry.so
plotTrajectory: /home/otooleat/slambook2/slambook2/3rdparty/Pangolin/build/libtinyobj.so
plotTrajectory: /home/otooleat/slambook2/slambook2/3rdparty/Pangolin/build/libpango_plot.so
plotTrajectory: /home/otooleat/slambook2/slambook2/3rdparty/Pangolin/build/libpango_display.so
plotTrajectory: /home/otooleat/slambook2/slambook2/3rdparty/Pangolin/build/libpango_vars.so
plotTrajectory: /home/otooleat/slambook2/slambook2/3rdparty/Pangolin/build/libpango_windowing.so
plotTrajectory: /home/otooleat/slambook2/slambook2/3rdparty/Pangolin/build/libpango_opengl.so
plotTrajectory: /usr/lib/x86_64-linux-gnu/libGLEW.so
plotTrajectory: /usr/lib/x86_64-linux-gnu/libOpenGL.so
plotTrajectory: /usr/lib/x86_64-linux-gnu/libGLX.so
plotTrajectory: /usr/lib/x86_64-linux-gnu/libGLU.so
plotTrajectory: /home/otooleat/slambook2/slambook2/3rdparty/Pangolin/build/libpango_image.so
plotTrajectory: /home/otooleat/slambook2/slambook2/3rdparty/Pangolin/build/libpango_packetstream.so
plotTrajectory: /home/otooleat/slambook2/slambook2/3rdparty/Pangolin/build/libpango_core.so
plotTrajectory: CMakeFiles/plotTrajectory.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/otooleat/slambook2/ch3/examples/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable plotTrajectory"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/plotTrajectory.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/plotTrajectory.dir/build: plotTrajectory
.PHONY : CMakeFiles/plotTrajectory.dir/build

CMakeFiles/plotTrajectory.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/plotTrajectory.dir/cmake_clean.cmake
.PHONY : CMakeFiles/plotTrajectory.dir/clean

CMakeFiles/plotTrajectory.dir/depend:
	cd /home/otooleat/slambook2/ch3/examples/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/otooleat/slambook2/ch3/examples /home/otooleat/slambook2/ch3/examples /home/otooleat/slambook2/ch3/examples/build /home/otooleat/slambook2/ch3/examples/build /home/otooleat/slambook2/ch3/examples/build/CMakeFiles/plotTrajectory.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/plotTrajectory.dir/depend

