# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.25

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
CMAKE_SOURCE_DIR = "/home/deepam/Desktop/SEM 6/CSE560 - GPU Computing/CSE560-Lab/Assignment 2/Assignment02_code"

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = "/home/deepam/Desktop/SEM 6/CSE560 - GPU Computing/CSE560-Lab/Assignment 2/Assignment02_code"

# Include any dependencies generated for this target.
include CMakeFiles/Assignment02.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/Assignment02.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/Assignment02.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/Assignment02.dir/flags.make

CMakeFiles/Assignment02.dir/src/main.cu.o: CMakeFiles/Assignment02.dir/flags.make
CMakeFiles/Assignment02.dir/src/main.cu.o: CMakeFiles/Assignment02.dir/includes_CUDA.rsp
CMakeFiles/Assignment02.dir/src/main.cu.o: src/main.cu
CMakeFiles/Assignment02.dir/src/main.cu.o: CMakeFiles/Assignment02.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/home/deepam/Desktop/SEM 6/CSE560 - GPU Computing/CSE560-Lab/Assignment 2/Assignment02_code/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object CMakeFiles/Assignment02.dir/src/main.cu.o"
	/opt/cuda/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/Assignment02.dir/src/main.cu.o -MF CMakeFiles/Assignment02.dir/src/main.cu.o.d -x cu -c "/home/deepam/Desktop/SEM 6/CSE560 - GPU Computing/CSE560-Lab/Assignment 2/Assignment02_code/src/main.cu" -o CMakeFiles/Assignment02.dir/src/main.cu.o

CMakeFiles/Assignment02.dir/src/main.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/Assignment02.dir/src/main.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/Assignment02.dir/src/main.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/Assignment02.dir/src/main.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target Assignment02
Assignment02_OBJECTS = \
"CMakeFiles/Assignment02.dir/src/main.cu.o"

# External object files for target Assignment02
Assignment02_EXTERNAL_OBJECTS =

Assignment02: CMakeFiles/Assignment02.dir/src/main.cu.o
Assignment02: CMakeFiles/Assignment02.dir/build.make
Assignment02: CMakeFiles/Assignment02.dir/linkLibs.rsp
Assignment02: CMakeFiles/Assignment02.dir/objects1
Assignment02: CMakeFiles/Assignment02.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir="/home/deepam/Desktop/SEM 6/CSE560 - GPU Computing/CSE560-Lab/Assignment 2/Assignment02_code/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA executable Assignment02"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Assignment02.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/Assignment02.dir/build: Assignment02
.PHONY : CMakeFiles/Assignment02.dir/build

CMakeFiles/Assignment02.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/Assignment02.dir/cmake_clean.cmake
.PHONY : CMakeFiles/Assignment02.dir/clean

CMakeFiles/Assignment02.dir/depend:
	cd "/home/deepam/Desktop/SEM 6/CSE560 - GPU Computing/CSE560-Lab/Assignment 2/Assignment02_code" && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" "/home/deepam/Desktop/SEM 6/CSE560 - GPU Computing/CSE560-Lab/Assignment 2/Assignment02_code" "/home/deepam/Desktop/SEM 6/CSE560 - GPU Computing/CSE560-Lab/Assignment 2/Assignment02_code" "/home/deepam/Desktop/SEM 6/CSE560 - GPU Computing/CSE560-Lab/Assignment 2/Assignment02_code" "/home/deepam/Desktop/SEM 6/CSE560 - GPU Computing/CSE560-Lab/Assignment 2/Assignment02_code" "/home/deepam/Desktop/SEM 6/CSE560 - GPU Computing/CSE560-Lab/Assignment 2/Assignment02_code/CMakeFiles/Assignment02.dir/DependInfo.cmake" --color=$(COLOR)
.PHONY : CMakeFiles/Assignment02.dir/depend

