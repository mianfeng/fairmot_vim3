# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
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
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/khadas/mnt/workspace/fairmot

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/khadas/mnt/workspace/fairmot/build

# Include any dependencies generated for this target.
include CMakeFiles/fairmot.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/fairmot.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/fairmot.dir/flags.make

CMakeFiles/fairmot.dir/test.cc.o: CMakeFiles/fairmot.dir/flags.make
CMakeFiles/fairmot.dir/test.cc.o: ../test.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/khadas/mnt/workspace/fairmot/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/fairmot.dir/test.cc.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/fairmot.dir/test.cc.o -c /home/khadas/mnt/workspace/fairmot/test.cc

CMakeFiles/fairmot.dir/test.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/fairmot.dir/test.cc.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/khadas/mnt/workspace/fairmot/test.cc > CMakeFiles/fairmot.dir/test.cc.i

CMakeFiles/fairmot.dir/test.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/fairmot.dir/test.cc.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/khadas/mnt/workspace/fairmot/test.cc -o CMakeFiles/fairmot.dir/test.cc.s

CMakeFiles/fairmot.dir/fairmot/src/demuxing_decoding.c.o: CMakeFiles/fairmot.dir/flags.make
CMakeFiles/fairmot.dir/fairmot/src/demuxing_decoding.c.o: ../fairmot/src/demuxing_decoding.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/khadas/mnt/workspace/fairmot/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building C object CMakeFiles/fairmot.dir/fairmot/src/demuxing_decoding.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/fairmot.dir/fairmot/src/demuxing_decoding.c.o   -c /home/khadas/mnt/workspace/fairmot/fairmot/src/demuxing_decoding.c

CMakeFiles/fairmot.dir/fairmot/src/demuxing_decoding.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/fairmot.dir/fairmot/src/demuxing_decoding.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/khadas/mnt/workspace/fairmot/fairmot/src/demuxing_decoding.c > CMakeFiles/fairmot.dir/fairmot/src/demuxing_decoding.c.i

CMakeFiles/fairmot.dir/fairmot/src/demuxing_decoding.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/fairmot.dir/fairmot/src/demuxing_decoding.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/khadas/mnt/workspace/fairmot/fairmot/src/demuxing_decoding.c -o CMakeFiles/fairmot.dir/fairmot/src/demuxing_decoding.c.s

CMakeFiles/fairmot.dir/fairmot/src/encoder.cpp.o: CMakeFiles/fairmot.dir/flags.make
CMakeFiles/fairmot.dir/fairmot/src/encoder.cpp.o: ../fairmot/src/encoder.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/khadas/mnt/workspace/fairmot/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/fairmot.dir/fairmot/src/encoder.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/fairmot.dir/fairmot/src/encoder.cpp.o -c /home/khadas/mnt/workspace/fairmot/fairmot/src/encoder.cpp

CMakeFiles/fairmot.dir/fairmot/src/encoder.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/fairmot.dir/fairmot/src/encoder.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/khadas/mnt/workspace/fairmot/fairmot/src/encoder.cpp > CMakeFiles/fairmot.dir/fairmot/src/encoder.cpp.i

CMakeFiles/fairmot.dir/fairmot/src/encoder.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/fairmot.dir/fairmot/src/encoder.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/khadas/mnt/workspace/fairmot/fairmot/src/encoder.cpp -o CMakeFiles/fairmot.dir/fairmot/src/encoder.cpp.s

CMakeFiles/fairmot.dir/fairmot/src/ionplayer.c.o: CMakeFiles/fairmot.dir/flags.make
CMakeFiles/fairmot.dir/fairmot/src/ionplayer.c.o: ../fairmot/src/ionplayer.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/khadas/mnt/workspace/fairmot/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building C object CMakeFiles/fairmot.dir/fairmot/src/ionplayer.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/fairmot.dir/fairmot/src/ionplayer.c.o   -c /home/khadas/mnt/workspace/fairmot/fairmot/src/ionplayer.c

CMakeFiles/fairmot.dir/fairmot/src/ionplayer.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/fairmot.dir/fairmot/src/ionplayer.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/khadas/mnt/workspace/fairmot/fairmot/src/ionplayer.c > CMakeFiles/fairmot.dir/fairmot/src/ionplayer.c.i

CMakeFiles/fairmot.dir/fairmot/src/ionplayer.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/fairmot.dir/fairmot/src/ionplayer.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/khadas/mnt/workspace/fairmot/fairmot/src/ionplayer.c -o CMakeFiles/fairmot.dir/fairmot/src/ionplayer.c.s

CMakeFiles/fairmot.dir/fairmot/src/lapjv.cpp.o: CMakeFiles/fairmot.dir/flags.make
CMakeFiles/fairmot.dir/fairmot/src/lapjv.cpp.o: ../fairmot/src/lapjv.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/khadas/mnt/workspace/fairmot/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/fairmot.dir/fairmot/src/lapjv.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/fairmot.dir/fairmot/src/lapjv.cpp.o -c /home/khadas/mnt/workspace/fairmot/fairmot/src/lapjv.cpp

CMakeFiles/fairmot.dir/fairmot/src/lapjv.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/fairmot.dir/fairmot/src/lapjv.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/khadas/mnt/workspace/fairmot/fairmot/src/lapjv.cpp > CMakeFiles/fairmot.dir/fairmot/src/lapjv.cpp.i

CMakeFiles/fairmot.dir/fairmot/src/lapjv.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/fairmot.dir/fairmot/src/lapjv.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/khadas/mnt/workspace/fairmot/fairmot/src/lapjv.cpp -o CMakeFiles/fairmot.dir/fairmot/src/lapjv.cpp.s

CMakeFiles/fairmot.dir/fairmot/src/tengine_operations.c.o: CMakeFiles/fairmot.dir/flags.make
CMakeFiles/fairmot.dir/fairmot/src/tengine_operations.c.o: ../fairmot/src/tengine_operations.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/khadas/mnt/workspace/fairmot/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building C object CMakeFiles/fairmot.dir/fairmot/src/tengine_operations.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/fairmot.dir/fairmot/src/tengine_operations.c.o   -c /home/khadas/mnt/workspace/fairmot/fairmot/src/tengine_operations.c

CMakeFiles/fairmot.dir/fairmot/src/tengine_operations.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/fairmot.dir/fairmot/src/tengine_operations.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/khadas/mnt/workspace/fairmot/fairmot/src/tengine_operations.c > CMakeFiles/fairmot.dir/fairmot/src/tengine_operations.c.i

CMakeFiles/fairmot.dir/fairmot/src/tengine_operations.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/fairmot.dir/fairmot/src/tengine_operations.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/khadas/mnt/workspace/fairmot/fairmot/src/tengine_operations.c -o CMakeFiles/fairmot.dir/fairmot/src/tengine_operations.c.s

CMakeFiles/fairmot.dir/fairmot/src/tracker.cc.o: CMakeFiles/fairmot.dir/flags.make
CMakeFiles/fairmot.dir/fairmot/src/tracker.cc.o: ../fairmot/src/tracker.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/khadas/mnt/workspace/fairmot/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object CMakeFiles/fairmot.dir/fairmot/src/tracker.cc.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/fairmot.dir/fairmot/src/tracker.cc.o -c /home/khadas/mnt/workspace/fairmot/fairmot/src/tracker.cc

CMakeFiles/fairmot.dir/fairmot/src/tracker.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/fairmot.dir/fairmot/src/tracker.cc.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/khadas/mnt/workspace/fairmot/fairmot/src/tracker.cc > CMakeFiles/fairmot.dir/fairmot/src/tracker.cc.i

CMakeFiles/fairmot.dir/fairmot/src/tracker.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/fairmot.dir/fairmot/src/tracker.cc.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/khadas/mnt/workspace/fairmot/fairmot/src/tracker.cc -o CMakeFiles/fairmot.dir/fairmot/src/tracker.cc.s

CMakeFiles/fairmot.dir/fairmot/src/trajectory.cc.o: CMakeFiles/fairmot.dir/flags.make
CMakeFiles/fairmot.dir/fairmot/src/trajectory.cc.o: ../fairmot/src/trajectory.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/khadas/mnt/workspace/fairmot/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CXX object CMakeFiles/fairmot.dir/fairmot/src/trajectory.cc.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/fairmot.dir/fairmot/src/trajectory.cc.o -c /home/khadas/mnt/workspace/fairmot/fairmot/src/trajectory.cc

CMakeFiles/fairmot.dir/fairmot/src/trajectory.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/fairmot.dir/fairmot/src/trajectory.cc.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/khadas/mnt/workspace/fairmot/fairmot/src/trajectory.cc > CMakeFiles/fairmot.dir/fairmot/src/trajectory.cc.i

CMakeFiles/fairmot.dir/fairmot/src/trajectory.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/fairmot.dir/fairmot/src/trajectory.cc.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/khadas/mnt/workspace/fairmot/fairmot/src/trajectory.cc -o CMakeFiles/fairmot.dir/fairmot/src/trajectory.cc.s

# Object files for target fairmot
fairmot_OBJECTS = \
"CMakeFiles/fairmot.dir/test.cc.o" \
"CMakeFiles/fairmot.dir/fairmot/src/demuxing_decoding.c.o" \
"CMakeFiles/fairmot.dir/fairmot/src/encoder.cpp.o" \
"CMakeFiles/fairmot.dir/fairmot/src/ionplayer.c.o" \
"CMakeFiles/fairmot.dir/fairmot/src/lapjv.cpp.o" \
"CMakeFiles/fairmot.dir/fairmot/src/tengine_operations.c.o" \
"CMakeFiles/fairmot.dir/fairmot/src/tracker.cc.o" \
"CMakeFiles/fairmot.dir/fairmot/src/trajectory.cc.o"

# External object files for target fairmot
fairmot_EXTERNAL_OBJECTS =

fairmot: CMakeFiles/fairmot.dir/test.cc.o
fairmot: CMakeFiles/fairmot.dir/fairmot/src/demuxing_decoding.c.o
fairmot: CMakeFiles/fairmot.dir/fairmot/src/encoder.cpp.o
fairmot: CMakeFiles/fairmot.dir/fairmot/src/ionplayer.c.o
fairmot: CMakeFiles/fairmot.dir/fairmot/src/lapjv.cpp.o
fairmot: CMakeFiles/fairmot.dir/fairmot/src/tengine_operations.c.o
fairmot: CMakeFiles/fairmot.dir/fairmot/src/tracker.cc.o
fairmot: CMakeFiles/fairmot.dir/fairmot/src/trajectory.cc.o
fairmot: CMakeFiles/fairmot.dir/build.make
fairmot: /usr/lib/aarch64-linux-gnu/libopencv_stitching.so.4.2.0
fairmot: /usr/lib/aarch64-linux-gnu/libopencv_aruco.so.4.2.0
fairmot: /usr/lib/aarch64-linux-gnu/libopencv_bgsegm.so.4.2.0
fairmot: /usr/lib/aarch64-linux-gnu/libopencv_bioinspired.so.4.2.0
fairmot: /usr/lib/aarch64-linux-gnu/libopencv_ccalib.so.4.2.0
fairmot: /usr/lib/aarch64-linux-gnu/libopencv_dnn_objdetect.so.4.2.0
fairmot: /usr/lib/aarch64-linux-gnu/libopencv_dnn_superres.so.4.2.0
fairmot: /usr/lib/aarch64-linux-gnu/libopencv_dpm.so.4.2.0
fairmot: /usr/lib/aarch64-linux-gnu/libopencv_face.so.4.2.0
fairmot: /usr/lib/aarch64-linux-gnu/libopencv_freetype.so.4.2.0
fairmot: /usr/lib/aarch64-linux-gnu/libopencv_fuzzy.so.4.2.0
fairmot: /usr/lib/aarch64-linux-gnu/libopencv_hdf.so.4.2.0
fairmot: /usr/lib/aarch64-linux-gnu/libopencv_hfs.so.4.2.0
fairmot: /usr/lib/aarch64-linux-gnu/libopencv_img_hash.so.4.2.0
fairmot: /usr/lib/aarch64-linux-gnu/libopencv_line_descriptor.so.4.2.0
fairmot: /usr/lib/aarch64-linux-gnu/libopencv_quality.so.4.2.0
fairmot: /usr/lib/aarch64-linux-gnu/libopencv_reg.so.4.2.0
fairmot: /usr/lib/aarch64-linux-gnu/libopencv_rgbd.so.4.2.0
fairmot: /usr/lib/aarch64-linux-gnu/libopencv_saliency.so.4.2.0
fairmot: /usr/lib/aarch64-linux-gnu/libopencv_shape.so.4.2.0
fairmot: /usr/lib/aarch64-linux-gnu/libopencv_stereo.so.4.2.0
fairmot: /usr/lib/aarch64-linux-gnu/libopencv_structured_light.so.4.2.0
fairmot: /usr/lib/aarch64-linux-gnu/libopencv_superres.so.4.2.0
fairmot: /usr/lib/aarch64-linux-gnu/libopencv_surface_matching.so.4.2.0
fairmot: /usr/lib/aarch64-linux-gnu/libopencv_tracking.so.4.2.0
fairmot: /usr/lib/aarch64-linux-gnu/libopencv_videostab.so.4.2.0
fairmot: /usr/lib/aarch64-linux-gnu/libopencv_viz.so.4.2.0
fairmot: /usr/lib/aarch64-linux-gnu/libopencv_xobjdetect.so.4.2.0
fairmot: /usr/lib/aarch64-linux-gnu/libopencv_xphoto.so.4.2.0
fairmot: /usr/lib/aarch64-linux-gnu/libopencv_highgui.so.4.2.0
fairmot: /usr/lib/aarch64-linux-gnu/libopencv_datasets.so.4.2.0
fairmot: /usr/lib/aarch64-linux-gnu/libopencv_plot.so.4.2.0
fairmot: /usr/lib/aarch64-linux-gnu/libopencv_text.so.4.2.0
fairmot: /usr/lib/aarch64-linux-gnu/libopencv_dnn.so.4.2.0
fairmot: /usr/lib/aarch64-linux-gnu/libopencv_ml.so.4.2.0
fairmot: /usr/lib/aarch64-linux-gnu/libopencv_phase_unwrapping.so.4.2.0
fairmot: /usr/lib/aarch64-linux-gnu/libopencv_optflow.so.4.2.0
fairmot: /usr/lib/aarch64-linux-gnu/libopencv_ximgproc.so.4.2.0
fairmot: /usr/lib/aarch64-linux-gnu/libopencv_video.so.4.2.0
fairmot: /usr/lib/aarch64-linux-gnu/libopencv_videoio.so.4.2.0
fairmot: /usr/lib/aarch64-linux-gnu/libopencv_imgcodecs.so.4.2.0
fairmot: /usr/lib/aarch64-linux-gnu/libopencv_objdetect.so.4.2.0
fairmot: /usr/lib/aarch64-linux-gnu/libopencv_calib3d.so.4.2.0
fairmot: /usr/lib/aarch64-linux-gnu/libopencv_features2d.so.4.2.0
fairmot: /usr/lib/aarch64-linux-gnu/libopencv_flann.so.4.2.0
fairmot: /usr/lib/aarch64-linux-gnu/libopencv_photo.so.4.2.0
fairmot: /usr/lib/aarch64-linux-gnu/libopencv_imgproc.so.4.2.0
fairmot: /usr/lib/aarch64-linux-gnu/libopencv_core.so.4.2.0
fairmot: CMakeFiles/fairmot.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/khadas/mnt/workspace/fairmot/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Linking CXX executable fairmot"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/fairmot.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/fairmot.dir/build: fairmot

.PHONY : CMakeFiles/fairmot.dir/build

CMakeFiles/fairmot.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/fairmot.dir/cmake_clean.cmake
.PHONY : CMakeFiles/fairmot.dir/clean

CMakeFiles/fairmot.dir/depend:
	cd /home/khadas/mnt/workspace/fairmot/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/khadas/mnt/workspace/fairmot /home/khadas/mnt/workspace/fairmot /home/khadas/mnt/workspace/fairmot/build /home/khadas/mnt/workspace/fairmot/build /home/khadas/mnt/workspace/fairmot/build/CMakeFiles/fairmot.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/fairmot.dir/depend

