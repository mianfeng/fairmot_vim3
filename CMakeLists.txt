# 声明要求的 cmake 最低版本
cmake_minimum_required(VERSION 3.10)

# 声明一个 cmake 工程
project(fairmot)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED on)
# set(CMAKE_BUILD_TYPE Debug)

# #指定交叉编译器路径
# set(TOOLCHAIN_CC "/usr/bin/aarch64-linux-gnu-gcc")
# set(TOOLCHAIN_CXX "/usr/bin/aarch64-linux-gnu-g++")
# #告诉cmake是进行交叉编译
# set(CMAKE_CROSSCOMPILING TRUE)
# #告知当前使用的是交叉编译方式，必须配置
# SET(CMAKE_SYSTEM_NAME Linux)
# # Define the compiler
# set(CMAKE_C_COMPILER ${TOOLCHAIN_CC})
# set(CMAKE_CXX_COMPILER ${TOOLCHAIN_CXX})

# set(OpenCV_DIR third_party/opencv/lib/cmake/opencv4)
# 添加OPENCV库
find_package(OpenCV)
# 添加OpenCV头文件
include_directories(${OpenCV_INCLUDE_DIRS})
# 显示OpenCV_INCLUDE_DIRS的值
message(${OpenCV_INCLUDE_DIRS})

FIND_PACKAGE( OpenMP REQUIRED)
if(OPENMP_FOUND)
# message("OPENMP FOUND")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

include_directories(
    ./
    ./include/
    /usr/include/eigen3/
    # ./utilities/
    # ./net/
    ./fairmot/include/
    ./fairmot/
)

link_directories(
    libs/aarch64/
    # libs/x86_64/
)

aux_source_directory(./fairmot/src SRC_DIR)

# 添加一个可执行程序
add_executable(
    fairmot
    # tm_fairmot_timvx.cpp
    # fairmot_mutithread.cc
    test.cc
    # old_tm.cpp
    # utilities/dataload.cpp
    # utilities/timer.cc
    # net/net.cpp
    ${SRC_DIR}
    )

# 将库文件链接到可执行程序上
target_link_libraries(
    fairmot
    ${OpenCV_LIBS}
    tengine-lite
    -lavutil -lavcodec -lavformat 
    -lamcodec -lamadec -lamvdec -lamavutils 
    -lpthread -lm -lrt  -lvpcodec
)
SET(CMAKE_BUILD_TYPE Release) 
# # 添加对gdb的支持 -DCMAKE_BUILD_TYPE=Debug
# SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
# SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")
# add_compile_options()
message(STATUS "Build type:${CMAKE_BUILD_TYPE}")