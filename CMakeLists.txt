cmake_minimum_required(VERSION 3.10.0)
project(personal-healthcare-ai VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(OpenCV_DIR "D:/Software/cmake/lib/opencv/opencv/build/x64/vc16/lib")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE})

add_executable(personal-healthcare-ai ${SOURCES})


target_link_libraries(personal-healthcare-ai ${OpenCV_LIB})

include(CTest)
enable_testing()

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
