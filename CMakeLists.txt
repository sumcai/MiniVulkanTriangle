cmake_minimum_required(VERSION 3.10)
project(HelloVulkan)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE "Debug")

# validation layer
#add_definitions(-DVALIDATE)
add_definitions(-DUSE_GLFW)
add_definitions(-DVK_NO_PROTOTYPES)

include_directories(Library)
include_directories(Library/glfw/include)

add_subdirectory(Library/glfw)
add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries (${PROJECT_NAME} glfw)

if(UNIX)
    message(STATUS "Linux link dl")
    target_link_libraries(${PROJECT_NAME} dl)
endif()

file(COPY shaders/ DESTINATION shaders)
