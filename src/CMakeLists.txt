cmake_minimum_required(VERSION 3.9)
project(RobTree)
set(CMAKE_CXX_STANDARD 20)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_COMPILE_COMMANDS_PATH ${CMAKE_SOURCE_DIR}/compile_commands.json)
include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported OUTPUT ipo_error)

# Set default build type to Release if not specified
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
    message(STATUS "Build type not specified: Use Release by default")
endif()

if (CMAKE_BUILD_TYPE STREQUAL "Release")
    message(STATUS "Compiling in release mode")
else()
    message(STATUS "Compiling in debug mode")    
    add_compile_definitions(DEBUG=1)
endif()

if (CMAKE_BUILD_TYPE STREQUAL Release)
    
    if (MSVC)
        message(STATUS "Using MSVC compiler")
        add_compile_options(/Ot)
    else()
        add_compile_options(-O3)
        add_compile_options(-march=native)
        add_compile_options(-funroll-loops)
    endif()

endif()
   
    
# Define include directories
set(INCLUDE_DIRS
    ${CMAKE_SOURCE_DIR}/Utilities/include
    ${CMAKE_SOURCE_DIR}/DataStructures/include
    ${CMAKE_SOURCE_DIR}/Engine/include
)

include_directories(${INCLUDE_DIRS})

# Shared library (all sources except main.cpp)
file(GLOB LIB_SRC_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/DataStructures/src/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Utilities/src/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Engine/src/*.cpp
)
add_library(RobTreeLib STATIC ${LIB_SRC_FILES})

# Main executable
add_executable(RobTree ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
target_link_libraries(RobTree RobTreeLib)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  find_package(Catch2 3 REQUIRED)

  file(GLOB TEST_SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp)
  add_executable(tests ${TEST_SRC_FILES})
  target_link_libraries(tests PRIVATE RobTreeLib Catch2::Catch2WithMain)
endif()
# Set target properties based on build type
if (CMAKE_BUILD_TYPE STREQUAL Release)        
    if( ipo_supported )
	    message(STATUS "IPO / LTO enabled")    
      set_property(TARGET RobTreeLib PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
      set_property(TARGET RobTree PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
      set_property(TARGET tests PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
    else()
         message(STATUS "IPO / LTO not supported: <${ipo_error}>")
    endif()
else() 
    
    # enable runtime profilling (with MSVC)
    if (MSVC)
        message(STATUS "Enabling profiler")    
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /PROFILE")
    else ()
        message(STATUS "No profiler")
    endif()
endif()

