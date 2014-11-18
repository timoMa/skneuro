#--------------------------------------------------------------
# default build-type (release)
#--------------------------------------------------------------
IF(DEFINED CMAKE_BUILD_TYPE)
   SET(CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE} CACHE STRING "Choose the type of build, options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel.")
ELSE()
   SET(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel.")
ENDIF()

#--------------------------------------------------------------
# skneuro
#--------------------------------------------------------------
cmake_minimum_required(VERSION 2.6)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules)
include(CMakeCopyIfDifferent)
project(skneuro)

#--------------------------------------------------------------
# c++ 11
#--------------------------------------------------------------
SET(CMAKE_CXX_FLAGS "-std=c++0x")


#--------------------------------------------------------------
# global headers
#--------------------------------------------------------------
file(GLOB_RECURSE headers include/*.hxx)
include_directories(include)

#--------------------------------------------------------------
# vigra
#--------------------------------------------------------------
FIND_PACKAGE(VIGRA REQUIRED)
include_directories(${VIGRA_INCLUDE_DIR})

#--------------------------------------------------------------
# lemon
#--------------------------------------------------------------
#FIND_PACKAGE(LEMON REQUIRED)
include_directories(${LEMON_INCLUDE_DIR})
add_definitions(-DWITH_LEMON_GRAPH)
add_definitions(-DWITH_LEMON)
#--------------------------------------------------------------
# Boost
#--------------------------------------------------------------
message(STATUS "build with Boost")
find_package(Boost 1.47 REQUIRED)
find_package(Boost COMPONENTS python REQUIRED)
add_definitions(-DWITH_BOOST)
include_directories(${Boost_INCLUDE_DIR})

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()



#--------------------------------------------------------------
# Python Lib
#--------------------------------------------------------------
find_package (PythonLibs REQUIRED)
include_directories(
    ${PYTHON_INCLUDE_DIRS}
    ${PYTHON_NUMPY_INCLUDE_DIR}
    ${Boost_INCLUDE_DIR}
    ${Boost_PYTHON_INCLUDE_DIR}
)


#--------------------------------------------------------------
# thread lib
#--------------------------------------------------------------
find_package(Threads)


include( CTest )
#--------------------------------------------------------------
# add src to project tree
#--------------------------------------------------------------
add_subdirectory(src) 


#--------------------------------------------------------------
# add python to project tree
#--------------------------------------------------------------
add_subdirectory(python) 
