cmake_minimum_required(VERSION 3.14)
project(hw4)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

# Find MPI
find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

# Find Eigen3
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# Add executable
add_executable(pcg distributed_pcg.cpp)

# Link MPI and Eigen3 libraries
target_link_libraries(pcg PRIVATE MPI::MPI_CXX Eigen3::Eigen)

# Setup packaging if needed (used when you want to package the project)
set(GROUP_NAME "None" CACHE STRING "Your group name as it appears on bCourses (no spaces)")
if (NOT ${GROUP_NAME} STREQUAL "None")
    set(CPACK_GENERATOR TGZ)
    set(CPACK_PACKAGE_FILE_NAME "cs267${GROUP_NAME}_hw4")
    install(FILES distributed_pcg.cpp DESTINATION .)
    install(FILES ${CPACK_PACKAGE_FILE_NAME}.pdf DESTINATION .)
    include(CPack)
endif ()

# Ensure GNU compiler is used, adjust module loading appropriately
if (NOT "${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
    message(WARNING "Must use GNU Compiler for submission. Make sure you ran:\nmodule swap PrgEnv-${CMAKE_C_COMPILER_ID} PrgEnv-gnu")
endif ()
