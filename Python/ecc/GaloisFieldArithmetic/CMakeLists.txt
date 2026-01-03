cmake_minimum_required(VERSION 3.13)
project(GaloisFieldArithmetic)

set(CMAKE_CXX_STANDARD 14)

include_directories(.)

add_executable(GaloisFieldArithmetic
        GaloisField.cpp
        GaloisField.h
        GaloisFieldElement.cpp
        GaloisFieldElement.h
        GaloisFieldOut.cpp
        GaloisFieldPolynomial.cpp
        GaloisFieldPolynomial.h
        GaloisFieldTest.cpp
        GaloisProto.cpp
        GaloisShiftProto.cpp
        GFPDerivTest.cpp)
