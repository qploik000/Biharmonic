# You have to link MKL library
# Compile with : gcc -o Biharmonic Biharmonic.c -lm -I/opt/intel/mkl/include -L/opt/intel/mkl/lib/intel64 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -ldl -fopenmp
# M = [M1 ~ M2], delta r = 1 / Mi, delta theta = 2pi / N
# ./Biharmonic M1 M2 N