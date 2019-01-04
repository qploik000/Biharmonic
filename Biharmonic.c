#include <stdio.h>
#include <stdlib.h>
#include <mkl_dfti.h>
#include <complex.h>
#include <time.h>
#include <math.h>
#include <omp.h>
/* 
gcc -o Final Final.c -lm -I/opt/intel/mkl/include -L/opt/intel/mkl/lib/intel64 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -ldl -fopenmp
*/
typedef double _Complex cmpx;

void dtrid(int M, double dk, double *b, double *x)
{
    double temp;
    double *T1 = malloc (M * sizeof(double));
    double *T2 = malloc (M * sizeof(double));
    double *T3 = malloc (M * sizeof(double));

    for(int i = 0; i < M; i++)
    {
        T1[i] = 1 + 1 / (2 * (i + 0.5));
        T2[i] = -2 - dk / pow(i + 0.5, 2);
        T3[i] = 1 - 1 / (2 * (i + 1.5));
    }
  
    for(int k = 1; k < M; k++)
    {
        temp = T3[k - 1] / T2[k - 1];
        T2[k] = T2[k] - temp * T1[k - 1];
        b[k] = b[k] - temp * b[k - 1];
    }
  
    x[M - 1] = b[M - 1] / T2[M - 1];
  
    for(int k = M - 2; k >= 0; k--)
        x[k] = (b[k] - T1[k] * x[k + 1]) / T2[k];

    free(T1); free(T2); free(T3);
}

void ctrid(int M, double dk, cmpx *b, cmpx *x)
{
  double temp;
  double *T1 = malloc (M * sizeof(double));
  double *T2 = malloc (M * sizeof(double));
  double *T3 = malloc (M * sizeof(double));

  for(int i = 0; i < M; i++)
  {
      T1[i] = 1 + 1 / (2 * (i + 0.5));
      T2[i] = -2 - dk / pow(i + 0.5, 2);
      T3[i] = 1 - 1 / (2 * (i + 1.5));
  }

  for(int k = 1; k < M; k++)
  {
      temp = T3[k - 1] / T2[k - 1];
      T2[k] = T2[k] - temp * T1[k - 1];
      b[k] = b[k] - temp * b[k - 1];
  }
  
  x[M - 1] = b[M - 1] / T2[M - 1];
  
  for(int k = M - 2; k >= 0; k--)
      x[k] = (b[k] - T1[k] * x[k + 1]) / T2[k];

  free(T1); free(T2); free(T3);
}

void fft_d(int d, int N, double *in, MKL_Complex16 *df)
{
    int i;
    double temp;
    DFTI_DESCRIPTOR_HANDLE handle;
    MKL_LONG myStatus;
    myStatus = DftiCreateDescriptor (&handle, DFTI_DOUBLE, DFTI_REAL, 1, N);
    myStatus = DftiSetValue(handle, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    myStatus = DftiCommitDescriptor (handle);
    myStatus = DftiComputeForward (handle, in, df);
    myStatus = DftiFreeDescriptor (&handle);
    ///////////////////////////////////////////////////////// 
    if (d % 4 == 1)
    {
        for (i = 0; i < N; i++)
        {
            temp = pow(i, d);
            df[i].real = -temp * df[i].imag;
            df[i].imag =  temp * df[i].real;
        }
    }
    else if (d % 4 == 2)
    {
        for (i = 0; i < N; i++)
        {
            temp = pow(i, d);
            df[i].real = -temp * df[i].real;
            df[i].imag = -temp * df[i].imag;
        }
    }
    else if (d % 4 == 3)
    {
        for (i = 0; i < N; i++)
        {
            temp = pow(i, d);
            df[i].real =  temp * df[i].imag;
            df[i].imag = -temp * df[i].real;
        }
    }
    else
    {
        for (i = 0; i < N; i++)
        {
            temp = pow(i, d);
            df[i].real =  temp * df[i].real;
            df[i].imag =  temp * df[i].imag;
        }
    }
}

void ifft(int N, MKL_Complex16 *fft_cof, double *out)
{
    DFTI_DESCRIPTOR_HANDLE handle;
    MKL_LONG myStatus;
    myStatus = DftiCreateDescriptor (&handle, DFTI_DOUBLE, DFTI_REAL, 1, N);
    myStatus = DftiSetValue(handle, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    myStatus = DftiCommitDescriptor (handle);
    myStatus = DftiComputeBackward (handle, fft_cof, out);
    myStatus = DftiFreeDescriptor (&handle);
}

int main(int argc, char **argv)
{
    int M, Mn, Mm, N;
    struct timespec start[5], end[5];
    double accum;

    if(argc != 4)
    {
        printf("./Biharmonic <Mn> <Mm> <N>\n");
        return 0;
    }
    
    Mn = atoi(argv[1]); Mm = atoi(argv[2]); N = atoi(argv[3]);
    printf("M = [%2d ~ %2d ], N = %d\n", Mn, Mm, N);

    int d0 = 0;
    double pi = 4 * atan(1);
    double *max = malloc ((Mm + 1 ) * sizeof(double));
    double a, b, delta_r1, delta_r2, point;

    double *d        = malloc (N * sizeof(double)),
           *theta    = malloc (N * sizeof(double)),
           *u1_theta = malloc (N * sizeof(double)),
           *un_theta = malloc (N * sizeof(double));

    MKL_Complex16 *G_k = (MKL_Complex16*) malloc (N * sizeof(MKL_Complex16)),
                  *H_k = (MKL_Complex16*) malloc (N * sizeof(MKL_Complex16));

    for (int i = 0; i < N; i++)
    {
        theta[i]    = 2 * pi / N * i;
        
        //u1_theta[i] = 0.0;
        //un_theta[i] = -0.5 * (1 + cos(theta[i]));

        u1_theta[i] = exp(cos(theta[i]) + sin(theta[i])); //g(1, theta)
        un_theta[i] = (cos(theta[i]) + sin(theta[i])) * exp(cos(theta[i]) + sin(theta[i])); //h(1, theta)
    }

    ///////////////////boundary condition/////////////////////
    fft_d(d0, N, u1_theta, G_k);
    fft_d(d0, N, un_theta, H_k);

    for (int i = 1; i < N / 2; i++)
    {
        G_k[N - i].real =  G_k[i].real;
        G_k[N - i].imag = -G_k[i].imag;

        H_k[N - i].real =  H_k[i].real;
        H_k[N - i].imag = -H_k[i].imag;
    }

    for(int k = 0; k < N / 2; k++)
    {
        d[k]         = pow(k, 2);
        d[N - 1 - k] = pow(k + 1, 2);
    }

    for(int size = Mn; size <= Mm; size++)
    {

        M = pow(2, size);
        printf("M = %6d, ", M);

        MKL_Complex16 *F_0 = (MKL_Complex16*) malloc (M * N * sizeof(MKL_Complex16)),
                      *F_k = (MKL_Complex16*) malloc (M * N * sizeof(MKL_Complex16)),
                      *u_k = (MKL_Complex16*) malloc (M * N * sizeof(MKL_Complex16)),
                      *v_k = (MKL_Complex16*) malloc (M * N * sizeof(MKL_Complex16));

        double *ur_theta = malloc (M * N * sizeof(double)),
               *u        = malloc (M * N * sizeof(double)),
               *ex       = malloc (M * N * sizeof(double)),
               *r  = malloc (M * sizeof(double)),
               *a1 = malloc (M * N * sizeof(double)),
               *a2 = calloc (M,  sizeof(double)),
               *z1 = malloc (M * N * sizeof(double)),
               *z2 = malloc (M * N * sizeof(double));

        cmpx *y1 = malloc (M * N * sizeof(cmpx)),
             *y2 = malloc (M * N * sizeof(cmpx)),
             *b1 = malloc (M * N * sizeof(cmpx)),
             *b2 = malloc (M * N * sizeof(cmpx));

        point    = (2 * M + 1);
        delta_r1 = 2 / point;
        delta_r2 = pow(delta_r1, 2);

        a = 2 / delta_r2;
        b = M / (M - 0.5);

        for (int i = 0; i < M; i++)
            r[i] = (i + 0.5) * delta_r1;

        //////////////////////////////F矩陣 F=[F F_k_s]////////////////////
        clock_gettime (CLOCK_REALTIME, &start[1]);
        #pragma omp parallel for
        for (int i = 0; i < M; i++)
        {
            for (int k = 0; k < N; k++)
            {
                //ur_theta[i * N + k] = 0.0;
                ur_theta[i * N + k] = 4 * exp(r[i] * (cos(theta[k]) + sin(theta[k])));

                F_0[i * N + k].real = 0.0;
                F_0[i * N + k].imag = 0.0;
            }

            fft_d(d0, N, &ur_theta[i * N], &F_k[i * N]);

            for (int k = 0; k < N / 2; k++)
            {
                F_k[i * N + k].real = delta_r2 * F_k[i * N + k].real;
                F_k[i * N + k].imag = delta_r2 * F_k[i * N + k].imag;

                F_k[i * N + N - 1 - k].real = delta_r2 * F_k[i * N + k + 1].real;
                F_k[i * N + N - 1 - k].imag = delta_r2 * F_k[i * N + k + 1].imag;
            }
        }
        clock_gettime (CLOCK_REALTIME, &end[1]);

        for (int k = 0; k < N; k++)
        {
            F_0[(M - 1) * N + k].real = -b * G_k[k].real;
            F_0[(M - 1) * N + k].imag = -b * G_k[k].imag;

            F_k[(M - 1) * N + k].real = F_k[(M - 1) * N + k].real - b * ((-2 / delta_r2 - pow(k, 2)) * G_k[k].real + (2 / delta_r1 + 1) * H_k[k].real);
            F_k[(M - 1) * N + k].imag = F_k[(M - 1) * N + k].imag - b * ((-2 / delta_r2 - pow(k, 2)) * G_k[k].imag + (2 / delta_r1 + 1) * H_k[k].imag);
        }
        ////////////////////////A矩陣//////////////////////////

        a2[M - 1] = a * b;

        clock_gettime (CLOCK_REALTIME, &start[2]);
        #pragma omp parallel for
        for (int k = 0; k < N; k++)
        {
            //Tz2 = alpha2
            dtrid(M, d[k], a2, &z2[k * M]);

            for (int i = 0; i < M; i++)
                a1[k * M + i] = delta_r2 * z2[k * M + i];

            //Tz1 = alpha1 + delta_r2 * z2
            dtrid(M, d[k], &a1[k * M], &z1[k * M]);

            for (int i = 0; i < M; i++)
                b2[k * M + i] = F_k[i * N + k].real + F_k[i * N + k].imag * I;
            
            //Ty2 = Fk
            ctrid(M, d[k], &b2[k * M], &y2[k * M]);

            for (int i = 0; i < M; i++)
                b1[k * M + i] = F_0[i * N + k].real + F_0[i * N + k].imag * I + delta_r2 * y2[k * M + i];

            //Ty1 = F + delta}_r2 * y2
            ctrid(M, d[k], &b1[k * M], &y1[k * M]);

            for (int i = 0; i < M; i++)
            {
                u_k[i * N + k].real = creal(y1[k * M + i] - y1[k * M + M - 1] / (1 + z1[k * M + M - 1]) * z1[k * M + i]);
                u_k[i * N + k].imag = cimag(y1[k * M + i] - y1[k * M + M - 1] / (1 + z1[k * M + M - 1]) * z1[k * M + i]);
            }
        }
        clock_gettime (CLOCK_REALTIME, &end[2]);

        max[size] = 0.0;
        clock_gettime (CLOCK_REALTIME, &start[3]);
        #pragma omp parallel for
        for (int i = 0; i < M; i++)
        {
            double error;
            ifft(N, &u_k[i * N], &u[i * N]);
            for (int k = 0; k < N; k++)
            {
                //ex[i * N + k] = (1 - pow(r[i], 2)) * (1 + r[i] * cos(theta[k])) / 4;
                ex[i * N + k] = exp(r[i] * (cos(theta[k]) + sin(theta[k])));
                error = fabs(ex[i * N + k] - u[i * N + k] / N);
                if (error > max[size])
                    max[size] = error;
            }
        }
        clock_gettime (CLOCK_REALTIME, &end[3]);

        printf("Max error = % e, ", max[size]);
        if(size != Mn)
            printf("% lf\n", log2(max[size - 1] / max[size]));
        else
            printf("\n");
/*
        printf("\n");
        for(int i = 1; i <= 3; i++)
        {
            accum = ( end[i].tv_sec - start[i].tv_sec ) + ( end[i].tv_nsec - start[i].tv_nsec ) / 1e+9;
            printf("%.3lf\t", accum);
        }
        */
        free(r); free(u); free(ex);
        free(a1); free(a2); free(z1); free(z2);
        free(y1); free(y2); free(b1); free(b2);
        free(F_0); free(F_k); free(u_k); free(v_k); free(ur_theta);
    }

    free(d); free(theta); free(u1_theta); free(un_theta); free(max);
    free(G_k), free(H_k);

    return 0;
}

