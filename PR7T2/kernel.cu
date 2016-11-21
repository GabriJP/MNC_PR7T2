#include <cstdio>
#include <random>
#include <string.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "eTimer.h" //Utilidad propia para medir tiempos

//Definir el tamaño de la matriz a 6K
#define N 6*1024

//Este es el kernel
#pragma region "Kernel"
__global__ void sumaKernel(double *c, const double *a, const double *b, const double alpha, const double beta) {
	//Localizar coordenadas absolutas en base al bloque y al hilo
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;

	//Realizar la operación que es coalescing
	c[y*N + x] = alpha*a[y*N + x] + beta*b[y*N + x];
}
#pragma endregion

int main(int argc, char *argv[]) {
#pragma region "Inicialización"
	double *A, *B, *C;
	double alpha = 0.7;
	double beta = 0.6;
	std::default_random_engine generador;
	std::normal_distribution<double> distribucion(0.0, 1.0);

	//Reservamos espacio en la memoria principal para A, B y C
	//Versión de Microsoft para malloc alineado
	A = (double*)_aligned_malloc(N*N*sizeof(double), 64);
	B = (double*)_aligned_malloc(N*N * sizeof(double), 64);
	C = (double*)_aligned_malloc(N*N * sizeof(double), 64);

	//Rellenamos aleatoriamente las matrices A y B
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			A[i*N + j] = distribucion(generador);
			B[i*N + j] = distribucion(generador);
		}
	}

	eTimer *Tcpu = new eTimer(); //CPU
	eTimer *THtD = new eTimer(); //Host to device
	eTimer *Tkernel = new eTimer(); //Kernel
	eTimer *TDtH = new eTimer(); //Device to host
#pragma endregion

#pragma region "CPU"
	//Parte de la CPU
	Tcpu->start();
	//Sumamos en la CPU
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			C[i*N + j] = alpha*A[i*N + j] + beta*B[i*N + j];
		}
	}

	Tcpu->stop();
	Tcpu->report("CPU");

	//Imprimimos unos casos de prueba
	for (int i = 0; i < 5; i++)
	{
		printf("%lf ", C[i]);
	}
	printf("\n\n");

	//Para evitar un posterior falso test, reseteamos el resultado de la CPU
	memset(C, 0, N*N * sizeof(double));
	for (int i = 0; i < 5; i++)
	{
		printf("%lf ", C[i]);
	}
	printf("\n\n");
	//Fin de la CPU
#pragma endregion

#pragma region "GPU"
	//Parte de la GPU
	cudaError_t cudaStatus;

	//Seleccionar el dispositivo al comenzar y resetearlo al final
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Error seleccionando el dispositivo");
		return 1;
	}

	//Almacén en la memoria de la GPU para A, B y C
	double *dev_A, *dev_B, *dev_C;

	//Reservar espacio en la GPU para A, B y C
	cudaStatus = cudaMalloc((void**)&dev_A, N*N * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Error reservando espacio para A");
		return 1;
	}

	cudaStatus = cudaMalloc((void**)&dev_B, N*N * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Error reservando espacio para B");
		return 1;
	}

	cudaStatus = cudaMalloc((void**)&dev_C, N*N * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Error reservando espacio para C");
		return 1;
	}

	//Inicio del proceso en GPU
	THtD->start();
	cudaStatus = cudaMemcpy(dev_A, A, N*N * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Error copiando A en la GPU");
		return 1;
	}
	cudaStatus = cudaMemcpy(dev_B, B, N*N * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Error copiando B en la GPU");
		return 1;
	}
	THtD->stop();
	THtD->report("HostToDevice");

	//Calculamos el ancho de banda efectivo de la transferencia
	double AnchoBanda = 2 * N*N * sizeof(double) / THtD->get();
	printf("\nAncho de banda promedio: %lf GiB/s\n", AnchoBanda*1.0e-9);

	//Cronómetro del Kernel
	Tkernel->start();
	//Dimensión del grid de bloques y el bloque de hilos
	dim3 Grid, Block;
	Block.x = 32;
	Block.y = 16;
	Grid.x = N / Block.x;
	Grid.y = N / Block.y;

	//Se lanza el Kernel
	sumaKernel <<< Grid, Block >>> (dev_C, dev_A, dev_B, alpha, beta);

	//Se comprueba el lanzamiento del Kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "sumaKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		return 1;
	}

	//Se espera a la finalización del Kernel
	cudaStatus = cudaDeviceSynchronize();
	Tkernel->stop();
	Tkernel->report("Kernel");

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Error en la sincrinizacion: %s\n", cudaGetErrorString(cudaStatus));
		return 1;
	}

	//Copia de resultado en C de GPU a CPU
	TDtH->start();
	cudaStatus = cudaMemcpy(C, dev_C, N*N * sizeof(double), cudaMemcpyDeviceToHost);
	TDtH->stop();
	TDtH->report("DeviceToHost");

	//Impresión de casos de prueba
	for (int i = 0; i < 5; i++)
	{
		printf("%lf ", C[i]);
	}
	printf("\n\n");

	//Reseteo de la GPU
	cudaStatus = cudaDeviceReset();
	//Fin de la parte de GPU
#pragma endregion

	delete Tcpu;
	delete THtD;
	delete Tkernel;
	delete TDtH;

	std::getchar();

	return 0;
}