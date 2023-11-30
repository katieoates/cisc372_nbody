// Katie Oates & John Henry

#include <stdlib.h>
#include <math.h>
#include "vector.h"
#include "config.h"
#include <cuda_runtime.h>


__global__ void compute_kernel(vector3 *values, double *hPos, double *hVel, double *mass){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (9 >= NUMENTITIES || j >= NUMENTITIES){
        return;
    }

    if (i == j){
        FILL_VECTOR (values[i*NUMENTITIES + j], 0, 0, 0);
    } else {
        vector3 distance;
        for (int k = 0; k < 3; k++){
            distance[k] = hPos[i*3 + k] - hPos[j*3 + k];
            double magnitude_sq = distance[0]*distance[0] + distance[1]*distance[1] + distance[2]*distance[2];
            double magnitude = sqrt(magnitude_sq);
            double accelmag = -1 * GRAV_CONSTANT * mass[j] / magnitude_sq;
            FILL_VECTOR(values[i*NUMENTITIES + j], accelmag * distance[0] / magnitude, accelmag * distance[1] / magnitude, accelmag * distance[2] / magnitude);
        }
    }
}

__global__ void update_kernel(vector3 *values, double *hPos, double *hVel, double *mass){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= NUMENTITIES) return;

    vector3 accel_sum = {0, 0, 0};
    for (int j = 0; j < NUMENTITIES; j++) {
        for (int k = 0; k < 3; k++)
            accel_sum[k] += values[i*NUMENTITIES + j][k];
    }

    // Compute the new velocity based on the acceleration and time interval
    // Compute the new position based on the velocity and time interval
    for (int k = 0; k < 3; k++) {
        hVel[i*3 + k] += accel_sum[k] * INTERVAL;
        hPos[i*3 + k] += hVel[i*3 + k] * INTERVAL;
    }
}

//compute: Updates the positions and locations of the objects in the system based on gravity.
//Parameters: None
//Returns: None
//Side Effect: Modifies the hPos and hVel arrays with the new positions and accelerations after 1 INTERVAL
extern "C" void compute() {
    // Allocate memory on the device
    vector3 *d_values;
    double *d_hPos, *d_hVel, *d_mass;
    size_t values_size = sizeof(vector3) * NUMENTITIES * NUMENTITIES;
    size_t pos_vel_size = sizeof(double) * NUMENTITIES * 3;
    size_t mass_size = sizeof(double) * NUMENTITIES;
    cudaMalloc((void **)&d_values, values_size);
    cudaMalloc((void **)&d_hPos, pos_vel_size);
    cudaMalloc((void **)&d_hVel, pos_vel_size);
    cudaMalloc((void **)&d_mass, mass_size);

    // Copy the data to the device
    cudaMemcpy(d_hPos, hPos, pos_vel_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_hVel, hVel, pos_vel_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mass, mass, mass_size, cudaMemcpyHostToDevice);

    // Define grid and block sizes
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((NUMENTITIES + threadsPerBlock.x - 1) / threadsPerBlock.x, (NUMENTITIES + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Call the compute_kernel
    compute_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_values, d_hPos, d_hVel, d_mass);
    cudaDeviceSynchronize();

    // Call the update_kernel
    update_kernel<<<(NUMENTITIES + threadsPerBlock.x - 1) / threadsPerBlock.x, threadsPerBlock.x>>>(d_values, d_hPos, d_hVel, d_mass);
    cudaDeviceSynchronize();

    // Copy the updated data back to the host
    cudaMemcpy(hPos, d_hPos, pos_vel_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(hVel, d_hVel, pos_vel_size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_values);
    cudaFree(d_hPos);
    cudaFree(d_hVel);
    cudaFree(d_mass);
}