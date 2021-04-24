// Assignment-4.cpp : Matrix multiplication by MPI parallel programming.
//

#include "mpi.h"
#include <iostream>
using namespace std;
#define N 6

double a[N][32], b[32][N], c[N][N];
MPI_Status status;
int main(int argc, char** argv)
{
    int process, uniqueId, numOfProcesses, source, destination, rows, offset, x, y, z, taskPerProcess, reminder;

    double startTime, endTime;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &uniqueId);
    MPI_Comm_size(MPI_COMM_WORLD, &process);

    // substract main process from num of processes
    numOfProcesses = process - 1;
    
    // exit execution if only 1 process is available
    if (process == 1) {
        cout << "\nNumber of process must be less than " << N + 1 << " and greater than 1.\n";
        MPI_Finalize();
        exit(0);
    }

    // divide tasks to each process
    if (process > N + 1) {
        taskPerProcess = 1;
    }
    else {
        taskPerProcess = N / numOfProcesses;
    }

    if (uniqueId == 0) {
        startTime = MPI_Wtime();

        // intialize two input arrays
        for (x = 0; x < N; x++) {
            for (y = 0; y < 32; y++) {
                a[x][y] = 2.0;
            }
        }
        for (x = 0; x < 32; x++) {
            for (y = 0; y < N; y++) {
                b[x][y] = 2.0;
            }
        }

        // Send data to process
        offset = 0;
        reminder = N % numOfProcesses;
        for (destination = 1; destination <= numOfProcesses; destination++)
        {
            rows = (destination <= reminder) ? taskPerProcess + 1 : taskPerProcess;
            MPI_Send(&offset, 1, MPI_INT, destination, 1, MPI_COMM_WORLD);
            MPI_Send(&rows, 1, MPI_INT, destination, 1, MPI_COMM_WORLD);
            MPI_Send(&a[offset][0], rows * 32, MPI_DOUBLE, destination, 1, MPI_COMM_WORLD);
            MPI_Send(&b, 32 * N, MPI_DOUBLE, destination, 1, MPI_COMM_WORLD);
            offset = offset + rows;
        }

        // Take the result from all the process
        for (x = 1; x <= numOfProcesses; x++)
        {
            source = x;
            MPI_Recv(&offset, 1, MPI_INT, source, 2, MPI_COMM_WORLD, &status);
            MPI_Recv(&rows, 1, MPI_INT, source, 2, MPI_COMM_WORLD, &status);
            MPI_Recv(&c[offset][0], rows * N, MPI_DOUBLE, source, 2, MPI_COMM_WORLD, &status);
        }

        endTime = MPI_Wtime();
        cout.precision(6);
        // calculate time consumed by program execution
        cout << "Time elapsed during the calculation: " << fixed << endTime - startTime << " seconds \n\n";

        // show result array
        cout << "Result:\n";
        cout.precision(2);
        for (x = 0; x < N; x++) {
            for (y = 0; y < N; y++)
                cout << fixed << c[x][y] << "  ";
            cout << "\n";
        }

    }

    if (uniqueId > 0) {
        // process multiplication from input sent by each process
        source = 0;
        MPI_Recv(&offset, 1, MPI_INT, source, 1, MPI_COMM_WORLD, &status);
        MPI_Recv(&rows, 1, MPI_INT, source, 1, MPI_COMM_WORLD, &status);
        MPI_Recv(&a, rows * 32, MPI_DOUBLE, source, 1, MPI_COMM_WORLD, &status);
        MPI_Recv(&b, 32 * N, MPI_DOUBLE, source, 1, MPI_COMM_WORLD, &status);

        // calculate multiplication
        for (z = 0; z < N; z++)
            for (x = 0; x < rows; x++) {
                c[x][z] = 0.0;
                for (y = 0; y < 32; y++)
                    c[x][z] += a[x][y] * b[y][z];
            }


        // send result data
        MPI_Send(&offset, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
        MPI_Send(&rows, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
        MPI_Send(&c, rows * N, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
    }

    MPI_Finalize();
}