#include <cmath>
#include <iostream>
#include <vector>
#include <cassert>
#include <mpi.h>
#include <Eigen/Sparse>

typedef Eigen::SparseMatrix<double> SpMat; // Column-major sparse matrix type of double
typedef Eigen::Triplet<double> T;

// Compressed Sparse Row (CSR) Matrix Class
class CSRSpMat {
public:
    int rows, cols; // Dimensions of the matrix
    std::vector<double> V; // Non-zero values
    std::vector<int> col_idxs; // Column indices for non-zero values
    std::vector<int> row_ptrs; // Row pointers (indices in V where each row starts)

    // Constructor
    CSRSpMat(int nr, int nc) : rows(nr), cols(nc) {
        row_ptrs.resize(nr + 1, 0);
    }

    // Insert a non-zero value into the matrix
    void insert(int i, int j, double val) {
        if (val != 0.0) {
            V.push_back(val);
            col_idxs.push_back(j);
            for (int r = i + 1; r <= rows; ++r) {
                row_ptrs[r]++;
            }
        }
    }

    // Function to distribute matrix data across MPI processes
    void distribute(MPI_Comm comm, std::vector<double>& local_V, std::vector<int>& local_col_idxs, std::vector<int>& local_row_ptrs) {
        int rank, size;
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &size);

        int local_rows = rows / size; // Number of rows per process
        int extra = rows % size;
        int start_row = rank < extra ? (local_rows + 1) * rank : local_rows * rank + extra;
        int end_row = start_row + (rank < extra ? local_rows + 1 : local_rows);

        int start_index = row_ptrs[start_row]; // Start index in V for the local rows
        int end_index = row_ptrs[end_row];     // End index in V for the local rows

        // Resize local structures to hold the distributed parts of the matrix
        local_row_ptrs.resize(end_row - start_row + 1);
        local_V.assign(V.begin() + start_index, V.begin() + end_index);
        local_col_idxs.assign(col_idxs.begin() + start_index, col_idxs.begin() + end_index);

        // Adjust local row pointers
        int offset = local_row_ptrs[0];
        for (int &ptr : local_row_ptrs) {
            ptr -= offset;
        }
    }

    // Matrix-vector multiplication (parallel)
    std::vector<double> multiply(const std::vector<double>& xi) const {
        std::vector<double> result(rows, 0.0);
        for (int i = 0; i < rows; i++) {
            for (int k = row_ptrs[i]; k < row_ptrs[i + 1]; k++) {
                result[i] += V[k] * xi[col_idxs[k]];
            }
        }
        return result;
    }
};

// Parallel scalar product using MPI
double parallel_dot(const std::vector<double>& u, const std::vector<double>& v, MPI_Comm comm) {
    assert(u.size() == v.size());
    double local_dot = 0.0, global_dot = 0.0;
    for (size_t i = 0; i < u.size(); ++i) {
        local_dot += u[i] * v[i];
    }
    MPI_Allreduce(&local_dot, &global_dot, 1, MPI_DOUBLE, MPI_SUM, comm);
    return global_dot;
}

// Norm of a vector (parallel)
double parallel_norm(const std::vector<double>& u, MPI_Comm comm) {
    return std::sqrt(parallel_dot(u, u, comm));
}

// Conjugate Gradient Solver
void parallel_CG(const CSRSpMat& A, const std::vector<double>& b, std::vector<double>& x, double tol, MPI_Comm comm) {
    int rank;
    MPI_Comm_rank(comm, &rank);

    std::vector<double> r = b;     // Residual vector
    std::vector<double> p = r;     // Search direction
    std::vector<double> Ap;        // A*p vector
    double alpha, beta, rnorm, pAp;

    double rnorm_old = parallel_dot(r, r, comm);
    if (rank == 0) std::cout << "Initial residual norm: " << std::sqrt(rnorm_old) << std::endl;

    int iter = 0;
    while (true) {
        Ap = A.multiply(p);
        pAp = parallel_dot(p, Ap, comm);
        alpha = rnorm_old / pAp;

        if (pAp == 0) {
        if (rank == 0) std::cerr << "Error: Division by zero in calculating alpha (pAp is zero)." << std::endl;
        break;
        }   


        // Update x and r
        for (size_t i = 0; i < x.size(); ++i) {
            x[i] += alpha * p[i];
            r[i] -= alpha * Ap[i];
        }

        double rnorm_new = parallel_dot(r, r, comm);
        if (std::sqrt(rnorm_new) < tol) break; // Convergence check

        beta = rnorm_new / rnorm_old;
        for (size_t i = 0; i < p.size(); ++i) {
            p[i] = r[i] + beta * p[i];
        }

        rnorm_old = rnorm_new;
        if (rank == 0) {
            std::cout << "Iteration " << ++iter << ": Residual norm = " << std::sqrt(rnorm_new) << std::endl;
        }
    }

    if (rank == 0) {
        std::cout << "Conjugate Gradient completed after " << iter << " iterations." << std::endl;
    }

    return default_value;
}

void testCSRMatrixAndCGSolver(MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int N = 1000; // Example size of the matrix, should be divisible by 'size'
    assert(N % size == 0); // Ensure N is evenly divisible by the number of processes

    int local_rows = N / size; // Number of rows per process
    int offset = rank * local_rows; // Starting row index for this process

    CSRSpMat A(local_rows, N); // Local part of the CSR matrix of size local_rows x N

    // Initialize matrix A
    for (int i = 0; i < local_rows; i++) {
        int global_row = i + offset; // Convert local row index to global row index
        A.insert(i, global_row, 2.0); // Diagonal dominance
        if (global_row > 0) A.insert(i, global_row - 1, -1.0); // Sub-diagonal
        if (global_row < N - 1) A.insert(i, global_row + 1, -1.0); // Super-diagonal
    }

    std::vector<double> b(local_rows, 1.0); // Right-hand side vector initialized to all ones
    std::vector<double> x(local_rows, 0.0); // Solution vector initialized to zero

    // Call the Conjugate Gradient function
    double tol = 1e-6; // Tolerance for the CG solver
    CG(A, b, x, tol);

    // Optionally, print the results
    if (rank == 0) {
        std::cout << "Test completed. Solution vector x[0]: " << x[0] << std::endl;
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;

    // Determine the rank of the process
    int rank;
    MPI_Comm_rank(comm, &rank);

    if (argc > 1 && std::string(argv[1]) == "--test") {
        // Run tests only if a specific command line argument is provided
        testCSRMatrixAndCGSolver(comm);
    } else {
        // Regular execution path
        int N = find_int_arg(argc, argv, "-N", 10000); // Default to a size of 10,000 if not specified
        if (rank == 0) {
            std::cout << "Solving a system with matrix size " << N << "x" << N << std::endl;
        }
        CSRSpMat A(N, N); // Example matrix initialization
        std::vector<double> b(N, 1.0), x(N, 0.0);
        CG(A, b, x, 1e-6);
    }

    MPI_Finalize();
    return 0;
}


// int main(int argc, char* argv[]) {

//   MPI_Init(&argc, &argv); // Initialize the MPI environment
//   int size, rank; 
//   MPI_Comm_size(MPI_COMM_WORLD, &size); // Get the number of processes
//   std::cout << "number of processes: " << size << std::endl;
//   MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Get the rank of the CURRENT process
//     if (find_arg_idx(argc, argv, "-h") >= 0) {
//         std::cout << "-N <int>: side length of the sparse matrix" << std::endl;
//         return 0;
//     }

//   int N = find_int_arg(argc, argv, "-N", 100000); // global size // global # rows 

//   assert(N%size == 0);
//   int p = size; // # processes 
//   int n = N/size; // number of local row, # rows each process will handle 
//   std::cout << "local N/size: " << n << std::endl;

//   // CSRSpMat A(N, N); // or 
//   CSRSpMat A(n, N);

//   int offset = rank*n; // start row index for CURRENT process changed by JP 
//   std::cout << "offset: " << offset << std::endl;
//   // local rows of the 1D Laplacian matrix; local column indices start at -1 for rank > 0
//   // JP: insert entries keeping in mind global indicies 
//   for (int i=0; i<n; i++) {
//     int global_row = offset+1;  // ex. N=100 and p=4, then each p has N/4 = 25 rows 
//     A.insert(i, i, 2.0);
//     if (offset + i - 1 >= 0) A.insert(i, i - 1, -1.0);  
//     if (offset + i + 1 < N) A.insert(i, i + 1, -1.0);  
//     if (offset + i + N < N) A.insert(i, i + N, -1.0);  
//     if (offset + i - N >= 0) A.insert(i, i - N, -1.0);  
//   }
//   std::cout << "Rank " << rank << " has rows from " << offset << " to " << offset + n - 1 << std::endl;
//   // std::cout << "Starting A:" << std::endl;
//   // A.display();

//   // // initial guess
//   std::vector<double> x(n,0);

//   // // right-hand side
//   std::vector<double> b(n,1);

//   MPI_Barrier(MPI_COMM_WORLD);
//   double time = MPI_Wtime();

//   CG(A,b,x);

//   MPI_Barrier(MPI_COMM_WORLD);
//   if (rank == 0) std::cout << "wall time for CG: " << MPI_Wtime()-time << std::endl;

//   std::vector<double> r = A*x + (-1)*b;

//   double err = Norm(r)/Norm(b);
//   if (rank == 0) std::cout << "|Ax-b|/|b| = " << err << std::endl;


//   MPI_Finalize(); // Finalize the MPI environment

//   return 0;
// }

