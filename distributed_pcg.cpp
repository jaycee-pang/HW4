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
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int N = (argc > 1) ? atoi(argv[1]) : 1000; // Default matrix size or from command line

    CSRSpMat A(N, N); // Initialize the matrix
    // Fill the matrix A appropriately here

    std::vector<double> b(N, 1.0); // Example vector b
    std::vector<double> x(N, 0.0); // Solution vector

    double tol = 1e-6; // Tolerance for convergence
    parallel_CG(A, b, x, tol, MPI_COMM_WORLD);

    MPI_Finalize();
    return 0;
}
