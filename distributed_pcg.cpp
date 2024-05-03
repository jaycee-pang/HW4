#include <iostream>
#include <map>
#include <vector>
#include <cassert>
#include <mpi.h>
#include <numeric>

#include <Eigen/Sparse>

typedef Eigen::SparseMatrix<double> SpMat; // declares a column-major sparse matrix type of double
typedef Eigen::Triplet<double> T;

class CSRSpMat {
public:
  int rows, cols; // m=num rows, n = num columns 
  std::vector<double> V; // nonzero values. size (num nonzeros) can call this 'data'
  std::vector<int> col_idxs; // column idx of nonzeros, size num nonzeros
  std::vector<int> row_ptrs; // length num rows + 1. It is the index in V where the row starts 

public:
  CSRSpMat(const int& nr, const int& nc):
    rows(nr), cols(nc) {row_ptrs.resize(nr + 1, 0);};

  CSRSpMat(const CSRSpMat& m): 
    rows(m.mrows()), cols(m.ncols()), V(m.V), col_idxs(m.col_idxs), row_ptrs(m.row_ptrs) {}; 

  CSRSpMat& operator=(const CSRSpMat& m){ 
    if(this!=&m){
      rows=m.rows;
      cols=m.cols;
      V = m.V; 
      col_idxs = m.col_idxs; 
      row_ptrs = m.row_ptrs; 

    }   
    return *this; 
  }

  int mrows() const {return rows;}
  int ncols() const {return cols;}

  void insert(int i, int j, double val) {
    if (val != 0.0) {
        V.push_back(val);
        col_idxs.push_back(j);
        // row_ptrs[i+1]++; // need to increment for the next row 
        for (int r =i+1; r<rows+1; ++r) {
            row_ptrs[r]++;
        }
    }
    
  }

  double operator()(const int& j, const int& k) const {
    int start = row_ptrs[j];
    int end = row_ptrs[j+ 1];
    for (int i = start; i < end; i++) {
        if (col_idxs[i] == k) {
            return V[i];
        }
    }
    return 0.0;

  }

  void display() const{
    std::cout << "nonzeros: " << V.size() << std::endl;
    std::cout << "rows: " << rows << std::endl; 
    std::cout << "row_ptrs size: " << row_ptrs.size() << std::endl;
    std::cout << "cols " << cols << std::endl;
    std::cout << "col idxs: " << col_idxs.size() << std::endl;
    int idx = 0; 
    for (int i = 0; i<rows; i++) {
        for (int j=0; j<cols;j++) {
            if (idx < row_ptrs[i+1] && col_idxs[idx]==j) {
              // if current col is column index at idx , found the nonzero to print 
              std::cout << V[idx] << "\t";
              idx++;
            } 
            else {
              std::cout << "0\t";
            }
        }
        std::cout << std::endl;
    }
  }
  

  // parallel matrix-vector product with distributed vector xi
  std::vector<double> operator*(const std::vector<double>& xi) const {
    std::vector<double> local_result(row_ptrs.size() - 1, 0.0); 
    for (int i=0; i < rows; i++) {
      for (int k=row_ptrs[i]; k < row_ptrs[i+1]; k++) {
        local_result[i] += V[k] * xi[col_idxs[k]];
      }
    }
    std::vector<double> result(rows);
    MPI_Allreduce(local_result.data(), result.data(), rows, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); 
    return result;
  }

  std::vector<double> serial_mult(const std::vector<double>& xi) const {
    CSR format has length m+1 (last element is NNZ) from wikipedia pg on CSR 
    std::vector<double> result(row_ptrs.size() - 1, 0.0); 
    // loop over rows, for each row, do nonzeros 
    for (int i=0; i < rows; i++) {
      // this is k=row_ptrs[i] to row_ptrs[i+1] - 1
      for (int k=row_ptrs[i]; k < row_ptrs[i+1]; k++) {
        result[i] += V[k] * xi[col_idxs[k]]; // k index into V and col_idx is aligned for these 2 vectors
        // k will index into the values list because row_ptrs holds the indices of the values in V in each row 
      }
    }
    return result; 

  }


  void print()
  {
    std::cout << "Rows: " << rows << std::endl;
    std::cout << "Cols: " << cols << std::endl;
    std::cout << "num_values: " << V.size() << std::endl;
    std::cout << "First col_idx: " << col_idxs[0] << std::endl;
    std::cout << "row_ptrs.size(): " << row_ptrs.size() << std::endl;

      for (int i = 0; i < row_ptrs.size() - 1; i++)  // Ensure we don't go out of bounds
      {
        std::cout << "row_ptr at i = " << i << " = " << row_ptrs[i] << std::endl;
          for (int j = row_ptrs[i]; j < row_ptrs[i + 1]; j++)
          {
              // std::cout << "A at (" << i << "," << col_idxs[j] << "): " << V[j] << std::endl;
              std::cout << "Now here" << std::endl;
          }
      }
  }

  void distribute(MPI_Comm comm) {
    int rank, size; 
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    std::vector<int> local_sizes(size);

    int nr_rows = rows/size; // subset of rows to go to each proc , rows per proc 
    int remaining_rows = rows % size; 
    int start_row = rank * nr_rows; 
    // need th elast process to handle remaining rows when leftover local rows
    if (rank == size-1) {
      nr_rows = rows-start_row; 
    }
    MPI_Bcast(&nr_rows, 1, MPI_INT, 0, comm); // bcast: one proc sends same data to all proc
                                              // all proc needs to know how many local rows
    // mem for the local matrix components 
    std::vector<double> local_Vs(nr_rows); 
    std::vector<int> local_col_idxs(nr_rows); 
    // this is how we distribute the matrix data 
    MPI_Scatterv(V.data(), &row_ptrs[0], &row_ptrs[1], MPI_DOUBLE,
                 local_Vs.data(),nr_rows, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(col_idxs.data(), &row_ptrs[0], &row_ptrs[1], MPI_INT,
                 local_col_idxs.data(), nr_rows, MPI_INT, 0, MPI_COMM_WORLD);

    // bcast sends same data: all procs needs to know row pointers 
    MPI_Bcast(row_ptrs.data(), row_ptrs.size(), MPI_INT, 0, MPI_COMM_WORLD);
  }

};

#include <cmath>

// parallel scalar product (u,v) (u and v are distributed)
double operator,(const std::vector<double>& u, const std::vector<double>& v){ 
  assert(u.size()==v.size());
  double sp=0.;
  for(int j=0; j<u.size(); j++){sp+=u[j]*v[j];}

  return sp; 
}

// norm of a vector u
double Norm(const std::vector<double>& u) { 
  return sqrt((u,u));
}

// addition of two vectors u+v
std::vector<double> operator+(const std::vector<double>& u, const std::vector<double>& v){ 
  assert(u.size()==v.size());
  std::vector<double> w=u;
  for(int j=0; j<u.size(); j++){w[j]+=v[j];}
  return w;
}

// multiplication of a vector by a scalar a*u
std::vector<double> operator*(const double& a, const std::vector<double>& u){ 
  std::vector<double> w(u.size());
  for(int j=0; j<w.size(); j++){w[j]=a*u[j];}
  return w;
}

// addition assignment operator, add v to u
void operator+=(std::vector<double>& u, const std::vector<double>& v){ 
  assert(u.size()==v.size());
  for(int j=0; j<u.size(); j++){u[j]+=v[j];}
}

/* block Jacobi preconditioner: perform forward and backward substitution
   using the Cholesky factorization of the local diagonal block computed by Eigen */
std::vector<double> prec(const Eigen::SimplicialCholesky<Eigen::SparseMatrix<double>>& P, const std::vector<double>& u){
  Eigen::VectorXd b(u.size());
  for (int i=0; i<u.size(); i++) b[i] = u[i];
  Eigen::VectorXd xe = P.solve(b); // solves Px=b (=xe)
  std::vector<double> x(u.size());
  for (int i=0; i<u.size(); i++) x[i] = xe[i];
  return x;
}

// distributed conjugate gradient
void CG(const CSRSpMat& A,
        const std::vector<double>& b,
        std::vector<double>& x,
        double tol=1e-6) {

  assert(b.size() == A.mrows());
  x.resize(b.size(),0.0);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Get the rank of the process

  int n = A.mrows();

  // get the local diagonal block of A
  std::vector<Eigen::Triplet<double>> coefficients;
  for (int i=0; i < n; i++) {
    for (int k=A.row_ptrs[i]; k <A.row_ptrs[i+1]; k++) {
      int j = A.col_idxs[k]; 
      if (j>= 0 && j < n) coefficients.push_back(Eigen::Triplet<double>(i,j,A.V[k])); 
    }
  }

  // compute the Cholesky factorization of the diagonal block for the preconditioner
  Eigen::SparseMatrix<double> B(n,n);
  B.setFromTriplets(coefficients.begin(), coefficients.end());
  Eigen::SimplicialCholesky<Eigen::SparseMatrix<double>> P(B);

  // std::cout << "Here" << std::endl;

  std::vector<double> r=b, z=prec(P,r), p=z, Ap=A*p;
  double np2=(p,Ap), alpha=0.,beta=0.;
  double nr = sqrt((z,r));

  std::vector<double> res = A*x;
  res += (-1)*b;

  // std::cout << "There" << std::endl;
  
  double rres = sqrt((res,res));

  int num_it = 0;
  while(rres>1e-5) {
    alpha = (nr*nr)/(np2);
    x += (+alpha)*p; 
    r += (-alpha)*Ap;
    z = prec(P,r);
    nr = sqrt((z,r));
    beta = (nr*nr)/(alpha*np2); 
    p = z+beta*p;    
    Ap=A*p;
    np2=(p,Ap);

    rres = sqrt((r,r));

    num_it++;
    if(rank == 0 && !(num_it%1)) {
      std::cout << "iteration: " << num_it << "\t";
      std::cout << "residual:  " << rres     << "\n";
    }
  }
}

// Command Line Option Processing
int find_arg_idx(int argc, char** argv, const char* option) {
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], option) == 0) {
            return i;
        }
    }
    return -1;
}

int find_int_arg(int argc, char** argv, const char* option, int default_value) {
    int iplace = find_arg_idx(argc, argv, option);

    if (iplace >= 0 && iplace < argc - 1) {
        return std::stoi(argv[iplace + 1]);
    }

    return default_value;
}




int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv); // Initialize the MPI environment
  int size, rank; 
  MPI_Comm_size(MPI_COMM_WORLD, &size); // Get the number of processes
  std::cout << "number of processes: " << size << std::endl;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Get the rank of the CURRENT process
    if (find_arg_idx(argc, argv, "-h") >= 0) {
        std::cout << "-N <int>: side length of the sparse matrix" << std::endl;
        return 0;
    }

  int N = find_int_arg(argc, argv, "-N", 100000); // global size // global # rows 

  assert(N%size == 0);
  int p = size; // # processes 
  int n = N/size; // number of local row, # rows each process will handle 
  
  // row-distributed matrix
  CSRSpMat A(N, N); // or CSRSpMat A(n, N);
  // int offset = n*rank; // changing this JP 
  int offset = rank*n; // start row index for CURRENT process changed by JP 
  
  // local rows of the 1D Laplacian matrix; local column indices start at -1 for rank > 0
  // JP: insert entries keeping in mind global indicies 
  for (int i=0; i<n; i++) {
    int global_row = offset+1;  // ex. N=100 and p=4, then each p has N/4 = 25 rows 
    A.insert(i, global_row, 2.0);
    if (global_row + i - 1 >= 0) A.insert(i,global_row - 1, -1.0); // insert if wihtin local p 
    if (global_row + i + 1 < N)  A.insert(i,global_row + 1, -1.0);
    if (global_row + i + N < N) A.insert(i, global_row + N, -1.0);
    if (global_row + i - N >= 0) A.insert(i, global_row - N, -1.0);
  }
  A.distribute(MPI_COMM_WORLD);
  std::cout << "Starting A:" << std::endl;
  A.print();

  // initial guess
  std::vector<double> x(n,0);

  // right-hand side
  std::vector<double> b(n,1);

  MPI_Barrier(MPI_COMM_WORLD);
  double time = MPI_Wtime();

  CG(A,b,x);

  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) std::cout << "wall time for CG: " << MPI_Wtime()-time << std::endl;

  std::vector<double> r = A*x + (-1)*b;

  double err = Norm(r)/Norm(b);
  if (rank == 0) std::cout << "|Ax-b|/|b| = " << err << std::endl;

  MPI_Finalize(); // Finalize the MPI environment

  return 0;
}

