const questionsByUnit = {
  unit1: [
    {
      question: "Which of the following is NOT a requirement for a set V to be a vector space over a field F?",
      options: [
        "Existence of a zero vector",
        "Existence of additive inverses",
        "Closure under scalar multiplication",
        "Existence of multiplicative identity for vectors"
      ],
      answer: "D"
    },
    {
      question: "The set of all polynomials with real coefficients forms a vector space over R under:",
      options: [
        "Standard addition and scalar multiplication",
        "Matrix multiplication",
        "Cross product",
        "Pointwise multiplication"
      ],
      answer: "A"
    },
    {
      question: "If f: V→R is a linear functional, what is Ker(f)?",
      options: [
        "The set of all vectors mapped to 1",
        "The set of all vectors mapped to 0",
        "The set of all unit vectors",
        "The set of all nonzero vectors"
      ],
      answer: "B"
    },
    {
      question: "If V is an n-dimensional vector space, what is the dimension of its dual space V∗?",
      options: ["0", "n", "2n", "3n"],
      answer: "B"
    },
    {
      question: "If f: R² → R is a linear functional given by f(x, y) = 3x + 4y, what is the Ker(f)?",
      options: [
        "{(x, y) ∣ 3x + 4y = 0}",
        "{(x, y) ∣ x = 0, y = 0}",
        "R",
        "{(3, 4)}"
      ],
      answer: "A"
    },
    {
      question: "Which of the following is NOT a property of an inner product?",
      options: [
        "⟨u, v⟩ = ⟨v, u⟩ (Symmetry)",
        "⟨u + v, w⟩ = ⟨u, w⟩ + ⟨v,w⟩ (Linearity)",
        "⟨cu, v⟩ = c⟨u, v⟩ (Scalar Multiplication)",
        "⟨u, u⟩ = −∥u∥²"
      ],
      answer: "D"
    },
    {
      question: "A set of vectors {v1, v2, ..., vn} in a vector space is linearly dependent if:",
      options: [
        "There exists a nontrivial linear combination that equals zero",
        "The determinant of the matrix formed by these vectors as columns is nonzero",
        "The vectors span the entire space",
        "The vectors are orthonormal"
      ],
      answer: "A"
    },
    {
      question: "If A and B are two n × n matrices, which of the following is NOT necessarily true?",
      options: [
        "(A + B)ᵀ = Aᵀ + Bᵀ",
        "(AB)ᵀ = AᵀBᵀ",
        "(AB)ᵀ = BᵀAᵀ",
        "AB = BA"
      ],
      answer: "D"
    },
    {
      question: "A function T: V → W is a linear transformation if:",
      options: [
        "T(v + w) = T(v) + T(w)",
        "T(cv) = cT(v)",
        "T maps zero to zero",
        "All of the above"
      ],
      answer: "D"
    },
    {
      question: "A square matrix Q is orthogonal if:",
      options: [
        "QᵀQ = I",
        "det(Q) = 0",
        "Q has all eigenvalues equal to zero",
        "Q is symmetric"
      ],
      answer: "A"
    },
    {
      question: "A system of linear equations has a unique solution if:",
      options: [
        "The determinant of the coefficient matrix is zero",
        "The coefficient matrix has full row rank and is square",
        "There are more equations than variables",
        "The system is inconsistent"
      ],
      answer: "B"
    },
    {
      question: "Which of the following conditions ensures a linear system Ax = b has at least one solution?",
      options: [
        "The coefficient matrix A is invertible",
        "A has a pivot in every row",
        "A has a pivot in every column",
        "A is symmetric"
      ],
      answer: "B"
    },
    {
      question: "A matrix A has a left inverse A⁻¹ if:",
      options: [
        "A is square and non-singular",
        "A is tall (more rows than columns) and has full column rank",
        "A is short (more columns than rows) and has full-row rank",
        "A is symmetric"
      ],
      answer: "B"
    },
    {
      question: "The determinant of an upper triangular matrix is:",
      options: [
        "The sum of its diagonal elements",
        "The product of its diagonal elements",
        "Always zero",
        "The trace of the matrix"
      ],
      answer: "B"
    },
    {
      question: "A system Ax = b has a unique solution if:",
      options: [
        "A is singular",
        "A is invertible",
        "A is an upper triangular matrix",
        "A has more columns than rows"
      ],
      answer: "B"
    },
    {
      question: "Which of the following is not a property of an inner product?",
      options: [
        "Linearity in the first argument",
        "Conjugate symmetry",
        "Distributivity over addition",
        "Antisymmetry"
      ],
      answer: "D"
    },
    {
      question: "If u = (2, -3) and v = (1, 4), what is u + v?",
      options: ["(3, 1)", "(1, 1)", "(2, 7)", "(3, -1)"],
      answer: "A"
    },
    {
      question: "Which set is linearly dependent?",
      options: [
        "{(1, 0), (0, 1)}",
        "{(1, 2), (2, 4)}",
        "{(2, 3, 1), (1, 0, 0), (0, 1, 0)}",
        "{(0, 0), (1, 1)}"
      ],
      answer: "B"
    },
    {
      question: "A matrix A is orthogonal if:",
      options: ["AᵀA = I", "A = Aᵀ", "det(A) = 0", "A² = A"],
      answer: "A"
    },
    {
      question: "What is the rank of a matrix?",
      options: [
        "The number of rows",
        "The number of columns",
        "The number of non-zero rows in its row echelon form",
        "The sum of diagonal elements"
      ],
      answer: "C"
    },
    {
      question: "The Moore-Penrose pseudo-inverse A⁺ of a matrix A is used when:",
      options: [
        "A is not invertible",
        "A is symmetric",
        "A is orthogonal",
        "A has integer entries"
      ],
      answer: "A"
    },
    {
      question: "If A is an m×n matrix and m > n, then the system Ax = b:",
      options: [
        "Always has a solution",
        "Has a solution only if b ∈ Col(A)",
        "Is overdetermined and inconsistent",
        "Has infinitely many solutions"
      ],
      answer: "B"
    },
    {
      question: "Which of the following matrices has both left and right inverses?",
      options: [
        "A 3×2 matrix of full rank",
        "A 2×3 matrix of full rank",
        "A 3×3 invertible matrix",
        "A singular matrix"
      ],
      answer: "C"
    },
    {
      question: "The solution to a triangular system is best found using:",
      options: [
        "Matrix inversion",
        "Gaussian elimination",
        "LU decomposition",
        "Forward or backward substitution"
      ],
      answer: "D"
    },
    {
      question: "Which of the following conditions guarantees linear independence of a set of vectors in Rⁿ?",
      options: [
        "The set contains n vectors and has rank n",
        "The vectors sum to zero",
        "Each vector is a scalar multiple of the others",
        "The determinant of their matrix is zero"
      ],
      answer: "A"
    },
    {
      question: "If the inner product of two vectors is zero, then the vectors are:",
      options: [
        "Parallel",
        "Orthogonal",
        "Linearly dependent",
        "Equal in magnitude"
      ],
      answer: "B"
    },
    {
      question: "A subspace of a vector space V must contain:",
      options: [
        "At least one non-zero vector",
        "Only linearly independent vectors",
        "The zero vector",
        "Only vectors from a basis"
      ],
      answer: "C"
    },
    {
      question: "Which operation is not defined for vectors in Rⁿ?",
      options: [
        "Vector addition",
        "Scalar multiplication",
        "Cross product (for n > 3)",
        "Dot product"
      ],
      answer: "C"
    },
    {
      question: "Which of the following defines a linear transformation T: R² → R²?",
      options: [
        "T(x, y) = (x², y)",
        "T(x, y) = (x + y, x − y)",
        "T(x, y) = (x + 1, y + 1)",
        "T(x, y) = (x, y, 0)"
      ],
      answer: "B"
    },
    {
      question: "If a set of vectors spans a space but is not linearly independent, then:",
      options: [
        "It contains redundant vectors",
        "It is a basis",
        "It cannot be used to express any vector in the space",
        "It has fewer vectors than the dimension"
      ],
      answer: "A"
    },
    {
      question: "Which of the following matrices is not invertible?",
      options: [
        "Identity matrix",
        "Diagonal matrix with no zero entries",
        "Singular matrix",
        "Matrix with full rank"
      ],
      answer: "C"
    },
    {
      question: "The pseudo-inverse of a matrix A minimizes the error in which sense?",
      options: [
        "Determinant",
        "Maximum norm",
        "Least squares",
        "Frobenius inner product"
      ],
      answer: "C"
    },
    {
      question: "If A is an orthogonal matrix, then which of the following is true?",
      options: [
        "Aᵀ = A",
        "A⁻¹ = Aᵀ",
        "det(A) = 0",
        "Columns of A are linearly dependent"
      ],
      answer: "B"
    },
    {
      question: "A system of equations has no solution if:",
      options: [
        "The coefficient matrix is invertible",
        "The augmented matrix has a row like [0 0 0 | 1]",
        "The rank of the coefficient matrix equals the number of variables",
        "It has free variables"
      ],
      answer: "B"
    },
    {
      question: "In an upper triangular system, backward substitution starts from:",
      options: [
        "First equation",
        "Last equation",
        "Middle equation",
        "Diagonal elements"
      ],
      answer: "B"
    },
    {
      question: "What is the dimension of the column space of a matrix called?",
      options: ["Nullity", "Trace", "Rank", "Determinant"],
      answer: "C"
    },
    {
      question: "Which of the following is true about the null space of a matrix A?",
      options: [
        "It always contains only the zero vector",
        "It is a subspace of the column space",
        "It contains all solutions to Ax = 0",
        "It has dimension equal to the rank of A"
      ],
      answer: "C"
    },
    {
      question: "If A is a 3×3 matrix and det(A) ≠ 0, then A is:",
      options: ["Singular", "Diagonalizable", "Orthogonal", "Invertible"],
      answer: "D"
    },
    {
      question: "Which statement is always true for a linear transformation T?",
      options: [
        "T(0) = 1",
        "T(u + v) = T(u) + T(v)",
        "T(u + v) = T(u) − T(v)",
        "T(cu) = c + T(u)"
      ],
      answer: "B"
    },
    {
      question: "The set of all solutions to a homogeneous system Ax = 0 forms:",
      options: [
        "A vector space",
        "An affine space",
        "A null set",
        "A scalar field"
      ],
      answer: "A"
    },
    {
      question: "A square matrix A is diagonalizable if:",
      options: [
        "It is orthogonal",
        "It has n linearly independent eigenvectors",
        "Its determinant is 1",
        "It is symmetric only"
      ],
      answer: "B"
    }
  ],
  unit2: [
    {
      question: "LU decomposition factors a square matrix A into:",
      options: ["A = QR", "A = LLt", "A = LU", "A = LDLt"],
      answer: "C"
    },
    {
      question: "What is the time complexity of computing LU decomposition of an n × n matrix (without pivoting)?",
      options: ["O(n²)", "O(n³)", "O(n log n)", "O(n)"],
      answer: "B"
    },
    {
      question: "LU decomposition fails without pivoting if:",
      options: ["A is a diagonal matrix", "A has a zero on the diagonal during elimination", "A is symmetric", "A is orthogonal"],
      answer: "B"
    },
    {
      question: "A Givens rotation is used to:",
      options: ["Scale a vector", "Reflect a vector", "Zero out elements in QR decomposition", "Diagonalize a matrix"],
      answer: "C"
    },
    {
      question: "A Householder reflector matrix H is:",
      options: ["Orthogonal and symmetric", "Diagonal", "Singular", "Only upper triangular"],
      answer: "A"
    },
    {
      question: "A Householder reflector transforms a vector into:",
      options: ["A scalar", "A zero vector", "A multiple of a unit vector", "A vector orthogonal to the original"],
      answer: "C"
    },
    {
      question: "QR decomposition of matrix A gives:",
      options: ["A = LU", "A = QR, where Q is orthogonal, R is upper triangular", "A = QR, where Q is diagonal", "A = QR, where R is symmetric"],
      answer: "B"
    },
    {
      question: "The Gram-Schmidt process is used to:",
      options: ["Solve linear equations", "Decompose a matrix into LU", "Orthogonalize a set of vectors", "Diagonalize a matrix"],
      answer: "C"
    },
    {
      question: "Which problem does the classical Gram-Schmidt process suffer from?",
      options: ["Instability due to pivoting", "Numerical instability due to loss of orthogonality", "Overflow error", "High space complexity"],
      answer: "B"
    },
    {
      question: "The condition number of a matrix A (in norm ||·||) is given by:",
      options: ["||A||", "||A⁻¹||", "||A|| + ||A⁻¹||", "||A||·||A⁻¹||"],
      answer: "D"
    },
    {
      question: "A matrix with a high condition number is:",
      options: ["Well-conditioned", "Ill-conditioned", "Singular", "Orthogonal"],
      answer: "B"
    },
    {
      question: "The geometric interpretation of the condition number is:",
      options: ["Ratio of determinant to trace", "Ratio of smallest to largest eigenvalue", "Maximum stretching to minimum stretching of space under A", "Number of zero entries in A"],
      answer: "C"
    },
    {
      question: "If A is orthogonal, its condition number (in 2-norm) is:",
      options: ["0", "1", "det(A)", "||A||"],
      answer: "B"
    },
    {
      question: "Which of the following is not a valid matrix norm?",
      options: ["Frobenius norm", "2-norm", "1-norm", "Hadamard norm"],
      answer: "D"
    },
    {
      question: "The solution x to Ax = b is sensitive to perturbations in b when:",
      options: ["A has low rank", "A is diagonal", "Condition number of A is high", "b is small"],
      answer: "C"
    },
    {
      question: "In sensitivity analysis, if A is ill-conditioned, then a small change in b may cause:",
      options: ["A more accurate solution", "No change at all", "A large change in x", "Matrix A to become singular"],
      answer: "C"
    },
    {
      question: "Sensitivity of the solution x to perturbations in A is governed by:",
      options: ["Frobenius norm only", "The residual vector", "Condition number of A", "Determinant of A"],
      answer: "C"
    },
    {
      question: "The relative error in the solution of Ax = b due to small perturbations in b is approximately bounded by:",
      options: ["The trace of A", "The spectral radius", "cond(A) × relative error in b", "||b||/||x||"],
      answer: "C"
    },
    {
      question: "Which matrix decomposition is typically used to solve Ax = b more efficiently multiple times for different b?",
      options: ["QR decomposition", "LU decomposition", "SVD", "Diagonalization"],
      answer: "B"
    },
    {
      question: "Which step in LU decomposition has the highest computational cost?",
      options: ["Forward substitution", "Backward substitution", "Matrix factorization", "Solving for the determinant"],
      answer: "C"
    },
    {
      question: "The LU decomposition is unique if:",
      options: ["The matrix is orthogonal", "The matrix is positive definite", "The matrix is nonsingular and partial pivoting is not used", "The matrix has distinct eigenvalues"],
      answer: "C"
    },
    {
      question: "A Householder reflector H satisfies which property?",
      options: ["H² = −I", "Hᵀ = H⁻¹", "H = Hᵀ = H⁻¹", "H is always a diagonal matrix"],
      answer: "C"
    },
    {
      question: "The QR decomposition is numerically stable when computed using:",
      options: ["Classical Gram-Schmidt", "Modified Gram-Schmidt", "Cholesky decomposition", "LU decomposition"],
      answer: "B"
    },
    {
      question: "In QR decomposition, the columns of Q are:",
      options: ["Eigenvectors of A", "Orthogonal unit vectors", "The same as columns of A", "The diagonal of A"],
      answer: "B"
    },
    {
      question: "What does Gram-Schmidt orthogonalization guarantee?",
      options: ["Orthogonality but not normalization", "Normalization but not orthogonality", "Both orthogonality and normalization", "None of the above"],
      answer: "C"
    },
    {
      question: "A matrix with condition number close to 1 is:",
      options: ["Ill-conditioned", "Unstable", "Well-conditioned", "Rank-deficient"],
      answer: "C"
    },
    {
      question: "The 2-norm of a matrix A equals:",
      options: ["The maximum row sum", "The largest singular value of A", "The trace of A", "The Frobenius norm"],
      answer: "B"
    },
    {
      question: "The Frobenius norm of a matrix A is:",
      options: ["Square root of the sum of the squares of all elements", "Maximum row sum", "Minimum eigenvalue", "Maximum column norm"],
      answer: "A"
    },
    {
      question: "The sensitivity of the linear system Ax = b to perturbations in b depends mainly on:",
      options: ["Norm of x", "Rank of A", "Condition number of A", "Trace of A"],
      answer: "C"
    },
    {
      question: "If A is ill-conditioned, then solving Ax = b may result in:",
      options: ["More accurate solutions", "Large errors even for small input changes", "A faster algorithm", "Singular solution"],
      answer: "B"
    },
    {
      question: "In numerical linear algebra, a matrix with small determinant is likely to be:",
      options: ["Orthogonal", "Well-conditioned", "Ill-conditioned or nearly singular", "Positive definite"],
      answer: "C"
    },
    {
      question: "In a perturbed system (A + ΔA)x = b + Δb, which term dominates the error in x?",
      options: ["Norm of x", "Condition number of A", "Determinant of A", "Rank of A"],
      answer: "B"
    },
    {
      question: "Which numerical method is especially sensitive to the condition number of a matrix?",
      options: ["Cramer's Rule", "LU Decomposition", "Gauss-Jordan Elimination", "All of the above"],
      answer: "D"
    },
    {
      question: "The main goal of sensitivity analysis is to:",
      options: ["Solve equations faster", "Determine matrix invertibility", "Understand how much error in inputs affects the solution", "Maximize the rank of the matrix"],
      answer: "C"
    },
    {
      question: "Sensitivity analysis is particularly critical in:",
      options: ["Symbolic computation", "Theoretical derivation of linear algebra results", "Real-world numerical applications with measurement error", "Constructing identity matrices"],
      answer: "C"
    },
    {
      question: "Which of the following techniques can reduce sensitivity in solving systems of equations?",
      options: ["Increasing the size of the matrix", "Using matrices with high condition numbers", "Preconditioning or scaling the system", "Ignoring rounding errors"],
      answer: "C"
    },
    {
      question: "Which of the following can cause a system to become sensitive to perturbations?",
      options: ["High sparsity in matrix A", "Orthogonality of the rows in A", "Nearly linearly dependent rows or columns in A", "Having more equations than unknowns"],
      answer: "C"
    },
    {
      question: "Which of the following best describes a well-conditioned system?",
      options: ["Small changes in input cause large changes in the output", "The matrix A is singular", "The determinant of A is zero", "Small changes in input cause small changes in the output"],
      answer: "D"
    },
    {
      question: "The condition number of a matrix gives information about:",
      options: ["The solution to a homogeneous system", "The number of equations in the system", "The sensitivity of the solution with respect to changes in A or b", "The convergence rate of the Jacobi method"],
      answer: "C"
    }
  ],
  unit3: [
    {
      question: "The solution to a linear least squares problem minimizes:",
      options: ["||x||", "||Ax − b||", "||A⁻¹b||", "det(A)"],
      answer: "B"
    },
    {
      question: "The least squares solution exists when:",
      options: ["A is invertible", "AᵀA is invertible", "A is square", "A has full row rank"],
      answer: "B"
    },
    {
      question: "The least squares solution x̂ to Ax = b satisfies:",
      options: ["Ax = b exactly", "AᵀA x̂ = Aᵀb", "A⁻¹x = b", "AᵀA = 0"],
      answer: "B"
    },
    {
      question: "Geometrically, the least squares solution projects b onto:",
      options: ["The null space of A", "The row space of A", "The column space of A", "The orthogonal complement of the column space of A"],
      answer: "C"
    },
    {
      question: "If the columns of A are linearly dependent, then AᵀA is:",
      options: ["Invertible", "Singular", "Orthogonal", "Diagonal"],
      answer: "B"
    },
    {
      question: "In data fitting, least squares is used to:",
      options: ["Maximize accuracy", "Minimize model complexity", "Fit a model by minimizing error between predictions and targets", "Increase bias"],
      answer: "C"
    },
    {
      question: "Adding polynomial features to the input variables is an example of:",
      options: ["Dimensionality reduction", "Feature engineering", "Regularization", "Classification"],
      answer: "B"
    },
    {
      question: "Overfitting in least squares is most likely when:",
      options: ["Too few features", "Data has noise", "Too many features with too little data", "The matrix A is sparse"],
      answer: "C"
    },
    {
      question: "In Vector Auto-Regressive (VAR) models, least squares is used to estimate:",
      options: ["Eigenvalues", "System dynamics from lagged observations", "Probability densities", "Hidden states"],
      answer: "B"
    },
    {
      question: "A continuous piecewise linear fit ensures:",
      options: ["Different linear fits with discontinuities", "A smooth curve that is nonlinear", "Linearity with breakpoints and continuity at joins", "Orthogonal projection"],
      answer: "C"
    },
    {
      question: "Discontinuous piecewise linear fitting may be useful when:",
      options: ["Smooth transitions are required", "Model simplicity is important", "Modeling abrupt changes in data", "Fitting polynomials"],
      answer: "C"
    },
    {
      question: "Least squares can be adapted for classification by:",
      options: ["Using cross-entropy", "Applying it to one-hot encoded labels", "Thresholding the residual", "Only using support vectors"],
      answer: "B"
    },
    {
      question: "Two-class least squares classifier assumes that:",
      options: ["Classes are linearly separable", "Outputs are in {−1, +1} or {0, 1}", "The decision boundary is circular", "Only binary features are allowed"],
      answer: "B"
    },
    {
      question: "In multiclass classification with least squares, each class is represented by:",
      options: ["A cluster", "A separate least squares model", "A unique row in the identity matrix", "A polynomial regressor"],
      answer: "C"
    },
    {
      question: "Polynomial least squares classifiers can handle:",
      options: ["Only linear decision boundaries", "Nonlinear boundaries via higher-order terms", "Binary-only features", "Very low-dimensional data only"],
      answer: "B"
    },
    {
      question: "In the MNIST dataset, each image can be treated as:",
      options: ["A label", "A sparse matrix", "A high-dimensional input vector", "A scalar function"],
      answer: "C"
    },
    {
      question: "Applying least squares to MNIST requires:",
      options: ["Clustering the images", "Converting digits to one-hot vectors", "Computing eigenvectors of the images", "Using convolution layers"],
      answer: "B"
    },
    {
      question: "A limitation of using least squares for MNIST classification is:",
      options: ["It performs poorly on high-dimensional data", "It's not scalable", "It's sensitive to outliers and assumes linear class boundaries", "It requires kernel methods"],
      answer: "C"
    },
    {
      question: "Which condition guarantees a unique least squares solution for Ax = b?",
      options: ["A is square", "A has full column rank", "A has full row rank", "A is diagonal"],
      answer: "B"
    },
    {
      question: "If Aᵀ is not invertible, the least squares solution:",
      options: ["Cannot be computed", "Is always zero", "Can be computed using the pseudo-inverse", "Must be scaled"],
      answer: "C"
    },
    {
      question: "Polynomial feature expansion helps in:",
      options: ["Reducing overfitting", "Creating nonlinear decision boundaries using linear models", "Making data linearly dependent", "Reducing the dimension of data"],
      answer: "B"
    },
    {
      question: "In least squares regression, adding irrelevant features can lead to:",
      options: ["Better accuracy always", "Reduced variance", "Overfitting", "More stable inversion"],
      answer: "C"
    },
    {
      question: "Normalizing features before least squares fitting helps:",
      options: ["Increase model bias", "Improve numerical stability", "Remove outliers", "Make matrix sparse"],
      answer: "B"
    },
    {
      question: "In a VAR model of order p, the number of parameters grows with:",
      options: ["Number of variables × lag order", "Number of observations", "Matrix rank", "Time step size"],
      answer: "A"
    },
    {
      question: "Piecewise linear models are preferred when:",
      options: ["Data is binary", "There are abrupt changes in behavior", "A single line fits well", "Features are sparse"],
      answer: "B"
    },
    {
      question: "A major challenge in piecewise fitting is:",
      options: ["Choosing breakpoints", "Over-normalization", "Full column rank", "Use of regularization"],
      answer: "A"
    },
    {
      question: "Least squares classifiers for two classes predict the label by:",
      options: ["Nearest centroid", "Minimizing residual error for each class", "Taking the sign or argmax of the output", "Averaging over labels"],
      answer: "C"
    },
    {
      question: "A one-vs-all least squares classifier builds:",
      options: ["One model for each pair of classes", "One model for all classes", "One model per class against all others", "A decision tree"],
      answer: "C"
    },
    {
      question: "Which of the following is not a typical concern in least squares classification?",
      options: ["Sensitivity to outliers", "Linear class boundaries", "Softmax scoring", "Overfitting with too many features"],
      answer: "C"
    },
    {
      question: "The input vector for each MNIST image has:",
      options: ["784 features", "256 features", "10 labels", "28 features"],
      answer: "A"
    },
    {
      question: "Least squares is generally not preferred for MNIST because:",
      options: ["It cannot be trained", "It's too slow", "It's not robust to nonlinearity and noise", "It requires backpropagation"],
      answer: "C"
    },
    {
      question: "Least squares classification on MNIST can be improved by:",
      options: ["Removing one-hot labels", "Increasing resolution", "Applying PCA or feature selection", "Reducing training data"],
      answer: "C"
    },
    {
      question: "In terms of performance on MNIST, polynomial classifiers are generally:",
      options: ["Outperformed by deep neural networks", "The most accurate classifiers", "Less interpretable than random forests", "Unusable without image segmentation"],
      answer: "A"
    },
    {
      question: "What is one benefit of applying polynomial classifiers to the MNIST dataset?",
      options: ["Faster inference time", "Simpler model interpretability", "Ability to model complex decision boundaries between digit classes", "Guaranteed 100% accuracy"],
      answer: "C"
    },
    {
      question: "In practice, how can we reduce overfitting in polynomial classifiers applied to MNIST?",
      options: ["By increasing the polynomial degree", "By reducing the training data", "By using regularization techniques (e.g., L2 penalty)", "By removing all nonlinear terms"],
      answer: "C"
    },
    {
      question: "What is a polynomial classifier?",
      options: ["A classifier that only works for linearly separable data", "A classifier that uses polynomial functions of the input features to make predictions", "A classifier that uses only binary decision trees", "A classifier that ignores feature interactions"],
      answer: "B"
    },
    {
      question: "Why might we prefer a polynomial classifier over a linear classifier for image data like MNIST?",
      options: ["Polynomial classifiers are faster to train", "They can capture nonlinear patterns in the data", "They use fewer features", "They avoid overfitting completely"],
      answer: "B"
    },
    {
      question: "What type of data does the MNIST dataset consist of?",
      options: ["3D point cloud data", "Grayscale images of handwritten digits", "Colored images of animals", "Audio recordings of numbers"],
      answer: "B"
    },
    {
      question: "How are polynomial features generated in a polynomial classifier?",
      options: ["By projecting data onto random lower dimensions", "By multiplying input features to create higher-degree terms", "By removing redundant features", "By converting integers to binary"],
      answer: "B"
    },
    {
      question: "What is the main drawback of using high-degree polynomial classifiers on high-dimensional data like MNIST?",
      options: ["They require feature normalization", "They are always underfitting", "They can lead to overfitting and high computational cost", "They convert images to text"],
      answer: "C"
    }
  ],
  unit4: [
    {
      question: "What is the goal of multi-objective least squares optimization?",
      options: ["Minimize a single objective function", "Maximize the determinant of the matrix", "Optimize multiple objective functions simultaneously", "Perform classification over multiple categories"],
      answer: "C"
    },
    {
      question: "A typical approach in multi-objective least squares is to:",
      options: ["Solve each objective independently", "Use Pareto optimality to balance objectives", "Disregard constraints", "Apply Fourier transforms to all objectives"],
      answer: "B"
    },
    {
      question: "Regularized inversion is used in least squares problems to:",
      options: ["Increase overfitting", "Improve the conditioning of ill-posed problems", "Decrease model complexity by ignoring data", "Remove all measurement noise"],
      answer: "B"
    },
    {
      question: "Which of the following is a common regularization technique?",
      options: ["Laplace transform", "Eigen decomposition", "Tikhonov regularization", "Fourier series"],
      answer: "C"
    },
    {
      question: "Regularization in image de-blurring helps to:",
      options: ["Amplify noise", "Eliminate the blur kernel", "Suppress noise and stabilize the solution", "Sharpen images without computation"],
      answer: "C"
    },
    {
      question: "In regularized image de-blurring, the cost function typically includes:",
      options: ["A data term and a noise term", "A fidelity term and a regularization term", "Only the observed blurred image", "Only the prior image"],
      answer: "B"
    },
    {
      question: "Constrained least squares problems involve:",
      options: ["No additional conditions on the solution", "Solving in the frequency domain", "Constraints such as bounds or equality/inequality relations", "Ignoring the residuals"],
      answer: "C"
    },
    {
      question: "In portfolio optimization, constrained least squares can be used to:",
      options: ["Maximize residuals", "Estimate a sparse signal", "Allocate assets under risk and return constraints", "Generate new asset classes"],
      answer: "C"
    },
    {
      question: "The eigenvalue decomposition of a square matrix A is given by:",
      options: ["A = UT", "A = PDP⁻¹", "A = AᵀA", "A = P⁻¹DP"],
      answer: "B"
    },
    {
      question: "The spectral theorem applies to:",
      options: ["Any non-square matrix", "Only lower triangular matrices", "Symmetric matrices", "Diagonalizable matrices with complex eigenvalues only"],
      answer: "C"
    },
    {
      question: "According to the spectral theorem, a real symmetric matrix can be:",
      options: ["Transformed into an upper triangular matrix", "Diagonalized using an orthogonal matrix", "Transposed to find eigenvectors", "Only inverted numerically"],
      answer: "B"
    },
    {
         
      question: "In multi-objective least squares, scalarization refers to:",
      options: ["Scaling down the data", "Converting a multi-objective problem into a single-objective one", "Maximizing multiple objectives", "Ignoring the objectives"],
      answer: "B"
    },
    {
      question: "Weighted sum method in multi-objective LS assigns:",
      options: ["Zero weights to less important terms", "Probabilities to each objective", "Fixed or tunable weights to each objective term", "Random weights to simulate noise"],
      answer: "C"
    },
    {
      question: "In regularized inversion, the regularization term often penalizes:",
      options: ["High variance in the data", "Large values of the solution", "Smoothness or complexity of the model", "Matrix rank"],
      answer: "C"
    },
    {
      question: "Regularized estimation can be interpreted as:",
      options: ["A purely probabilistic model", "An optimization problem with a prior", "A direct inversion of a matrix", "A Fourier-based solution"],
      answer: "B"
    },
    {
      question: "One commonly used prior in image deblurring is:",
      options: ["Gaussian prior on gradients", "Polynomial interpolation", "Lagrange multipliers", "Vandermonde matrix"],
      answer: "A"
    },
    {
      question: "In portfolio optimization, the objective is typically to:",
      options: ["Minimize expected return", "Maximize variance", "Minimize risk for a given return", "Maximize cost"],
      answer: "C"
    },
    {
      question: "A common constraint in portfolio optimization is:",
      options: ["Sum of weights equals zero", "Weights must be negative", "Sum of weights equals one", "Assets must be independent"],
      answer: "C"
    },
    {
      question: "For a real symmetric matrix A, the eigenvectors:",
      options: ["Are always complex", "Are not unique", "Form an orthonormal basis", "Must be non-negative"],
      answer: "C"
    },
    {
      question: "If all eigenvalues of a symmetric matrix are positive, the matrix is:",
      options: ["Singular", "Negative definite", "Positive definite", "Diagonalizable but not invertible"],
      answer: "C"
    },
    {
      question: "The regularization parameter λ in Tikhonov regularization controls:",
      options: ["The scale of the measurements", "The smoothness of the operator", "The trade-off between fitting the data and penalizing the solution", "The size of the input matrix"],
      answer: "C"
    },
    {
      question: "Regularization is especially important when:",
      options: ["The matrix is full rank", "The problem is overdetermined", "The problem is ill-posed or underdetermined", "There are no errors in measurements"],
      answer: "C"
    },
    {
      question: "In image de-blurring, the blur kernel is also known as the:",
      options: ["Prior distribution", "Inverse matrix", "Point Spread Function (PSF)", "Denoising function"],
      answer: "C"
    },
    {
      question: "A commonly used edge-preserving regularizer in image restoration is:",
      options: ["L2 norm", "L0 norm", "Total Variation (TV)", "Kullback-Leibler divergence"],
      answer: "C"
    },
    {
      question: "A quadratic programming approach is commonly used in portfolio optimization because:",
      options: ["The objective function is non-linear", "The objective function is quadratic and constraints are linear", "All solutions are binary", "It's faster than linear programming"],
      answer: "B"
    },
    {
      question: "What is a typical risk measure minimized in portfolio optimization?",
      options: ["The expected return", "The trace of the covariance matrix", "The variance of portfolio returns", "The number of assets"],
      answer: "C"
    },
    {
      question: "Eigenvalue decomposition is not possible when:",
      options: ["The matrix is non-square", "The matrix is symmetric", "All eigenvalues are real", "The matrix is diagonalizable"],
      answer: "A"
    },
    {
      question: "For a symmetric matrix A, which of the following is always true?",
      options: ["All eigenvalues are complex", "All eigenvalues are real", "The matrix is singular", "The matrix has no inverse"],
      answer: "B"
    },
    {
      question: "What are the properties of eigenvalues of a real symmetric matrix?",
      options: ["They are all positive", "They are always complex", "They are always real", "They are imaginary conjugates"],
      answer: "C"
    },
    {
      question: "The Spectral Theorem states that every real symmetric matrix can be:",
      options: ["Reduced to row echelon form", "Diagonalized by an orthogonal matrix", "Converted to a skew-symmetric matrix", "Inverted using Gaussian elimination"],
      answer: "B"
    },
    {
      question: "The columns of the orthogonal matrix Q in the spectral theorem are:",
      options: ["Eigenvectors of A and not necessarily orthonormal", "Arbitrary vectors", "A basis of null space", "Orthonormal eigenvectors of A"],
      answer: "D"
    },
    {
      question: "Which of the following must be true if A is symmetric?",
      options: ["A⁻¹ is also symmetric", "A² is symmetric", "A has real eigenvalues and orthogonal eigenvectors", "A must be diagonal"],
      answer: "C"
    },
    {
      question: "What is the geometric significance of the spectral theorem for symmetric matrices?",
      options: ["The matrix performs a rotation", "The matrix transforms a vector into an orthogonal one", "The matrix can be viewed as scaling along mutually orthogonal directions", "The matrix has no geometric interpretation"],
      answer: "C"
    },
    {
      question: "Which of the following statements is false for real symmetric matrices?",
      options: ["They are always diagonalizable", "Their eigenvectors form an orthonormal basis", "Their eigenvalues are always distinct", "They can be represented as QλQᵀ"],
      answer: "C"
    },
    {
      question: "The spectral theorem allows a symmetric matrix to be interpreted as:",
      options: ["A projection operator", "A unitary transformation", "A rotation and scaling along eigen-directions", "A permutation of eigenvalues"],
      answer: "C"
    },
    {
      question: "What is the main goal of regularized data fitting?",
      options: ["To reduce training time", "To improve matrix inversion accuracy", "To prevent overfitting by penalizing model complexity", "To eliminate all noise in the data"],
      answer: "C"
    },
    {
      question: "In image de-blurring, the observed image is usually modeled as:",
      options: ["The sum of the original image and a Gaussian noise term", "A rotated version of the original image", "The convolution of the original image with a blur kernel plus noise", "A compressed version of the original image"],
      answer: "C"
    },
    {
      question: "What happens if the regularization parameter λ is set too high in image de-blurring?",
      options: ["The image becomes too sharp", "The de-blurred image looks over-smoothed and may lose details", "The noise remains unfiltered", "The blur kernel becomes unknown"],
      answer: "B"
    },
    {
      question: "What type of regularization is commonly used in image de-blurring problems?",
      options: ["Lasso (L1) regularization", "Ridge (L2) regularization", "Elastic Net", "Tikhonov regularization"],
      answer: "D"
    },
    {
      question: "What does Tikhonov regularization add to the optimization problem in image de-blurring?",
      options: ["A constraint on the size of the image", "A penalty on the image gradient", "A term that penalizes large values in the solution", "A normalization term for the input data"],
      answer: "C"
    }
  ],
  unit5: [
    {
      question: "Which of the following is true about the Singular Value Decomposition (SVD)?",
      options: ["SVD decomposes any matrix into a sum of rank-1 matrices", "SVD can only be applied to square matrices", "SVD always results in orthogonal eigenvectors", "SVD decomposes a matrix into U, D, and V where D is skew-symmetric"],
      answer: "A"
    },
    {
      question: "What is the relation between the condition number of a matrix and its singular values?",
      options: ["It is the sum of all singular values", "It is the difference between the largest and smallest singular values", "It is the ratio of the largest to the smallest non-zero singular value", "It is the square of the Frobenius norm"],
      answer: "C"
    },
    {
      question: "A high condition number implies:",
      options: ["The matrix is well-conditioned", "The matrix is symmetric", "The matrix is close to being singular and the problem is ill-conditioned", "The matrix has full rank"],
      answer: "C"
    },
    {
      question: "In least squares problems, sensitivity analysis is used to:",
      options: ["Find exact solutions for overdetermined systems", "Improve the speed of matrix inversion", "Study how small changes in input affect the solution", "Select the number of principal components"],
      answer: "C"
    },
    {
      question: "In regression, multicollinearity causes:",
      options: ["Higher model accuracy", "Instability in parameter estimates", "Faster computation", "A lower condition number"],
      answer: "B"
    },
    {
      question: "Which method is commonly used to address multicollinearity in regression?",
      options: ["Ordinary least squares", "Linear discriminant analysis", "Principal Component Regression (PCR)", "k-means clustering"],
      answer: "C"
    },
    {
      question: "Principal Component Analysis (PCA) is primarily used for:",
      options: ["Feature expansion", "Dimensionality reduction", "Clustering", "Data replication"],
      answer: "B"
    },
    {
      question: "The directions of the principal components in PCA are:",
      options: ["Random linear combinations of variables", "Orthogonal directions that maximize variance", "Based on regression coefficients", "Chosen to minimize classification error"],
      answer: "B"
    },
    {
      question: "The power method is used to:",
      options: ["Solve non-linear equations", "Compute the smallest eigenvalue of a matrix", "Find the dominant eigenvalue and its eigenvector", "Reduce the number of variables in regression"],
      answer: "C"
    },
    {
      question: "Google's PageRank algorithm is most closely related to:",
      options: ["k-means clustering", "Eigenvalue decomposition of the web graph matrix", "Singular value decomposition", "Linear regression"],
      answer: "B"
    },
    {
      question: "In the context of SVD, the diagonal entries of the Σ matrix represent:",
      options: ["The eigenvalues of the original matrix", "The variances of the original features", "The singular values of the matrix", "The coefficients of regression"],
      answer: "C"
    },
    {
      question: "When is the least squares solution most sensitive to noise in data?",
      options: ["When the matrix has full column rank", "When the design matrix has orthogonal columns", "When the design matrix is ill-conditioned", "When all features are independent"],
      answer: "C"
    },
    {
      question: "Which of the following best describes multicollinearity?",
      options: ["Multiple variables are linearly independent", "Two or more independent variables are highly correlated", "The dependent variable is collinear with the independent variables", "The variance of errors is constant"],
      answer: "B"
    },
    {
      question: "Which of the following techniques transforms correlated variables into a smaller number of uncorrelated variables?",
      options: ["Ridge regression", "Principal Component Analysis (PCA)", "Logistic regression", "Lasso regression"],
      answer: "B"
    },
    {
      question: "In PCA, which components are retained for dimensionality reduction?",
      options: ["Those with the highest eigenvalues", "Those with the lowest eigenvalues", "All components are always retained", "Components chosen at random"],
      answer: "A"
    },
    {
      question: "What does a large condition number indicate about a matrix used in regression?",
      options: ["The matrix is diagonal", "The regression coefficients are stable", "The system is robust to small input changes", "The matrix is nearly singular and solution may be unstable"],
      answer: "D"
    },
    {
      question: "The power method converges if:",
      options: ["The matrix is diagonalizable and has a unique largest eigenvalue", "The matrix is non-square", "The matrix is skew-symmetric", "The initial vector is an eigenvector"],
      answer: "A"
    },
    {
      question: "Which step is typically performed first in PCA?",
      options: ["Select the number of clusters", "Normalize the data", "Run logistic regression", "Compute PageRank"],
      answer: "B"
    },
    {
      question: "In PageRank, the importance of a webpage is determined by:",
      options: ["Its HTML structure", "Its content length", "The number and importance of pages linking to it", "The date it was created"],
      answer: "C"
    },
    {
      question: "Why is PCA useful in regression when multicollinearity is present?",
      options: ["It reduces prediction error by increasing complexity", "It uses original variables instead of linear combinations", "It transforms correlated variables into orthogonal components, stabilizing regression", "It removes all independent variables with low variance"],
      answer: "C"
    },
    {
      question: "The matrix V in the SVD of a matrix A (A = UΣVᵀ) is:",
      options: ["An orthogonal matrix of eigenvectors of A", "An orthogonal matrix of eigenvectors of AᵀA", "A matrix of random directions", "A matrix used for clustering"],
      answer: "B"
    },
    {
      question: "A major drawback of using the normal equation (XᵀX)⁻¹Xᵀy in least squares is:",
      options: ["It works only with sparse matrices", "It doesn't minimize residuals", "It becomes unstable when XᵀX is nearly singular", "It cannot be used with large datasets"],
      answer: "C"
    },
    {
      question: "Sensitivity of regression coefficients can be quantified using:",
      options: ["Variance Inflation Factor (VIF)", "Chi-squared test", "AIC", "ROC curve"],
      answer: "A"
    },
    {
      question: "Which of the following reduces multicollinearity without completely eliminating variables?",
      options: ["Forward selection", "Ridge regression", "Backward elimination", "Ordinary Least Squares"],
      answer: "B"
    },
    {
      question: "In PCA, the eigenvectors of the covariance matrix correspond to:",
      options: ["Principal components", "Regression coefficients", "Latent variables in neural networks", "Decision boundaries"],
      answer: "A"
    },
    {
      question: "Dimensionality reduction is useful in machine learning because it:",
      options: ["Increases training time", "Increases the risk of overfitting", "Reduces noise and redundancy in the data", "Ignores variance in the data"],
      answer: "C"
    },
    {
      question: "The eigenvalue corresponding to the dominant eigenvector in the power method converges:",
      options: ["To the smallest eigenvalue", "To the largest eigenvalue in absolute value", "To the mean of all eigenvalues", "To the Frobenius norm of the matrix"],
      answer: "B"
    },
    {
      question: "Which step in the power method ensures convergence to the dominant eigenvector?",
      options: ["Repeated matrix-vector multiplication", "Taking the inverse of the matrix", "Setting all values in the vector to 1", "Computing the determinant"],
      answer: "A"
    },
    {
      question: "In the context of PageRank, the transition matrix is:",
      options: ["Diagonal", "Symmetric", "Stochastic", "Sparse symmetric positive-definite"],
      answer: "C"
    },
    {
      question: "Which of the following ensures the PageRank algorithm doesn't get stuck in dead ends or cycles?",
      options: ["Matrix inversion", "Random surfer model with damping factor", "Linear regression on link data", "Eigen decomposition of identity matrix"],
      answer: "B"
    },
    {
      question: "What is the main idea behind the Google PageRank algorithm?",
      options: ["To count the number of visits to a webpage", "To evaluate the importance of webpages based on their links", "To sort web pages by alphabetical order", "To analyze keyword frequency on web pages"],
      answer: "B"
    },
    {
      question: "Which type of matrix is primarily used to represent the link structure of the web in the PageRank algorithm?",
      options: ["Diagonal matrix", "Covariance matrix", "Adjacency (or hyperlink) matrix", "Identity matrix"],
      answer: "C"
    },
    {
      question: "What property must the Google matrix (used in PageRank) satisfy?",
      options: ["It must be orthogonal", "It must be symmetric", "It must be a stochastic matrix (columns sum to 1)", "It must be positive definite"],
      answer: "C"
    },
    {
      question: "The PageRank vector is the:",
      options: ["Smallest eigenvector of the Google matrix", "Eigenvector corresponding to the eigenvalue 0", "Eigenvector corresponding to the largest eigenvalue (usually 1)", "Transpose of the adjacency matrix"],
      answer: "C"
    },
    {
      question: "Which numerical method is typically used to compute the dominant eigenvector in the PageRank algorithm?",
      options: ["LU Decomposition", "Gauss-Jordan Elimination", "Power Method", "Newton-Raphson Method"],
      answer: "C"
    },
    {
      question: "What does the power method do in the context of the PageRank algorithm?",
      options: ["Finds the inverse of the hyperlink matrix", "Iteratively computes the dominant eigenvector of the Google matrix", "Computes shortest paths in the web graph", "Multiplies matrices to reduce size"],
      answer: "B"
    },
    {
      question: "In the PageRank algorithm, what does a higher value in the PageRank vector indicate about a webpage?",
      options: ["The page has fewer words", "The page has many outgoing links", "The page is more important or influential in the network", "The page loads faster"],
      answer: "C"
    },
    {
      question: "Why is the damping factor important in the PageRank algorithm?",
      options: ["It ensures the adjacency matrix becomes symmetric", "It normalizes the data", "It handles rank sinks and makes the matrix irreducible and aperiodic", "It improves the precision of matrix multiplication"],
      answer: "C"
    },
    {
      question: "In matrix reduction techniques related to PageRank, what is typically reduced or transformed?",
      options: ["Page titles", "The original link matrix to a stochastic matrix", "HTML content", "Eigenvalues to zeros"],
      answer: "B"
    },
    {
      question: "What is the stopping criterion in the power method used in PageRank computation?",
      options: ["When all eigenvalues become zero", "When the adjacency matrix is singular", "When successive PageRank vectors change very little (convergence)", "After exactly 100 iterations"],
      answer: "C"
    }
  ],
  unit6: [
    {
      question: "An underdetermined system of linear equations has:",
      options: ["A unique solution", "No solution", "Fewer equations than unknowns", "Equal number of equations and unknowns"],
      answer: "C"
    },
    {
      question: "The least-norm solution to an underdetermined system minimizes:",
      options: ["The number of non-zero entries in the solution", "The error in predictions", "The Euclidean norm ∥x∥₂ subject to Ax = b", "The determinant of the coefficient matrix"],
      answer: "C"
    },
    {
      question: "In sparse recovery, the goal is to:",
      options: ["Maximize the number of non-zero elements", "Find the solution with the fewest non-zero entries that satisfies the system", "Increase the condition number of the matrix", "Compute eigenvalues of the matrix"],
      answer: "B"
    },
    {
      question: "Which optimization technique is commonly used for sparse solutions in compressed sensing?",
      options: ["L2 norm minimization", "Principal Component Analysis", "L1 norm minimization", "Singular value decomposition"],
      answer: "C"
    },
    {
      question: "Dictionary learning in signal processing is used to:",
      options: ["Reduce data variance", "Identify principal components", "Learn a set of basis vectors for sparse representation of data", "Perform matrix inversion efficiently"],
      answer: "C"
    },
    {
      question: "In sparse coding, each input vector is approximated as:",
      options: ["A sum of random projections", "A sparse linear combination of dictionary atoms", "An eigenvector of the data matrix", "A linear combination with full support"],
      answer: "B"
    },
    {
      question: "The inverse eigenvalue problem seeks to:",
      options: ["Compute eigenvalues of a given matrix", "Find a matrix that has a given set of eigenvalues", "Minimize eigenvectors' angles", "Determine the number of principal components"],
      answer: "B"
    },
    {
      question: "In the context of Markov chains, a stationary distribution π satisfies:",
      options: ["π = Pπᵀ", "πᵀ = πᵀP", "πP = π", "π = π"],
      answer: "C"
    },
    {
      question: "Given a stationary distribution π, a valid transition matrix P for a Markov chain must satisfy:",
      options: ["P is symmetric", "P is stochastic and πP = π", "P is invertible and sparse", "P has all negative entries"],
      answer: "B"
    },
    {
      question: "One approach to constructing a transition matrix P from a stationary distribution π is:",
      options: ["P = ππᵀ", "Use P = D⁻¹A, where A is a symmetric matrix and D is a diagonal matrix with π", "Apply PCA on π", "Minimize the spectral norm of π"],
      answer: "B"
    },
    {
      question: "Which of the following methods is typically used to solve an underdetermined system Ax = b when many solutions exist?",
      options: ["Gradient descent", "Moore-Penrose pseudoinverse", "Eigenvalue decomposition", "LU decomposition"],
      answer: "B"
    },
    {
      question: "In compressed sensing, which property of the measurement matrix helps in exact sparse recovery?",
      options: ["Diagonal dominance", "Restricted Isometry Property (RIP)", "Toeplitz structure", "Low rank"],
      answer: "B"
    },
    {
      question: "Which norm is not typically used to promote sparsity in optimization problems?",
      options: ["L0 norm", "L1 norm", "L2 norm", "Elastic net (L1 + L2)"],
      answer: "C"
    },
    {
      question: "In dictionary learning, the optimization objective usually alternates between:",
      options: ["Data whitening and clustering", "Matrix factorization and eigenvalue updates", "Updating dictionary atoms and sparse coefficients", "Reducing rank and dimension"],
      answer: "C"
    },
    {
      question: "In the inverse eigenvalue problem, which of the following is generally specified?",
      options: ["Only the eigenvectors", "A target matrix and its inverse", "A set of eigenvalues and a structure for the matrix", "The trace of the matrix"],
      answer: "C"
    },
    {
      question: "Which of the following is not a valid constraint on a stochastic matrix for Markov chains?",
      options: ["All rows sum to one", "All elements are non-negative", "The matrix is symmetric", "It represents transition probabilities"],
      answer: "C"
    },
    {
      question: "The Markov chain property that makes it memoryless is known as:",
      options: ["Reversibility", "Stationarity", "Markov property", "Time-homogeneity"],
      answer: "C"
    },
    {
      question: "The stationary distribution of a finite irreducible, aperiodic Markov chain:",
      options: ["Does not exist", "Depends on initial state", "Is unique and independent of the starting distribution", "Changes with time"],
      answer: "C"
    },
    {
      question: "Which application is best modeled using sparse coding?",
      options: ["Image classification using CNNs", "Topic modeling in documents", "Image denoising or compression using learned basis", "Computing exact solutions to linear systems"],
      answer: "C"
    },
    {
      question: "In an underdetermined linear system Ax = b, if multiple solutions exist, which solution is typically preferred in practice?",
      options: ["The one with the highest norm", "The one with the smallest L1 norm", "The one with the smallest L2 norm", "The one with the highest determinant"],
      answer: "C"
    },
    {
      question: "Which of the following best defines sparse coding?",
      options: ["Mapping high-dimensional data to labels", "Representing a signal as a combination of as many atoms as possible", "Learning a set of basis vectors to express data with few active coefficients", "Projecting data onto orthogonal axes"],
      answer: "C"
    },
    {
      question: "The L0 norm counts:",
      options: ["The number of non-zero entries in a vector", "The sum of all vector entries", "The length of a vector", "The square of the Euclidean norm"],
      answer: "A"
    },
    {
      question: "In dictionary learning, the 'dictionary' refers to:",
      options: ["A list of hyperparameters", "A matrix whose columns are basis vectors", "A lookup table of signal values", "A matrix of eigenvectors"],
      answer: "B"
    },
    {
      question: "Solving an inverse eigenvalue problem is essential when:",
      options: ["You know the desired dynamic behavior of a system", "You want to find eigenvectors of a known matrix", "You want to reduce dimensionality", "You perform Gaussian elimination"],
      answer: "A"
    },
    {
      question: "Which of the following would not be a typical objective when solving a sparse recovery problem?",
      options: ["Minimize the number of active coefficients", "Reconstruct the signal exactly", "Minimize L2 norm for maximum sparsity", "Minimize L1 norm for approximate sparsity"],
      answer: "C"
    },
    {
      question: "In Markov chains, which property ensures that a stationary distribution exists and is unique?",
      options: ["Diagonalization of the transition matrix", "Reversibility and symmetry", "Irreducibility and aperiodicity", "Non-negativity of eigenvalues"],
      answer: "C"
    },
    {
      question: "Constructing a Markov chain with a given stationary distribution can involve:",
      options: ["Making the chain symmetric", "Solving the inverse eigenvalue problem", "Designing a stochastic matrix P such that πP = π", "Ensuring all eigenvalues are real"],
      answer: "C"
    },
    {
      question: "What does the sparsity constraint encourage in the solution of an optimization problem?",
      options: ["The solution vector has large values", "The solution vector has many zeros", "The matrix is positive definite", "All vectors lie in the same subspace"],
      answer: "B"
    },
    {
      question: "Which of the following is a typical application of sparse representation in real-world tasks?",
      options: ["Matrix multiplication", "Sorting algorithms", "Face recognition and compressed sensing", "PageRank algorithm"],
      answer: "C"
    },
    {
      question: "In sparse coding, an 'atom' in the dictionary refers to:",
      options: ["A scalar multiplier in the cost function", "A row in the data matrix", "A basis vector used to build signals", "A sample input vector"],
      answer: "C"
    },
    {
      question: "Which algorithm is commonly used for finding sparse solutions in high-dimensional linear systems?",
      options: ["K-means", "LASSO (Least Absolute Shrinkage and Selection Operator)", "Ridge regression", "Gaussian elimination"],
      answer: "B"
    },
    {
      question: "What is a primary challenge in solving the inverse eigenvalue problem?",
      options: ["Too few eigenvalues to work with", "The solution is always non-unique", "Additional structural constraints are often required", "It only applies to diagonal matrices"],
      answer: "C"
    },
    {
      question: "A Markov chain transition matrix P is stochastic if:",
      options: ["All columns sum to 1", "All rows sum to 1", "Its determinant is 1", "It has symmetric eigenvalues"],
      answer: "B"
    },
    {
      question: "Which of the following would most likely result in multiple stationary distributions for a Markov chain?",
      options: ["The matrix is doubly stochastic", "The chain is reducible", "The chain is aperiodic", "The transition matrix is diagonal"],
      answer: "B"
    },
    {
      question: "In compressed sensing, measurements are usually:",
      options: ["More than the number of variables", "Equal to the number of variables", "Fewer than the number of variables", "Unrelated to sparsity"],
      answer: "C"
    },
    {
      question: "Which of these is not a property required in dictionary learning?",
      options: ["Overcompleteness of the dictionary", "Orthogonality of dictionary atoms", "Sparsity of coding coefficients", "Ability to reconstruct input signals"],
      answer: "B"
    },
    {
      question: "In the inverse eigenvalue problem, if the desired matrix must be symmetric, then the eigenvalues must be:",
      options: ["Positive", "Negative", "Real", "Complex"],
      answer: "C"
    },
    {
      question: "What type of matrix is a Markov transition matrix P?",
      options: ["Identity matrix", "Stochastic matrix (rows sum to 1, all entries non-negative)", "Diagonal matrix", "Symmetric matrix"],
      answer: "B"
    },
    {
      question: "What is a stationary distribution in the context of a Markov chain?",
      options: ["A distribution that changes at each time step", "A distribution that the chain converges to over time", "A distribution that maximizes entropy", "The initial distribution of states"],
      answer: "B"
    },
    {
      question: "When constructing a Markov chain with a known stationary distribution π, which of the following must be true for the transition matrix P?",
      options: ["All rows must be equal to π", "All columns must sum to zero", "Each row of P must sum to 1", "All eigenvalues of P must be zero"],
      answer: "C"
    }
  ]
};

function startQuiz() {
  const selectedUnits = Array.from(document.querySelectorAll('input[name="unit"]:checked'))
    .map(el => el.value);

  if (selectedUnits.length === 0) {
    alert("Please select at least one unit.");
    return;
  }

  let allQuestions = [];
  selectedUnits.forEach(unit => {
    allQuestions = allQuestions.concat(questionsByUnit[unit]);
  });

  renderQuestions(allQuestions);
}

function renderQuestions(questions) {
  const quizArea = document.getElementById('quiz-area');
  quizArea.innerHTML = '';

  questions.forEach((q, index) => {
    const qDiv = document.createElement('div');
    qDiv.className = 'question';

    let html = `<p>Q${index + 1}. ${q.question}</p>`;
    q.options.forEach((opt, i) => {
      html += `
        <label>
          <input type="radio" name="q${index}" value="${String.fromCharCode(65 + i)}" />
          ${opt}
        </label>`;
    });

    qDiv.innerHTML = html;
    quizArea.appendChild(qDiv);
  });

  const submitBtn = document.createElement('button');
  submitBtn.innerText = "Submit Quiz";
  submitBtn.onclick = () => evaluateQuiz(questions);
  quizArea.appendChild(submitBtn);
}

function evaluateQuiz(questions) {
  let score = 0;

  questions.forEach((q, index) => {
    const selected = document.querySelector(`input[name="q${index}"]:checked`);
    if (selected && selected.value === q.answer) {
      score++;
    }
  });

  document.getElementById('result').innerText = `You scored ${score} out of ${questions.length}`;
}
