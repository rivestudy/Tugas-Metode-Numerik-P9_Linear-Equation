public class LinearEquationSolver {
    public static void main(String[] args) {
        double[][] A = {
                {1, -1, 2},
                {3, 0, 1},
                {1, 0, 2}
        };
        double[] b = {5, 10, 5};

        Matrix matrixA = new Matrix(A);
        Vector vectorB = new Vector(b);

        // Solve using LU Gauss method
        Vector solutionLU = solveLU_Gauss(matrixA, vectorB);
        System.out.println("Solution using LU Gauss:");
        System.out.println(solutionLU);

        // Solve using matrix inverse method
        Vector solutionInverse = solveInverse(matrixA, vectorB);
        System.out.println("Solution using Matrix Inverse:");
        System.out.println(solutionInverse);

        // Solve using Crout method
        Vector solutionCrout = solveCrout(matrixA, vectorB);
        System.out.println("Solution using Crout:");
        System.out.println(solutionCrout);
    }

    public static Vector solveLU_Gauss(Matrix A, Vector b) {
        int n = A.getRows();
        Vector x = new Vector(n);

        for (int i = 0; i < n; i++) {
            for (int k = i + 1; k < n; k++) {
                double factor = A.get(k, i) / A.get(i, i);
                for (int j = i; j < n; j++) {
                    A.set(k, j, A.get(k, j) - factor * A.get(i, j));
                }
                b.set(k, b.get(k) - factor * b.get(i));
            }
        }

        for (int i = n - 1; i >= 0; i--) {
            double sum = 0;
            for (int j = i + 1; j < n; j++) {
                sum += A.get(i, j) * x.get(j);
            }
            x.set(i, (b.get(i) - sum) / A.get(i, i));
        }

        return x;
    }

    public static Vector solveInverse(Matrix A, Vector b) {
        Matrix inverse = A.inverse();
        return Vector.multiply(inverse, b);
    }

    public static Vector solveCrout(Matrix A, Vector b) {
        int n = A.getRows();
        double[] x = new double[n];
        double[] y = new double[n];

        double[][] L = new double[n][n];
        double[][] U = new double[n][n];

        for (int i = 0; i < n; i++) {
            L[i][i] = 1;
        }

        for (int i = 0; i < n; i++) {
            U[i][i] = A.get(i, i);
            for (int k = i + 1; k < n; k++) {
                U[i][k] = A.get(i, k);
                L[k][i] = A.get(k, i) / U[i][i];
            }
            for (int j = i + 1; j < n; j++) {
                for (int k = i + 1; k < n; k++) {
                    A.set(j, k, A.get(j, k) - L[j][i] * U[i][k]);
                }
            }
        }

        y[0] = b.get(0) / L[0][0];
        for (int i = 1; i < n; i++) {
            double sum = 0;
            for (int j = 0; j < i; j++) {
                sum += L[i][j] * y[j];
            }
            y[i] = (b.get(i) - sum) / L[i][i];
        }

        x[n - 1] = y[n - 1] / U[n - 1][n - 1];
        for (int i = n - 2; i >= 0; i--) {
            double sum = 0;
            for (int j = i + 1; j < n; j++) {
                sum += U[i][j] * x[j];
            }
            x[i] = (y[i] - sum) / U[i][i];
        }

        Vector solution = new Vector(x);
        return solution;
    }
}

class Matrix {
    private final int rows;
    private final int columns;
    private final double[][] data;

    public Matrix(int rows, int columns) {
        this.rows = rows;
        this.columns = columns;
        this.data = new double[rows][columns];
    }

    public Matrix(double[][] data) {
        this.rows = data.length;
        this.columns = data[0].length;
        this.data = data;
    }

    public int getRows() {
        return rows;
    }

    public int getColumns() {
        return columns;
    }

    public double get(int i, int j) {
        return data[i][j];
    }

    public void set(int i, int j, double value) {
        data[i][j] = value;
    }

    public static Matrix identity(int size) {
        Matrix result = new Matrix(size, size);
        for (int i = 0; i < size; i++) {
            result.set(i, i, 1);
        }
        return result;
    }

    public Matrix multiply(Matrix other) {
        if (columns != other.rows) {
            throw new IllegalArgumentException("Matrices cannot be multiplied");
        }
        Matrix result = new Matrix(rows, other.columns);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < other.columns; j++) {
                double sum = 0;
                for (int k = 0; k < columns; k++) {
                    sum += data[i][k] * other.data[k][j];
                }
                result.data[i][j] = sum;
            }
        }
        return result;
    }

    public Matrix inverse() {
        if (rows != columns) {
            throw new UnsupportedOperationException("Matrix is not square");
        }
        Matrix identity = Matrix.identity(rows);
        Matrix augmented = augment(identity);
        gaussJordanElimination(augmented);
        Matrix inverse = new Matrix(rows, columns);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                inverse.set(i, j, augmented.get(i, j + columns));
            }
        }
        return inverse;
    }

    private void gaussJordanElimination(Matrix augmentedMatrix) {
        int lead = 0;
        int rowCount = augmentedMatrix.rows;
        int colCount = augmentedMatrix.columns;
        for (int r = 0; r < rowCount; r++) {
            if (colCount <= lead) {
                return;
            }
            int i = r;
            while (augmentedMatrix.get(i, lead) == 0) {
                i++;
                if (rowCount == i) {
                    i = r;
                    lead++;
                    if (colCount == lead) {
                        return;
                    }
                }
            }
            double[] temp = augmentedMatrix.data[r];
            augmentedMatrix.data[r] = augmentedMatrix.data[i];
            augmentedMatrix.data[i] = temp;

            double lv = augmentedMatrix.get(r, lead);
            for (int j = 0; j < colCount; j++) {
                augmentedMatrix.data[r][j] /= lv;
            }

            for (i = 0; i < rowCount; i++) {
                if (i != r) {
                    double[] subtractedRow = new double[colCount];
                    double factor = augmentedMatrix.get(i, lead);
                    for (int j = 0; j < colCount; j++) {
                        subtractedRow[j] = augmentedMatrix.get(r, j) * factor;
                    }
                    for (int j = 0; j < colCount; j++) {
                        augmentedMatrix.data[i][j] -= subtractedRow[j];
                    }
                }
            }
            lead++;
        }
    }

    private Matrix augment(Matrix other) {
        if (rows != other.rows) {
            throw new IllegalArgumentException("Matrices have different number of rows");
        }
        double[][] resultData = new double[rows][columns + other.columns];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                resultData[i][j] = data[i][j];
            }
            for (int j = 0; j < other.columns; j++) {
                resultData[i][columns + j] = other.data[i][j];
            }
        }
        return new Matrix(resultData);
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                sb.append(data[i][j]).append("\t");
            }
            sb.append("\n");
        }
        return sb.toString();
    }
}

class Vector {
    private final int size;
    private final double[] data;

    public Vector(int size) {
        this.size = size;
        this.data = new double[size];
    }

    public Vector(double[] data) {
        this.size = data.length;
        this.data = data;
    }

    public int getSize() {
        return size;
    }

    public double get(int i) {
        return data[i];
    }

    public void set(int i, double value) {
        data[i] = value;
    }

    public static Vector multiply(Matrix matrix, Vector vector) {
        if (matrix.getColumns() != vector.getSize()) {
            throw new IllegalArgumentException("Matrix and vector dimensions don't match");
        }
        Vector result = new Vector(matrix.getRows());
        for (int i = 0; i < matrix.getRows(); i++) {
            double sum = 0;
            for (int j = 0; j < matrix.getColumns(); j++) {
                sum += matrix.get(i, j) * vector.get(j);
            }
            result.set(i, sum);
        }
        return result;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < size; i++) {
            sb.append(data[i]).append("\n");
        }
        return sb.toString();
    }
}
