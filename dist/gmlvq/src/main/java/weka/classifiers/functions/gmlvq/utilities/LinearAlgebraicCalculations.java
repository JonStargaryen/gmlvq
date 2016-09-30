package weka.classifiers.functions.gmlvq.utilities;

import java.util.ArrayList;
import java.util.List;

import weka.classifiers.functions.GMLVQ;
import weka.classifiers.functions.gmlvq.model.DataPoint;
import weka.classifiers.functions.gmlvq.model.EmbeddedSpaceVector;
import weka.classifiers.functions.gmlvq.model.Prototype;
import weka.classifiers.functions.gmlvq.model.Vector;
import weka.core.matrix.Matrix;

/**
 * Collection of common calculation based on vectors and double arrays.
 *
 * @author S
 *
 */
public class LinearAlgebraicCalculations {

    /**
     * the numeric cutoff used to limit calculations
     */
    public static final double NUMERIC_CUTOFF = 1E-9;

    /**
     * Index of the minimal value in the {@code double} array returned by
     * {@link #getMinAndMaxValuesFromMatrix(Matrix)}
     */
    public static final int MINIMAL_INDEX = 0;

    /**
     * Index of the maximal value in the {@code double} array returned by
     * {@link #getMinAndMaxValuesFromMatrix(Matrix)}
     */
    public static final int MAXIMAL_INDEX = 1;

    private LinearAlgebraicCalculations() {
    }

    public static Vector substract(Vector subtrahend, Vector minuend) {
        double[] difference = new double[subtrahend.getDimension()];
        for (int attributeIndex = 0; attributeIndex < subtrahend.getDimension(); attributeIndex++) {
            difference[attributeIndex] = subtrahend.getValues()[attributeIndex] - minuend.getValues()[attributeIndex];
        }
        return new Vector(difference, subtrahend.getClassLabel());
    }

    public static Vector add(Vector summand1, Vector summand2) {
        double[] difference = new double[summand1.getDimension()];
        for (int attributeIndex = 0; attributeIndex < summand1.getDimension(); attributeIndex++) {
            difference[attributeIndex] = summand1.getValues()[attributeIndex] + summand2.getValues()[attributeIndex];
        }
        return new Vector(difference, summand1.getClassLabel());
    }

    /**
     * multiplies a vector by a scalar
     *
     * @param vector
     * @param scalar
     * @return the product vector
     */
    public static Vector multiply(Vector vector, double scalar) {
        double[] product = new double[vector.getDimension()];
        for (int attributeIndex = 0; attributeIndex < vector.getDimension(); attributeIndex++) {
            product[attributeIndex] = vector.getValues()[attributeIndex] * scalar;
        }
        return new Vector(product, vector.getClassLabel());
    }

    /**
     * multiplies a vector with a matrix - is used to map a {@link Vector} to an
     * {@link EmbeddedSpaceVector}
     *
     * @param vector
     *            a data point or prototype
     * @param matrix
     *            the mapping rule
     * @return
     */
    public static Vector multiply(Vector vector, Matrix matrix) {
        // TODO: this tends to be slow
        if (!GMLVQ.isRelevanceLearning(matrix)) {
            return vector;
        }
        // acutally, do something when relevance learning occurs
        double[] product = new double[matrix.getRowDimension()];
        for (int rowIndex = 0; rowIndex < matrix.getRowDimension(); rowIndex++) {
            double sum = 0;
            // iterate over the columns of the matrix
            for (int columnIndex = 0; columnIndex < matrix.getColumnDimension(); columnIndex++) {
                // iterate over the vector
                sum += vector.getValues()[columnIndex] * matrix.get(rowIndex, columnIndex);
            }
            product[rowIndex] = sum;
        }
        return new Vector(product, vector.getClassLabel());
    }

    /**
     * calculates the outer product respectively dyadic product of a
     * {@link Vector} with itself
     *
     * @param vector
     * @return a matrix containing the result
     */
    public static Matrix dyadicProduct(Vector vector) {
        int dataDimension = vector.getDimension();
        Matrix productMatrix = new Matrix(dataDimension, dataDimension);
        for (int rowIndex = 0; rowIndex < dataDimension; rowIndex++) {
            for (int columnIndex = rowIndex; columnIndex < dataDimension; columnIndex++) {
                double product = vector.getValues()[rowIndex] * vector.getValues()[columnIndex];
                productMatrix.set(rowIndex, columnIndex, product);
                if (rowIndex != columnIndex) {
                    productMatrix.set(columnIndex, rowIndex, product);
                }
            }
        }

        return productMatrix;
    }

    /**
     * calculates the covariance matrix based on the definition of
     *
     * @see http://www.itl.nist.gov/div898/handbook/pmc/section5/pmc541.htm
     * @param dataPoints
     * @return the covariance matrix
     */
    public static Matrix calculateCovarianceFromMeanVector(List<DataPoint> dataPoints) {

        double[][] data = new double[dataPoints.size()][];
        for (int index = 0; index < data.length; index++) {
            data[index] = dataPoints.get(index).getValues();
        }

        double[] meanVector = createMeanVectorFromListOfVectors(dataPoints);

        Matrix covarianceMatrix = new Matrix(meanVector.length, meanVector.length);

        for (int i = 0; i < data.length; i++) {

            double[] dataRowVector = data[i];

            Matrix x = new Matrix(substract(new Vector(dataRowVector, 0.0), new Vector(meanVector, 0.0)).getValues(),
                    1);

            covarianceMatrix.plusEquals(x.transpose().times(x));
        }

        return covarianceMatrix.times(1.0 / (dataPoints.size() - 1));
    }

    /**
     * condensed addition of one vector with another vector multiplied by a
     * scalar - used e.g. to calculate the updated prototypes
     *
     * @param originalInstance
     *            the original prototypes
     * @param factor
     *            the prototype learning rate
     * @param delta
     *            the prototype delta
     * @return the proposed prototype update
     */
    public static Vector scaledTranslate(Vector originalInstance, double factor, Vector delta) {
        return add(originalInstance, multiply(delta, factor));
    }

    /**
     * calculates the <i>squared</i> euclidean distance between 2 double
     * vectors/arrays of the same dimension<br />
     * this is the some what low-level function used to find nearest prototypes
     * to a given data point
     *
     * @param frist
     *            the first vector
     * @param second
     *            the second vector
     * @return the squared distance between both entities as primitive double
     *         value
     */
    public static double calculateSquaredEuclideanDistance(Vector frist, Vector second) {
        double sum = 0;
        for (int attributeIndex = 0; attributeIndex < frist.getValues().length; attributeIndex++) {
            double rawValue = frist.getValues()[attributeIndex] - second.getValues()[attributeIndex];
            sum += rawValue * rawValue;
        }
        return sum;
    }

    /**
     * used to normalize the prototype update
     *
     * @param prototypes
     */
    public static void normalizeWithMaximalValue(List<Prototype> prototypes) {
        // calculate factor
        double prototypeSum = 0;
        for (Vector prototype : prototypes) {
            for (double d : prototype.getValues()) {
                prototypeSum += d * d;
            }
        }
        double prototypeNormalizationFactor = 1 / Math.max(prototypeSum, NUMERIC_CUTOFF);
        // apply
        for (Vector prototype : prototypes) {
            prototype.setValues(multiply(prototype, prototypeNormalizationFactor).getValues());
        }
    }

    /**
     * used to normalized the omega update
     *
     * @param matrix
     */
    public static void normalizeWithMaximalValue(Matrix matrix) {
        // calculate factor
        double omegaMatrixSum = 0;
        for (double d : matrix.getRowPackedCopy()) {
            omegaMatrixSum += d * d;
        }
        // apply
        double omegaMatrixNormalizationFactor = 1 / Math.max(omegaMatrixSum, NUMERIC_CUTOFF);
        matrix.timesEquals(omegaMatrixNormalizationFactor);
    }

    /**
     * computes the average vector of a set of vectors - each feature is set to
     * the average of these feature for all input data points
     *
     * @param datapoints
     * @return
     */
    public static double[] createMeanVectorFromListOfVectors(List<DataPoint> datapoints) {
        int dimension = datapoints.get(0).getDimension();
        double[] meanVector = new double[dimension];
        for (DataPoint datapoint : datapoints) {
            for (int attributeIndex = 0; attributeIndex < dimension; attributeIndex++) {
                meanVector[attributeIndex] += datapoint.getValue(attributeIndex);
            }
        }
        for (int i = 0; i < meanVector.length; i++) {
            meanVector[i] /= datapoints.size();
        }
        return meanVector;
    }

    /**
     * return all data points with a requested class label
     * 
     * @param datapoints
     *            what data points to process?
     * @param classLabel
     *            what class label are we interested in?
     * @return all data points with the requested class label
     */
    public static List<DataPoint> collectDatapointsWithClassLabel(List<DataPoint> datapoints, double classLabel) {
        List<DataPoint> collectedDataPoints = new ArrayList<DataPoint>();
        for (DataPoint dataPoint : datapoints) {
            if (dataPoint.getClassLabel() == classLabel) {
                collectedDataPoints.add(dataPoint);
            }
        }
        return collectedDataPoints;
    }

    /**
     * Retrieves the minimal and maximal values in the given matrix at the same
     * time, traversing every value only once. <br/>
     * The returned array contains the minimal values at {@link #MINIMAL_INDEX}
     * = {@value #MINIMAL_INDEX} and the maximal value at {@link #MAXIMAL_INDEX}
     * = {@value #MAXIMAL_INDEX}.
     *
     * @param matrix
     *            The matrix to retrieve minimal and maximal value from.
     * @return An array with two entries containing minimal and maximal values.
     */
    public static double[] getMinAndMaxValuesFromMatrix(Matrix matrix) {

        double[] minAndMaxValue = new double[2];
        minAndMaxValue[MINIMAL_INDEX] = Double.MAX_VALUE;
        minAndMaxValue[MAXIMAL_INDEX] = -Double.MAX_VALUE;

        for (int rowIndex = 0; rowIndex < matrix.getRowDimension(); rowIndex++) {
            for (int columnIndex = 0; columnIndex < matrix.getColumnDimension(); columnIndex++) {
                double currentValue = matrix.get(rowIndex, columnIndex);
                if (currentValue < minAndMaxValue[MINIMAL_INDEX]) {
                    minAndMaxValue[MINIMAL_INDEX] = currentValue;
                } else if (currentValue > minAndMaxValue[MAXIMAL_INDEX]) {
                    minAndMaxValue[MAXIMAL_INDEX] = currentValue;
                }
            }
        }

        return minAndMaxValue;
    }
}
