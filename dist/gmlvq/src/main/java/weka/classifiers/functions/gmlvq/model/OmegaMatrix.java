package weka.classifiers.functions.gmlvq.model;

import java.util.Arrays;

import weka.core.matrix.Matrix;

/**
 * The matrix describing the mapping rule which is used to project
 * {@link DataPoint}s and {@link Prototype}s to their respective
 * {@link EmbeddedSpaceVector}.<br />
 * This matrix is of dimension <code>dataDimension x omegaDimension</code>. It
 * is updated over the course of learning and can indirectly be visualized via
 * the lambda matrix, by evaluating the expression
 * <code>lambda = omega * omega'</code>.
 *
 * @author S
 *
 */
public class OmegaMatrix extends Matrix {

    private static final long serialVersionUID = 1L;

    public OmegaMatrix(double[][] A) {
        super(A);
    }

    public OmegaMatrix(Matrix matrix) {
        this(matrix.getArray());
    }

    @Override
    public int hashCode() {
        return Arrays.deepHashCode(getArray());
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) {
            return true;
        }
        if (obj == null) {
            return false;
        }
        if (getClass() != obj.getClass()) {
            return false;
        }
        OmegaMatrix other = (OmegaMatrix) obj;
        if (!Arrays.deepEquals(this.getArray(), other.getArray())) {
            return false;
        }
        return true;
    }

    @Override
    public String toString() {
        StringBuffer resultString = new StringBuffer();
        // assemble string
        for (int rowIndex = 0; rowIndex < this.getRowDimension(); rowIndex++) {
            for (int columnIndex = 0; columnIndex < this.getColumnDimension(); columnIndex++) {
                resultString
                        .append(String.format(this.get(rowIndex, columnIndex) < 0 ? "%.3f" : " %.3f",
                                this.get(rowIndex, columnIndex)))
                        .append(columnIndex == this.getColumnDimension() - 1 ? "\n" : " ");
            }
        }
        return resultString.toString();
    }

}
