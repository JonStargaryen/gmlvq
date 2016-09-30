package weka.classifiers.functions.gmlvq.model;

import java.io.Serializable;
import java.util.Arrays;

/**
 * GMLVQ's internal data structure. Each vector contains values as
 * <code>double[]</code> and its class label as primitive double.
 *
 * @author S
 *
 */
public class Vector implements Serializable {

    private static final long serialVersionUID = 1L;

    private double[] values;
    private double classLabel;
    private int dimension;

    public Vector(double[] values, double classLabel) {
        this.values = values;
        this.dimension = values.length;
        this.classLabel = classLabel;
    }

    public int getDimension() {
        return this.dimension;
    }

    public double[] getValues() {
        return this.values;
    }

    public void setValues(Vector vector) {
        setValues(vector.getValues());
    }

    public double getValue(int index) {
        return this.values[index];
    }

    public void setValues(double[] values) {
        this.values = values;
        this.dimension = values.length;
    }

    public double getClassLabel() {
        return this.classLabel;
    }

    public void setClassLabel(double classLabel) {
        this.classLabel = classLabel;
    }

    @Override
    public int hashCode() {
        final int prime = 31;
        int result = 1;
        long temp;
        temp = Double.doubleToLongBits(this.classLabel);
        result = prime * result + (int) (temp ^ temp >>> 32);
        result = prime * result + Arrays.hashCode(this.values);
        return result;
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
        Vector other = (Vector) obj;
        if (Double.doubleToLongBits(this.classLabel) != Double.doubleToLongBits(other.classLabel)) {
            return false;
        }
        if (!Arrays.equals(this.values, other.values)) {
            return false;
        }
        return true;
    }

    @Override
    public String toString() {
        return "Vector " + this.dimension + "D " + Arrays.toString(this.values) + " class = " + this.classLabel;
    }

}
