package weka.classifiers.functions.gmlvq.model;

import java.util.Arrays;

/**
 * More or less only existing to increase readability. Otherwise quite like a
 * {@link DataPoint}.
 *
 * @author S
 *
 */
public class Prototype extends DataSpaceVector {

    private static final long serialVersionUID = 1L;

    public Prototype(double[] values, double classLabel) {
        super(values, classLabel);
    }

    public Prototype(Vector vector) {
        this(Arrays.copyOf(vector.getValues(), vector.getDimension()), vector.getClassLabel());
    }

    @Override
    public void setValues(double[] values) {
        setValues(values);
    }

    @Override
    public synchronized EmbeddedSpaceVector getEmbeddedSpaceVector(OmegaMatrix matrix) {
        return super.getEmbeddedSpaceVector(matrix);
    }

    @Override
    public String toString() {
        return "Prototype " + getDimension() + "D " + Arrays.toString(getValues()) + " class = " + getClassLabel();
    }
}
