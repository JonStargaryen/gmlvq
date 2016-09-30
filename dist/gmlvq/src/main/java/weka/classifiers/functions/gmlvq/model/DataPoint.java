package weka.classifiers.functions.gmlvq.model;

import java.util.Arrays;
import java.util.List;

import weka.core.Instance;

/**
 * The equivalent to Weks's {@link Instance}: the representation of raw input
 * data.
 *
 * @author S
 *
 */
public class DataPoint extends DataSpaceVector {

    private static final long serialVersionUID = 1L;

    public DataPoint(double[] values, double classLabel) {
        super(values, classLabel);
    }

    @Override
    public void setValues(double[] values) {
        throw new UnsupportedOperationException("this is not allowed for data points, values can never change");
    }

    public void deregisterAllMappingsBut(OmegaMatrix matrix, List<Prototype> prototype) {
        deregisterAllMappingBut(matrix);
        if (this.embeddedSpaceVectors.containsKey(matrix)) {
            this.embeddedSpaceVectors.get(matrix).deregisterAllWinnersBut(prototype);
        }
    }

    @Override
    public String toString() {
        return "DataPoint " + getDimension() + "D " + Arrays.toString(getValues()) + " class = " + getClassLabel()
                + " containing " + getNumberOfMappings() + " mappings to embedded space";
    }

}
