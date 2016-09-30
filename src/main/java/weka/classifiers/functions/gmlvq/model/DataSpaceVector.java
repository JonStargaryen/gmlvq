package weka.classifiers.functions.gmlvq.model;

import java.util.HashMap;
import java.util.Map;

import weka.classifiers.functions.gmlvq.utilities.LinearAlgebraicCalculations;

/**
 * A {@link Vector} living in the data space (as opposed to the embedded space).
 * Their dimensionality is equal to the data dimension.
 * 
 * @author S
 *
 */
public abstract class DataSpaceVector extends Vector {

    private static final long serialVersionUID = 1L;
    protected Map<OmegaMatrix, EmbeddedSpaceVector> embeddedSpaceVectors;

    public DataSpaceVector(double[] values, double classLabel) {
        super(values, classLabel);
        this.embeddedSpaceVectors = new HashMap<OmegaMatrix, EmbeddedSpaceVector>();
    }

    public EmbeddedSpaceVector getEmbeddedSpaceVector(OmegaMatrix matrix) {
        if (!this.embeddedSpaceVectors.containsKey(matrix)) {
            this.embeddedSpaceVectors.put(matrix, determineMapping(matrix));
        }
        return this.embeddedSpaceVectors.get(matrix);
    }

    private EmbeddedSpaceVector determineMapping(OmegaMatrix matrix) {
        return new EmbeddedSpaceVector(LinearAlgebraicCalculations.multiply(this, matrix), matrix);
    }

    public int getNumberOfMappings() {
        return this.embeddedSpaceVectors.size();
    }

    public void deregisterAllMappingBut(OmegaMatrix matrix) {
        if (this.embeddedSpaceVectors.containsKey(matrix)) {
            EmbeddedSpaceVector value = this.embeddedSpaceVectors.get(matrix);
            this.embeddedSpaceVectors.clear();
            this.embeddedSpaceVectors.put(matrix, value);
        } else {
            this.embeddedSpaceVectors.clear();
        }
    }

}
