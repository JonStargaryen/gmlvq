package weka.classifiers.functions.gmlvq.model;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import weka.classifiers.functions.gmlvq.utilities.LinearAlgebraicCalculations;

/**
 * The representation of a {@link Vector} in the embedded space. The nature of
 * this mapping is described in the {@link OmegaMatrix}. Their dimensionality is
 * equal to the omega dimension.
 *
 * @author S
 *
 */
public class EmbeddedSpaceVector extends Vector implements Cloneable {

    private static final long serialVersionUID = 1L;

    private Map<List<Prototype>, WinningInformation> winningInformation;
    private OmegaMatrix omegaMatrix;

    public EmbeddedSpaceVector(double[] values, double classLabel, OmegaMatrix omegaMatrix) {
        super(values, classLabel);
        this.omegaMatrix = omegaMatrix;
        this.winningInformation = new HashMap<List<Prototype>, WinningInformation>();
    }

    public EmbeddedSpaceVector(Vector vector, OmegaMatrix omegaMatrix) {
        this(vector.getValues(), vector.getClassLabel(), omegaMatrix);
    }

    public void deregisterAllWinnersBut(List<Prototype> prototypes) {
        if (this.winningInformation.containsKey(prototypes)) {
            WinningInformation value = this.winningInformation.get(prototypes);
            this.winningInformation.clear();
            this.winningInformation.put(prototypes, value);
        } else {
            this.winningInformation.clear();
        }
    }

    private WinningInformation determineWinningInformation(List<Prototype> prototypes) {
        WinningInformation winningInformation = new WinningInformation();
        for (int index = 0; index < prototypes.size(); index++) {
            Prototype prototype = prototypes.get(index);
            double distance = LinearAlgebraicCalculations.calculateSquaredEuclideanDistance(this,
                    prototype.getEmbeddedSpaceVector(this.omegaMatrix));
            if (getClassLabel() == prototype.getClassLabel()) {
                if (distance < winningInformation.getDistanceSameClass()) {
                    winningInformation.setDistanceSameClass(distance);
                    winningInformation.setIndexWinnerSameClass(index);
                }
            } else {
                if (distance < winningInformation.getDistanceOtherClass()) {
                    winningInformation.setDistanceOtherClass(distance);
                    winningInformation.setIndexWinnerOtherClass(index);
                }
            }
        }
        winningInformation.setWinnerSameClass(prototypes.get(winningInformation.getIndexWinnerSameClass()));
        winningInformation.setWinnerOtherClass(prototypes.get(winningInformation.getIndexWinnerOtherClass()));
        return winningInformation;
    }

    public WinningInformation getWinningInformation(List<Prototype> prototypes) {
        if (!this.winningInformation.containsKey(prototypes)) {
            this.winningInformation.put(prototypes, determineWinningInformation(prototypes));
        }
        return this.winningInformation.get(prototypes);
    }

    @Override
    public String toString() {
        return "EmbeddedSpaceVector " + getDimension() + "D " + Arrays.toString(getValues()) + " class = "
                + getClassLabel();
    }

}
