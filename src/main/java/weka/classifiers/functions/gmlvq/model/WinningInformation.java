package weka.classifiers.functions.gmlvq.model;

import java.io.Serializable;

/**
 * Contains information about the closest prototypes regarding to the parent
 * {@link EmbeddedSpaceVector}. The closest prototype with the same class is
 * tracked. Same goes for the closest one with any different class. For them
 * both, the distances are also provided. These values are utilized to compose
 * the updates and to evaluate the costs of any particular proposed update.
 *
 * @author S
 *
 */
public class WinningInformation implements Serializable {

    private static final long serialVersionUID = 1L;

    private int indexWinnerSameClass = -1;
    private Prototype winnerSameClass;
    private double distanceSameClass = Double.MAX_VALUE;
    private Prototype winnerOtherClass;
    private double distanceOtherClass = Double.MAX_VALUE;
    private int indexWinnerOtherClass = -1;

    public double getDistanceOtherClass() {
        return this.distanceOtherClass;
    }

    public double getDistanceSameClass() {
        return this.distanceSameClass;
    }

    public int getIndexWinnerOtherClass() {
        return this.indexWinnerOtherClass;
    }

    public int getIndexWinnerSameClass() {
        return this.indexWinnerSameClass;
    }

    public Prototype getWinnerOtherClass() {
        return this.winnerOtherClass;
    }

    public Prototype getWinnerSameClass() {
        return this.winnerSameClass;
    }

    public void setDistanceOtherClass(double distanceOtherClass) {
        this.distanceOtherClass = distanceOtherClass;
    }

    public void setDistanceSameClass(double distanceSameClass) {
        this.distanceSameClass = distanceSameClass;
    }

    public void setIndexWinnerOtherClass(int indexWinnerOtherClass) {
        this.indexWinnerOtherClass = indexWinnerOtherClass;
    }

    public void setIndexWinnerSameClass(int indexWinnerSameClass) {
        this.indexWinnerSameClass = indexWinnerSameClass;
    }

    public void setWinnerOtherClass(Prototype winnerOtherClass) {
        this.winnerOtherClass = winnerOtherClass;
    }

    public void setWinnerSameClass(Prototype winnerSameClass) {
        this.winnerSameClass = winnerSameClass;
    }

    @Override
    public String toString() {
        return "WinningInformation [winnerSameClass=" + this.winnerSameClass + ", distanceSameClass="
                + this.distanceSameClass + ", distanceOtherClass=" + this.distanceOtherClass + "]";
    }

}
