package weka.classifiers.functions.gmlvq.core.cost;

import java.io.Serializable;
import java.util.List;

import weka.classifiers.functions.gmlvq.core.SigmoidFunction;
import weka.classifiers.functions.gmlvq.model.DataPoint;
import weka.classifiers.functions.gmlvq.model.OmegaMatrix;
import weka.classifiers.functions.gmlvq.model.Prototype;
import weka.classifiers.functions.gmlvq.model.WinningInformation;

/**
 * Provides the implementation of any confusion matrix based cost function.
 * These costs can only be computed for cases where 2 classes are present. Each
 * cost function is interlinked to an update rule. TODO: implement safety net
 * for non-2-class-problems (safety = mucho exceptiano)
 *
 * @author S
 *
 */
public class ConfusionMatrix implements Serializable {

    private static final long serialVersionUID = 1L;
    public static final double POSITIVE_CLASS_LABEL = 0.0;
    public static final double NEGATIVE_CLASS_LABEL = 1.0;

    private double truePositiveApprox;
    private double trueNegativeApprox;
    private double falsePositiveApprox;
    private double falseNegativeApprox;

    private SigmoidFunction sigmoidFunction;

    public ConfusionMatrix(SigmoidFunction sigmoidFunction, List<DataPoint> chosenDataPoints,
            List<Prototype> prototypes, OmegaMatrix omegaMatrix) {
        this.sigmoidFunction = sigmoidFunction;
        computeConfusionMatrix(chosenDataPoints, prototypes, omegaMatrix);
    }

    public double computeWeightedAccuracy(double truePositiveWeight, double trueNegativeWeight) {
        return this.truePositiveApprox * truePositiveWeight + this.trueNegativeApprox * trueNegativeWeight;
    }

    public double computeWeightedAccuracyUpdate(DataPoint dataPoint, double truePositiveWeight,
            double trueNegativeWeight) {
        double kroneckerDelta = determineKroneckerDelta(dataPoint);
        return truePositiveWeight * kroneckerDelta + trueNegativeWeight * (1 - kroneckerDelta);
    }

    public double computePrecisionRecall(double precisionWeight, double recallWeight) {
        double precision = this.truePositiveApprox / (this.truePositiveApprox + this.falsePositiveApprox);
        double recall = this.truePositiveApprox / (this.truePositiveApprox + this.falseNegativeApprox);
        return precisionWeight * precision + recallWeight * recall;
    }

    public double computePrecisionRecallUpdate(DataPoint dataPoint, double precisionWeight, double recallWeight) {
        double kroneckerDelta = determineKroneckerDelta(dataPoint);
        double precisionTerm = precisionWeight
                * (kroneckerDelta * this.falsePositiveApprox + (1 - kroneckerDelta) * this.truePositiveApprox)
                / Math.pow(this.truePositiveApprox + this.falsePositiveApprox, 2);
        double recallTerm = recallWeight
                * (kroneckerDelta * this.falseNegativeApprox + kroneckerDelta + this.truePositiveApprox)
                / Math.pow(this.truePositiveApprox + this.falsePositiveApprox, 2);
        return precisionTerm + recallTerm;
    }

    public double computeFMeasure(double beta) {
        return (1 + beta) * this.truePositiveApprox
                / ((1 + beta) * this.truePositiveApprox + beta * this.falseNegativeApprox + this.falsePositiveApprox);
    }

    /**
     * @return 1.0 if this dataPoint's class label is that of the 'positive'
     *         class and 0.0 otherwise
     */
    private static double determineKroneckerDelta(DataPoint dataPoint) {
        return dataPoint.getClassLabel() == POSITIVE_CLASS_LABEL ? 1.0 : 0.0;
    }

    public double computeFMeasureUpdate(DataPoint dataPoint, double beta) {
        double kroneckerDelta = determineKroneckerDelta(dataPoint);
        double t1 = (1 + beta) / Math.pow(
                (1 + beta) * this.truePositiveApprox + beta * this.falseNegativeApprox + this.falsePositiveApprox, 2);
        double t2 = kroneckerDelta
                * (beta * this.falseNegativeApprox + this.falsePositiveApprox + (beta - 1) * this.truePositiveApprox)
                + this.truePositiveApprox;
        return t1 * t2;
    }

    public double getTruePositiveApprox() {
        return this.truePositiveApprox;
    }

    public double getTrueNegativeApprox() {
        return this.trueNegativeApprox;
    }

    public double getFalsePositiveApprox() {
        return this.falsePositiveApprox;
    }

    public double getFalseNegativeApprox() {
        return this.falseNegativeApprox;
    }

    public SigmoidFunction getSigmoidFunction() {
        return this.sigmoidFunction;
    }

    @Override
    public String toString() {
        return "approximated ConfusionMatrix\n\t[TP=" + this.truePositiveApprox + ",FN=" + this.falseNegativeApprox
                + "\n\tFP=" + this.falsePositiveApprox + ",TN=" + this.trueNegativeApprox
                + "]"/*
                      * \n" + "real ConfusionMatrix\n\t[TP=" +
                      * this.truePositiveCount + ",FN=" +
                      * this.falseNegativeCount + "\n\tFP=" +
                      * this.falsePositiveCount + ",TN=" +
                      * this.trueNegativeCount + "]"
                      */;
    }

    private void computeConfusionMatrix(List<DataPoint> chosenDataPoints, List<Prototype> prototypes,
            OmegaMatrix omegaMatrix) {

        int numberOfPositiveInstances = 0;
        for (DataPoint dataPoint : chosenDataPoints) {
            if (dataPoint.getClassLabel() == POSITIVE_CLASS_LABEL) {
                numberOfPositiveInstances++;
            }
        }
        int numberOfNegativeInstance = chosenDataPoints.size() - numberOfPositiveInstances;

        for (DataPoint dataPoint : chosenDataPoints) {
            evaluateDataPoint(dataPoint, prototypes, omegaMatrix);
        }

        normalizeValues(numberOfPositiveInstances, numberOfNegativeInstance);
    }

    private void normalizeValues(int numberOfPositiveInstances, int numberOfNegativeInstance) {
        if (numberOfPositiveInstances > 0) {
            this.truePositiveApprox /= numberOfPositiveInstances;
            this.falseNegativeApprox /= numberOfPositiveInstances;
        }

        if (numberOfNegativeInstance > 0) {
            this.trueNegativeApprox /= numberOfNegativeInstance;
            this.falsePositiveApprox /= numberOfNegativeInstance;
        }
    }

    /**
     * adds this data point to the confusion matrix
     *
     * @param dataPoint
     *            the chosen data point
     * @param prototypes
     *            the prototypes
     * @param omegaMatrix
     *            the mapping rule
     * @return true if this data point was a representative of the positive
     *         class (so we can count them)
     */
    private void evaluateDataPoint(DataPoint dataPoint, List<Prototype> prototypes, OmegaMatrix omegaMatrix) {
        WinningInformation winningInformation = dataPoint.getEmbeddedSpaceVector(omegaMatrix)
                .getWinningInformation(prototypes);
        boolean correctlyClassified = winningInformation.getDistanceSameClass() < winningInformation
                .getDistanceOtherClass();

        double fmu = this.sigmoidFunction
                .evaluate((winningInformation.getDistanceOtherClass() - winningInformation.getDistanceSameClass())
                        / (winningInformation.getDistanceSameClass() + winningInformation.getDistanceOtherClass()));

        if (dataPoint.getClassLabel() == POSITIVE_CLASS_LABEL) {
            if (correctlyClassified) {
                this.truePositiveApprox += fmu;
            } else {
                this.falseNegativeApprox += 1 - fmu;
            }
        } else {
            if (correctlyClassified) {
                this.trueNegativeApprox += fmu;
            } else {
                this.falsePositiveApprox += 1 - fmu;
            }
        }
    }

}
