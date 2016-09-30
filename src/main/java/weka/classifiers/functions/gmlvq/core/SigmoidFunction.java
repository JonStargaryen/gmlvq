package weka.classifiers.functions.gmlvq.core;

import java.io.Serializable;

import weka.classifiers.functions.gmlvq.core.cost.DefaultCostFunction;

/**
 * GMLVQ utilizes a sigmoid/Heaviside function to scale e.g. the costs within
 * {@link DefaultCostFunction}. This function depends on a internal value called
 * sigmoid sigma which will increase over the course of learning. This parameter
 * will determine the exact shape of the Heaviside function. Furthermore, an
 * interval has to be defined which specifies where the sigmoid sigma starts and
 * which value it will adapt once the learning process is finished. Within this
 * interval the sigma changes after each epoch and is increased in a logarithmic
 * manner.
 *
 * @author S
 *
 */
public class SigmoidFunction implements Serializable {

    private static final long serialVersionUID = 1L;

    /**
     * keeps track of the sigmoid sigma value as it increases over the course of
     * learning
     */
    private double currentSigmoidSigma;
    /**
     * the lower bound given by the user's input
     */
    private final double sigmoidSigmaIntervalStart;
    /**
     * the upper bound of the sigmoid sigma value
     */
    private final double sigmoidSigmaIntervalEnd;
    /**
     * the total number of epochs GMLVQ is supposed to train - this value is
     * necessary as for correct adaption of the sigmoid sigma value after each
     * learning iteration the relation between already performed epochs and the
     * total number is crucial
     */
    private final int totalNumberOfEpochs;

    public SigmoidFunction(double sigmoidSigmaIntervalStart, double sigmoidSigmaIntervalEnd, int totalNumberOfEpochs) {
        this.sigmoidSigmaIntervalStart = sigmoidSigmaIntervalStart;
        this.currentSigmoidSigma = sigmoidSigmaIntervalStart;
        this.sigmoidSigmaIntervalEnd = sigmoidSigmaIntervalEnd;
        this.totalNumberOfEpochs = totalNumberOfEpochs;
    }

    /**
     * increase the {@link SigmoidFunction#currentSigmoidSigma} depending on the
     * current epoch number
     *
     * @param epoch
     *            the number of the current epoch
     */
    public void increaseSigmoidSigma(int epoch) {
        double proposedSigmoidSigma = (this.sigmoidSigmaIntervalStart
                + (this.sigmoidSigmaIntervalEnd - this.sigmoidSigmaIntervalStart)) / Math.log(this.totalNumberOfEpochs)
                * Math.log(epoch);
        this.currentSigmoidSigma = capToInterval(proposedSigmoidSigma);
    }

    /**
     * computes the sigmoid function for the input value
     *
     * @param x
     * @return
     */
    public double evaluate(double x) {
        return 1 / (1 + Math.exp(-this.currentSigmoidSigma * x));
    }

    /**
     * computes the derivative of the sigmoid function for the input value
     *
     * @param x
     * @return
     */
    public double evaluatePrime(double x) {
        double ss = evaluate(x);
        return this.currentSigmoidSigma * ss * (1 - ss);
    }

    public double getCurrentSigmoidSigma() {
        return this.currentSigmoidSigma;
    }

    public double getSigmoidSigmaIntervalStart() {
        return this.sigmoidSigmaIntervalStart;
    }

    public double getSigmoidSigmaIntervalEnd() {
        return this.sigmoidSigmaIntervalEnd;
    }

    public int getTotalNumberOfEpochs() {
        return this.totalNumberOfEpochs;
    }

    /**
     * ensures the new sigmoid sigma value can escape the defined interval
     *
     * @param proposedSigmoidSigma
     *            the new sigmoid sigma value
     * @return the input parameter, when it is within the interval - otherwise:
     *         <ul>
     *         <li>if it is smaller than the lower bound,
     *         {@link SigmoidFunction#sigmoidSigmaIntervalStart} is returned
     *         </li>
     *         <li>if it exceeds the upper bound,
     *         {@link SigmoidFunction#sigmoidSigmaIntervalEnd}</li>
     *         </ul>
     */
    private double capToInterval(double proposedSigmoidSigma) {
        if (proposedSigmoidSigma > this.sigmoidSigmaIntervalEnd) {
            return this.sigmoidSigmaIntervalEnd;
        } else if (proposedSigmoidSigma < this.sigmoidSigmaIntervalStart) {
            return this.sigmoidSigmaIntervalStart;
        } else {
            return proposedSigmoidSigma;
        }
    }

}
