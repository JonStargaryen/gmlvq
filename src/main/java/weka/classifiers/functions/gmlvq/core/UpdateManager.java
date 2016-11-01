package weka.classifiers.functions.gmlvq.core;

import java.io.Serializable;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;

import weka.classifiers.functions.GMLVQ;
import weka.classifiers.functions.gmlvq.core.cost.CostFunctionCalculator;
import weka.classifiers.functions.gmlvq.core.cost.CostFunctionValue;
import weka.classifiers.functions.gmlvq.model.DataPoint;
import weka.classifiers.functions.gmlvq.model.Observer;
import weka.classifiers.functions.gmlvq.model.OmegaMatrix;
import weka.classifiers.functions.gmlvq.model.Prototype;
import weka.classifiers.functions.gmlvq.utilities.DataRandomizer;
import weka.classifiers.functions.gmlvq.utilities.LinearAlgebraicCalculations;
import weka.classifiers.functions.gmlvq.visualization.Visualizer;
import weka.core.matrix.Matrix;

/**
 * The instance directing the learning process. For each epoch
 * {@link GradientDescent#performStochasticGradientDescent(List, List, OmegaMatrix, double, double)}
 * is invoked to compose a {@link ProposedUpdate}. The UpdateManager will then
 * decide - based on the result of the {@link CostFunctionCalculator} - which
 * updates to accept and which to reject.<br />
 * Also, this class handles output to the console as well as the
 * {@link Visualizer}. Last but not least, the
 * {@link SigmoidFunction#increaseSigmoidSigma(int)} is called in order to
 * update the sigmoid sigma value.
 *
 * @author S
 *
 */
public class UpdateManager implements Serializable {

    private static final long serialVersionUID = 1L;

    private List<DataPoint> dataPoints;
    private List<Prototype> prototypes;
    private OmegaMatrix omegaMatrix;
    private SigmoidFunction sigmoidFunction;

    private DataRandomizer dataRandomizer;
    private double currentCostValueToOptimize;
    private double prototypeLearningRate;
    private double omegaLearningRate;
    private double learnRateChange;
    private double stopCriterion;
    private boolean relevanceLearning;
    private int currentEpoch;
    private int numberOfTotalEpochs;
    private int numberOfPerformedPrototypeUpdates;
    private int numberOfPerformedOmegaUpdates;

    private final double initialCostValueToOptimize;

    private Observer observer;

    private Matrix lambdaMatrix;

    private double lambdaMatrixScalingFactor;

    private CostFunctionCalculator costFunctionCalculator;

    private Map<CostFunctionValue, Double> currentCostValues;

    public UpdateManager(GMLVQCore gmlvqCore, CostFunctionCalculator costFunctionCalculator, Observer observer)
            throws InterruptedException, ExecutionException {
        this.dataPoints = gmlvqCore.getDataPoints();
        this.prototypes = gmlvqCore.getPrototypes();
        this.omegaMatrix = gmlvqCore.getOmegaMatrix();
        this.sigmoidFunction = gmlvqCore.getSigmoidFunction();
        this.dataRandomizer = gmlvqCore.getDataRandomizer();
        this.numberOfTotalEpochs = gmlvqCore.getNumberOfTotalEpochs();
        this.prototypeLearningRate = gmlvqCore.getPrototypeLearningRate();
        this.omegaLearningRate = gmlvqCore.getOmegaLearningRate();
        this.learnRateChange = gmlvqCore.getLearnRateChange();
        this.stopCriterion = gmlvqCore.getStopCriterion();
        this.relevanceLearning = GMLVQ.isRelevanceLearning(this.omegaMatrix);
        this.costFunctionCalculator = costFunctionCalculator;
        this.observer = observer;

        this.currentCostValues = this.costFunctionCalculator.evaluate(this.dataPoints, this.prototypes,
                this.omegaMatrix);
        this.currentCostValueToOptimize = this.currentCostValues.get(CostFunctionValue.COST_FUNCTION_VALUE_TO_OPTIMIZE);
        this.initialCostValueToOptimize = this.currentCostValueToOptimize;

        GMLVQCore.LOGGER.info("initial costs: " + this.currentCostValueToOptimize);

        outputCurrentCostFunctionValues();
    }

    private void outputCurrentCostFunctionValues() {
        for (CostFunctionValue costFunctionValue : this.currentCostValues.keySet()) {
            GMLVQCore.LOGGER.info("costs for function " + costFunctionValue + " are "
                    + this.currentCostValues.get(costFunctionValue));
        }
    }

    public double getPrototypeLearningRate() {
        return this.prototypeLearningRate;
    }

    public double getOmegaLearningRate() {
        return this.omegaLearningRate;
    }

    public boolean update(ProposedUpdate proposedUpdate) throws InterruptedException, ExecutionException {
        // decide whether to update prototypes or matrix
        // and notify with the correct update

        if (this.currentEpoch % 200 == 0) {
            GMLVQCore.LOGGER.info("epoch " + this.currentEpoch + " / " + this.numberOfTotalEpochs);
            outputCurrentCostFunctionValues();
        }

        List<DataPoint> chosenDataPoints = this.dataRandomizer.generateRandomizedSubListOf(this.dataPoints);
        OmegaMatrix updatedOmega = proposedUpdate.getUpdatedOmegaMatrix();
        List<Prototype> updatedPrototypes = proposedUpdate.getUpdatedPrototypes();

        // calculate costs for changing the prototypes
        Map<CostFunctionValue, Double> prototypeUpdateCostsValues = this.costFunctionCalculator
                .evaluate(chosenDataPoints, updatedPrototypes, this.omegaMatrix);
        double prototypeUpdateCost = prototypeUpdateCostsValues.get(CostFunctionValue.COST_FUNCTION_VALUE_TO_OPTIMIZE);

        // calculate costs for changing the omega matrix, iff matrix
        // learning is enabled
        Map<CostFunctionValue, Double> omegaUpdateCostValues = this.costFunctionCalculator.evaluate(chosenDataPoints,
                this.prototypes, updatedOmega);
        double omegaUpdateCost = this.relevanceLearning
                ? omegaUpdateCostValues.get(CostFunctionValue.COST_FUNCTION_VALUE_TO_OPTIMIZE)
                : prototypeUpdateCost - LinearAlgebraicCalculations.NUMERIC_CUTOFF;

        // apply update, iff the current cost value is smaller than one of
        // the cost of one proposed update
        if (this.currentCostValueToOptimize >= Math.max(prototypeUpdateCost, omegaUpdateCost)) {
            // update rejected: decrease learning rates
            this.prototypeLearningRate -= this.learnRateChange * this.prototypeLearningRate;
            this.omegaLearningRate -= this.learnRateChange * this.omegaLearningRate;
            GMLVQCore.LOGGER.fine("learned nothing, decreasing learning rates to prototypeLearningRate="
                    + this.prototypeLearningRate + "\talphaO=" + this.omegaLearningRate);

        } else {
            // prototype learning occurs if the corresponding cost is
            // preferred or when no matrix learning is happening
            if (prototypeUpdateCost >= omegaUpdateCost || !this.relevanceLearning) {
                this.prototypes.clear();
                this.prototypes.addAll(updatedPrototypes);
                this.prototypeLearningRate += this.learnRateChange * this.prototypeLearningRate;
                this.currentCostValueToOptimize = prototypeUpdateCost;
                this.numberOfPerformedPrototypeUpdates++;

                if (this.observer != null) {
                    this.observer.updatePrototypes(this.prototypes);
                }

                // set new costs
                this.currentCostValues = prototypeUpdateCostsValues;

                GMLVQCore.LOGGER.fine("learned prototypes " + this.prototypes);
            } else {
                this.omegaMatrix.setMatrix(0, this.omegaMatrix.getRowDimension() - 1, 0,
                        this.omegaMatrix.getColumnDimension() - 1, updatedOmega);
                this.omegaLearningRate += this.learnRateChange * this.omegaLearningRate;

                // when visualization is happening and there is something to
                // visualize, do so
                if (this.observer != null && GMLVQ.isRelevanceLearning(this.omegaMatrix)) {
                    computeLambdaMatrix();
                    normalizeOmegaMatrix();

                    // submit update to observer
                    this.observer.updateLambdaMatrix(this.lambdaMatrix);
                }

                this.currentCostValueToOptimize = omegaUpdateCost;
                this.numberOfPerformedOmegaUpdates++;

                // set new costs
                this.currentCostValues = omegaUpdateCostValues;

                GMLVQCore.LOGGER.fine("learned omega matrix\n" + this.omegaMatrix);
            }
        }

        if (this.observer != null) {
            this.observer.updateCostFunctions(this.currentCostValues);
        }

        // deregister all mappings which are not relevant any|more
        for (DataPoint dataPoint : this.dataPoints) {
            dataPoint.deregisterAllMappingsBut(this.omegaMatrix, this.prototypes);
        }
        for (Prototype prototype : this.prototypes) {
            prototype.deregisterAllMappingBut(this.omegaMatrix);
        }

        // increase sigma by the given percentage
        this.sigmoidFunction.increaseSigmoidSigma(this.currentEpoch);
        this.currentEpoch++;

        // if learning rates were decreasing for a long time or the number of
        // the current epoch exceeds the specified number of epochs to train
        return stopCriterionNotMet();
    }

    private void computeLambdaMatrix() {
        this.lambdaMatrix = this.omegaMatrix.transpose().times(this.omegaMatrix);
        this.lambdaMatrixScalingFactor = Math.sqrt(this.lambdaMatrix.trace());
    }

    /**
     * normalizes the omega matrix<br />
     * before this call the relevance lambda matrix ought to be
     * calculated/updated
     */
    private void normalizeOmegaMatrix() {
        this.omegaMatrix.timesEquals(1 / this.lambdaMatrixScalingFactor);
    }

    /**
     * determines whether the stop criterion of the algorithm is met and, thus,
     * learning should cease prematurely
     *
     * @return true if the stop criterion is met
     */
    private boolean stopCriterionNotMet() {
        if (this.prototypeLearningRate < this.stopCriterion && this.omegaLearningRate < this.stopCriterion
                || this.currentEpoch >= this.numberOfTotalEpochs) {
            GMLVQCore.LOGGER.info("stop criterion met, exiting after " + this.currentEpoch + " / "
                    + this.numberOfTotalEpochs + " epochs with cost " + this.currentCostValueToOptimize);

            if (GMLVQ.isRelevanceLearning(this.omegaMatrix)) {
                computeLambdaMatrix();
                normalizeOmegaMatrix();
                computeLambdaMatrix();
            }

            summarizeLearningProcess();
            return false;
        }
        return true;
    }

    private void summarizeLearningProcess() {

        int numberOfPerformedUpdates = this.numberOfPerformedOmegaUpdates + this.numberOfPerformedPrototypeUpdates;
        int numberOfRejectedUpdates = this.numberOfTotalEpochs - numberOfPerformedUpdates;

        StringBuilder sb = new StringBuilder();
        sb.append("\nreturned prototypes:\n");
        for (Prototype prototype : this.prototypes) {
            sb.append(prototype + "\n");
        }

        sb.append("\nover the course of learning the cost function changed from " + this.initialCostValueToOptimize
                + " to " + this.currentCostValueToOptimize + "\n");

        outputCurrentCostFunctionValues();

        sb.append("\nlearning iterations:\n" + "total: " + numberOfPerformedUpdates + " / " + this.numberOfTotalEpochs
                + " (" + 100 * ((double) numberOfPerformedUpdates / this.numberOfTotalEpochs) + "%)\nrejected: "
                + numberOfRejectedUpdates + " / " + this.numberOfTotalEpochs + " ("
                + 100 * ((double) numberOfRejectedUpdates / this.numberOfTotalEpochs) + "%)\n" + "prototypes: "
                + this.numberOfPerformedPrototypeUpdates + " / " + this.numberOfTotalEpochs + " ("
                + 100 * ((double) this.numberOfPerformedPrototypeUpdates / this.numberOfTotalEpochs) + "%)\n"
                + "omega: " + this.numberOfPerformedOmegaUpdates + " / " + this.numberOfTotalEpochs + " ("
                + 100 * ((double) this.numberOfPerformedOmegaUpdates / this.numberOfTotalEpochs) + "%)\n");

        if (this.relevanceLearning) {
            // just for output we have to encapsulate the lambda matrix
            sb.append("\nomega matrix:\n" + this.omegaMatrix.toString() + "\nlambda matrix:\n"
                    + new OmegaMatrix(this.lambdaMatrix).toString());
        }

        GMLVQCore.LOGGER.info(sb.toString());
    }

}
