package weka.classifiers.functions.gmlvq.core.cost;

import weka.classifiers.functions.gmlvq.core.SigmoidFunction;
import weka.classifiers.functions.gmlvq.model.DataPoint;
import weka.classifiers.functions.gmlvq.model.OmegaMatrix;
import weka.classifiers.functions.gmlvq.model.Prototype;

import java.io.Serializable;
import java.util.*;
import java.util.concurrent.ExecutionException;

/**
 * A wrapping class for all {@link CostFunction}s to be calculated during
 * training. One of them guides the learning process and dictates which updates
 * to choose and which to reject - this is the
 * {@link CostFunctionValue#COST_FUNCTION_VALUE_TO_OPTIMIZE}. Additional cost
 * functions can be computed for the sole purpose of visualization.
 *
 * @author S
 *
 */
public class CostFunctionCalculator implements Serializable {

    private static final long serialVersionUID = 1L;

    private SigmoidFunction sigmoidFunction;
    private CostFunctionValue costFunctionValueToOptimize;
    private EnumSet<CostFunctionValue> additionalCostFunctionValuesToCalculate;
    private ConfusionMatrix confusionMatrix;
    private Map<CostFunctionValue, CostFunction> persistentCostFunctions;
    private double costFunctionBeta;
    private double[] costFunctionWeights;
    public static final double DEFAULT_BETA = 2.0;
    public static final double[] DEFAULT_WEIGHTS = new double[] { 0.5, 0.5 };

    public CostFunctionCalculator(SigmoidFunction sigmoidFunction, double costFunctionBeta,
            double[] costFunctionWeights, CostFunctionValue costFunctionValueToOptimize,
            CostFunctionValue... additionalCostFunctionValuesToCalculate) {
        this.sigmoidFunction = sigmoidFunction;
        this.costFunctionBeta = costFunctionBeta;
        this.costFunctionWeights = costFunctionWeights;
        this.costFunctionValueToOptimize = costFunctionValueToOptimize;
        this.additionalCostFunctionValuesToCalculate = EnumSet.of(costFunctionValueToOptimize,
                additionalCostFunctionValuesToCalculate);
        initializePersistentCostFunctions();
    }

    public CostFunctionCalculator(SigmoidFunction sigmoidFunction, CostFunctionValue costFunctionValueToOptimize,
            CostFunctionValue... additionalCostFunctionValuesToCalculate) {
        this(sigmoidFunction, DEFAULT_BETA, DEFAULT_WEIGHTS, costFunctionValueToOptimize,
                additionalCostFunctionValuesToCalculate);
    }

    /**
     * initializes all persistent cost functions that will be used over the
     * course of learning
     */
    private void initializePersistentCostFunctions() {

        this.persistentCostFunctions = new HashMap<CostFunctionValue, CostFunction>();
        // add cost function value to optimize
        addPersistenceCostFunction(this.costFunctionValueToOptimize);

        // iterate over all additional cost functions
        for (CostFunctionValue costFunctionValue : this.additionalCostFunctionValuesToCalculate) {
            addPersistenceCostFunction(costFunctionValue);
        }
    }

    /**
     * adds the specified cost function to persistence cost functions
     *
     * @param costFunctionValue
     */
    private void addPersistenceCostFunction(CostFunctionValue costFunctionValue) {

        // do not double add
        if (this.persistentCostFunctions.containsKey(costFunctionValue)) {
            return;
        }
        switch (costFunctionValue) {
        case CLASSIFICATION_ERROR:
            this.persistentCostFunctions.put(costFunctionValue, new ClassificationErrorFunction(this.sigmoidFunction));
            break;
        case DEFAULT_COST:
            this.persistentCostFunctions.put(costFunctionValue, new DefaultCostFunction(this.sigmoidFunction));
            break;
        default:
            break;
        }
    }

    public double update(DataPoint dataPoint) {
        double result = updateInternal(dataPoint);
        // System.out.println("update: " + result);
        return result;
    }

    private double updateInternal(DataPoint dataPoint) {
        switch (this.costFunctionValueToOptimize) {
        case WEIGHTED_ACCURACY:
            return this.confusionMatrix.computeWeightedAccuracyUpdate(dataPoint, this.costFunctionWeights[0],
                    this.costFunctionWeights[1]);
        case FMEASURE:
            return this.confusionMatrix.computeFMeasureUpdate(dataPoint, this.costFunctionBeta);
        case PRECISION_RECALL:
            return this.confusionMatrix.computePrecisionRecallUpdate(dataPoint, this.costFunctionWeights[0],
                    this.costFunctionWeights[1]);
        default:
            // this happens when non-confusion-based cost functions are employed
            return 1.0;
        }
    }

    public ConfusionMatrix getConfusionMatrix() {
        return this.confusionMatrix;
    }

    public Map<CostFunctionValue, Double> evaluate(List<DataPoint> chosenDataPoints, List<Prototype> prototypes,
            OmegaMatrix omegaMatrix) throws InterruptedException, ExecutionException {

        // ensure confusion matrix is reset so no old results are considered
        this.confusionMatrix = null;
        // maybe other values need to be reseted as well here

        Map<CostFunctionValue, Double> costs = new HashMap<CostFunctionValue, Double>();

        for (CostFunctionValue costFunctionValue : this.additionalCostFunctionValuesToCalculate) {
            double value = computeCostFunctionValue(costFunctionValue, chosenDataPoints, prototypes, omegaMatrix);
            costs.put(costFunctionValue, value);
            if (costFunctionValue == this.costFunctionValueToOptimize) {
                costs.put(CostFunctionValue.COST_FUNCTION_VALUE_TO_OPTIMIZE, value);
            }
        }

        return costs;
    }

    private double computeCostFunctionValue(CostFunctionValue costFunctionValue, List<DataPoint> chosenDataPoints,
            List<Prototype> prototypes, OmegaMatrix omegaMatrix) throws InterruptedException, ExecutionException {
        // does this feature need the confM and isn't it present yet? then
        // compute, duh
        if (costFunctionValue.requiresConfusionMatrix() && this.confusionMatrix == null) {
            this.confusionMatrix = new ConfusionMatrix(this.sigmoidFunction, chosenDataPoints, prototypes, omegaMatrix);
            // System.out.println(this.confusionMatrix);
        }

        // check for requirement which cannot be computed on the fly and fail
        // accordingly
        if (costFunctionValue.requiresBeta() && this.costFunctionBeta == 0
                || costFunctionValue.requiresWeightVector() && this.costFunctionWeights == null) {
            // throw more detailed exception
            throw new IllegalStateException(
                    "cannot evaluate costs for " + costFunctionValue.name() + " as requirements are missing");
        }

        switch (costFunctionValue) {
        case WEIGHTED_ACCURACY:
            return this.confusionMatrix.computeWeightedAccuracy(this.costFunctionWeights[0],
                    this.costFunctionWeights[1]);
        case FMEASURE:
            return this.confusionMatrix.computeFMeasure(this.costFunctionBeta);
        case PRECISION_RECALL:
            return this.confusionMatrix.computePrecisionRecall(this.costFunctionWeights[0],
                    this.costFunctionWeights[1]);
        case DEFAULT_COST:
            // TODO: is this really the best place to 'invert'???
            return 1 - this.persistentCostFunctions.get(CostFunctionValue.DEFAULT_COST).evaluate(chosenDataPoints,
                    prototypes, omegaMatrix);
        case CLASSIFICATION_ERROR:
            return 1 - this.persistentCostFunctions.get(CostFunctionValue.CLASSIFICATION_ERROR)
                    .evaluate(chosenDataPoints, prototypes, omegaMatrix);
        default:
            throw new UnsupportedOperationException("no calculation method known for " + costFunctionValue.name());
        }
    }

    public String defaultCostFunctionString() {
        StringBuilder builder = new StringBuilder();
        builder.append(costFunctionValueToOptimize.name());
        if (costFunctionValueToOptimize.requiresBeta()) {
            builder.append(" with beta ").append(costFunctionBeta);
        }
        if (costFunctionValueToOptimize.requiresWeightVector()) {
            builder.append(" with weights ").append(Arrays.toString(costFunctionWeights));
        }
        return builder.toString();
    }

}
