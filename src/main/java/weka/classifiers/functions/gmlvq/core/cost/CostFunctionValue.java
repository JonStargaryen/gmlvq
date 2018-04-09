package weka.classifiers.functions.gmlvq.core.cost;

/**
 * Gathers all {@link CostFunction} implementations. Each entry can tell what
 * additional parameters it depends on (if any).<br />
 * Provides also a convenience handle to the cost function to optimize.
 *
 * @author S
 *
 */
public enum CostFunctionValue {

    NONE(false, false, false),
    COST_FUNCTION_VALUE_TO_OPTIMIZE(false, false, false),
    WEIGHTED_ACCURACY(true, false, false),
    FMEASURE(true, true, false),
    PRECISION_RECALL(true, false, true),
    DEFAULT_COST(false, false, false),
    CLASSIFICATION_ACCURACY(false, false, false);

    private boolean requiresConfusionMatrix;
    private boolean requiresBeta;
    private boolean requiresWeightVector;

    private CostFunctionValue(boolean requiresConfusionMatrix, boolean requiresBeta, boolean requiresWeightVector) {
        this.requiresConfusionMatrix = requiresConfusionMatrix;
        this.requiresBeta = requiresBeta;
        this.requiresWeightVector = requiresWeightVector;
    }

    public boolean requiresConfusionMatrix() {
        return this.requiresConfusionMatrix;
    }

    public boolean requiresBeta() {
        return this.requiresBeta;
    }

    public boolean requiresWeightVector() {
        return this.requiresWeightVector;
    }
}
