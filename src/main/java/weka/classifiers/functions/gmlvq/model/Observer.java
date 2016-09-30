
package weka.classifiers.functions.gmlvq.model;

import java.util.List;
import java.util.Map;

import weka.classifiers.functions.gmlvq.core.UpdateManager;
import weka.classifiers.functions.gmlvq.core.cost.CostFunctionValue;
import weka.core.matrix.Matrix;

/**
 * Enables implementing classes to be informed about data to visualized
 * propagated by the {@link UpdateManager}.
 *
 * @author S
 *
 */
public interface Observer {

    /**
     * hand over the current cost values to visualize
     * 
     * @param currentCostValues
     */
    void updateCostFunctions(Map<CostFunctionValue, Double> currentCostValues);

    /**
     * hand over the new lambda matrix to visualize
     * 
     * @param lambdaMatrix
     */
    void updateLambdaMatrix(Matrix lambdaMatrix);

    /**
     * hand over the new prototypes to visualize
     * 
     * @param prototypes
     */
    void updatePrototypes(List<Prototype> prototypes);
}
