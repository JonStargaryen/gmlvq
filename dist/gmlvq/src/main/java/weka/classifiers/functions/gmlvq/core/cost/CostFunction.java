package weka.classifiers.functions.gmlvq.core.cost;

import java.io.Serializable;
import java.util.List;
import java.util.concurrent.ExecutionException;

import weka.classifiers.functions.gmlvq.core.Disposable;
import weka.classifiers.functions.gmlvq.core.ProposedUpdate;
import weka.classifiers.functions.gmlvq.model.DataPoint;
import weka.classifiers.functions.gmlvq.model.OmegaMatrix;
import weka.classifiers.functions.gmlvq.model.Prototype;

/**
 * Defines the contract of each cost function.
 *
 * @author S
 *
 */
public interface CostFunction extends Serializable, Disposable {

    /**
     * computes the costs for the given configuration of data and prototypes
     * 
     * @param dataPoints
     *            the data points to be evaluated
     * @param prototypes
     *            the prototypes to be evaluated (either original or in a
     *            {@link ProposedUpdate})
     * @param omegaMatrix
     *            how to map data space vector (either original or in a
     *            {@link ProposedUpdate})?
     * @return a double value describing the costs of the current combination of
     *         data points, prototypes and mapping rule.
     * @throws InterruptedException
     * @throws ExecutionException
     */
    double evaluate(List<DataPoint> dataPoints, List<Prototype> prototypes, OmegaMatrix omegaMatrix)
            throws InterruptedException, ExecutionException;
}
