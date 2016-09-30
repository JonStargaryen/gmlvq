package weka.classifiers.functions.gmlvq.core.cost;

import weka.classifiers.functions.gmlvq.core.SigmoidFunction;
import weka.classifiers.functions.gmlvq.model.WinningInformation;
import weka.classifiers.functions.gmlvq.utilities.LinearAlgebraicCalculations;

/**
 * The default cost function which can be employed in any case. For problems
 * with more than 2 classes, currently the only {@link CostFunction}
 * implementation applicable.
 * 
 * @author S
 *
 */
public class DefaultCostFunction extends AbstractCostFunction {

    private static final long serialVersionUID = 1L;

    public DefaultCostFunction(SigmoidFunction sigmoidFunction) {
        super(sigmoidFunction);
    }

    @Override
    protected double evaluateWinningInformation(WinningInformation winningInformation) {
        double dplus = winningInformation.getDistanceSameClass();
        double dminus = winningInformation.getDistanceOtherClass();
        double scalingFactor = Math.max(dplus + dminus, LinearAlgebraicCalculations.NUMERIC_CUTOFF);
        return this.sigmoidFunction.evaluate((dplus - dminus) / scalingFactor);
    }
}
