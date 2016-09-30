package weka.classifiers.functions.gmlvq.core.cost;

import weka.classifiers.functions.gmlvq.core.SigmoidFunction;
import weka.classifiers.functions.gmlvq.model.WinningInformation;

/**
 * Can be used to compute the classification error.<br />
 * <b>IMPORTANT: For 2 class problems, use the {@link ConfusionMatrix}
 * implementation. This class is specificably designed for more than 2 classes
 * and cannot really be used to guide the training process, but rather be only
 * employed for additional information.</b> TODO: safety net: die upon 2 class
 * problems
 *
 * @author S
 *
 */
public class ClassificationErrorFunction extends AbstractCostFunction {

    public ClassificationErrorFunction(SigmoidFunction sigmoidFunction) {
        super(sigmoidFunction);
    }

    private static final long serialVersionUID = 1L;

    /**
     * return 1 if classes do not match and 0 if the are in agreement
     */
    @Override
    protected double evaluateWinningInformation(WinningInformation winningInformation) {
        return winningInformation.getDistanceSameClass() > winningInformation.getDistanceOtherClass() ? 1 : 0;
    }
}
