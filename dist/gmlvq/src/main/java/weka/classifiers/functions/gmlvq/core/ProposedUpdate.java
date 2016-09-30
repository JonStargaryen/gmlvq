package weka.classifiers.functions.gmlvq.core;

import static weka.classifiers.functions.GMLVQ.isRelevanceLearning;
import static weka.classifiers.functions.gmlvq.utilities.LinearAlgebraicCalculations.add;
import static weka.classifiers.functions.gmlvq.utilities.LinearAlgebraicCalculations.dyadicProduct;
import static weka.classifiers.functions.gmlvq.utilities.LinearAlgebraicCalculations.multiply;
import static weka.classifiers.functions.gmlvq.utilities.LinearAlgebraicCalculations.scaledTranslate;
import static weka.classifiers.functions.gmlvq.utilities.LinearAlgebraicCalculations.substract;

import java.util.ArrayList;
import java.util.List;

import weka.classifiers.functions.gmlvq.core.cost.CostFunctionCalculator;
import weka.classifiers.functions.gmlvq.model.DataPoint;
import weka.classifiers.functions.gmlvq.model.EmbeddedSpaceVector;
import weka.classifiers.functions.gmlvq.model.OmegaMatrix;
import weka.classifiers.functions.gmlvq.model.Prototype;
import weka.classifiers.functions.gmlvq.model.Vector;
import weka.classifiers.functions.gmlvq.model.WinningInformation;
import weka.classifiers.functions.gmlvq.utilities.LinearAlgebraicCalculations;
import weka.core.matrix.Matrix;

/**
 * Each stochastic gradient descent composes a update which consists of updated
 * prototypes and an updated omega matrix (which defines how data points and
 * prototypes are mapped to the embedded space).<br />
 * Most essential, this class provides the
 * {@link ProposedUpdate#incorporate(DataPoint)} method which processes
 * individual data points selected by the {@link GradientDescent} and utilizes
 * their information to build the potential update. The {@link UpdateManager}
 * will subsequently decide whether the update failed and should be rejected or
 * if either the updated prototypes or the update omega matrix will be used for
 * the next epoch.
 *
 * @author S
 *
 */
public class ProposedUpdate {

    // adaptation information
    /**
     * copies the structure of the current prototypes, initializes all values as
     * 0 and will monitor the accumulating changes dictated by GMLVQ's update
     * definition
     */
    private List<Vector> prototypeDeltas;
    /**
     * copies the structure of the current omega matrix, initializes all values
     * as 0 and will monitor the accumulating changes dictated by GMLVQ's update
     * definition
     */
    private Matrix omegaDelta;

    /**
     * combination of prototype delta and the prototypes at the start of this
     * epoch
     */
    private List<Prototype> updatedPrototypes;
    /**
     * combination of omega delta and the matrix at the start of this epoch
     */
    private OmegaMatrix updatedOmegaMatrix;

    /**
     * the original prototypes
     */
    private List<Prototype> prototypes;
    /**
     * the original omega matrix
     */
    private OmegaMatrix omegaMatrix;
    /**
     * handle to the sigmoidFunction function
     */
    private SigmoidFunction sigmoidFunction;
    /**
     * convenience handle to <code>-2*omega'</code>
     */
    private Matrix scaledTransposedOmegaMatrix;
    /**
     * flag whether data points and prototypes are mapped to the embedded space
     * and whether the omega matrix ('mapping rule') update has to be considered
     */
    private boolean relevanceLearning;
    private boolean updateFinished;
    /**
     * current omega learning rate
     */
    private double alphaO;
    /**
     * current prototype learning rate
     */
    private double alphaW;
    /**
     * needed as the manner of how the update is composed, indirectly depends on
     * the employed cost function
     */
    private CostFunctionCalculator costFunctionCalculator;

    public ProposedUpdate(List<Prototype> prototypes, SigmoidFunction sigmoidFunction, OmegaMatrix omegaMatrix, double alphaW,
            double alphaO, CostFunctionCalculator costFunctionCalculator) {

        this.prototypes = prototypes;
        this.sigmoidFunction = sigmoidFunction;
        this.omegaMatrix = omegaMatrix;
        this.alphaW = alphaW;
        this.alphaO = alphaO;
        this.costFunctionCalculator = costFunctionCalculator;
        this.scaledTransposedOmegaMatrix = omegaMatrix.transpose().times(-2.0);
        this.relevanceLearning = isRelevanceLearning(omegaMatrix);

        initializePrototypeDelta();
        initializeOmegaDelta();
    }

    public ProposedUpdate(List<Prototype> prototypes, SigmoidFunction sigmoid, OmegaMatrix omegaMatrix, double alphaW,
            double alphaO, List<ProposedUpdate> proposedUpdates, CostFunctionCalculator costFunctionCalculator) {

        this(prototypes, sigmoid, omegaMatrix, alphaW, alphaO, costFunctionCalculator);
        sumUpProposedUpdates(proposedUpdates);
    }

    private void initializeOmegaDelta() {
        if (this.relevanceLearning) {
            this.omegaDelta = new Matrix(this.omegaMatrix.getRowDimension(), this.omegaMatrix.getColumnDimension());
        } else {
            this.omegaDelta = new Matrix(1, 1);
        }
    }

    private void initializePrototypeDelta() {
        this.prototypeDeltas = new ArrayList<Vector>();
        for (Prototype scaffoldPrototype : this.prototypes) {
            this.prototypeDeltas
                    .add(new Vector(new double[scaffoldPrototype.getDimension()], scaffoldPrototype.getClassLabel()));
        }
    }

    /**
     * needed when more than 1 thread is used to compose updates
     *
     * @param proposedUpdates
     */
    private void sumUpProposedUpdates(List<ProposedUpdate> proposedUpdates) {

        for (ProposedUpdate proposedUpdate : proposedUpdates) {

            this.omegaDelta = this.omegaDelta.plusEquals(proposedUpdate.omegaDelta);
        }

        for (int i = 0; i < this.prototypeDeltas.size(); i++) {

            for (ProposedUpdate proposedUpdate : proposedUpdates) {

                this.prototypeDeltas.get(i)
                        .setValues(add(this.prototypeDeltas.get(i), proposedUpdate.prototypeDeltas.get(i)));;
            }
        }
    }

    /**
     * analyzes the given data point and its {@link WinningInformation} (such as
     * the closest prototypes and their distances), this information is added to
     * the prototype and omega deltas
     *
     * @param dataPoint
     */
    public void incorporate(DataPoint dataPoint) {
        EmbeddedSpaceVector embeddedSpaceVector = dataPoint.getEmbeddedSpaceVector(this.omegaMatrix);
        WinningInformation winningInformation = embeddedSpaceVector.getWinningInformation(this.prototypes);
        // calculate potential updates for a single data point
        double dSum = winningInformation.getDistanceSameClass() + winningInformation.getDistanceOtherClass();

        double glvqMuHat = (winningInformation.getDistanceOtherClass() - winningInformation.getDistanceSameClass())
                / Math.max(dSum, LinearAlgebraicCalculations.NUMERIC_CUTOFF);

        double xsi = this.sigmoidFunction.evaluatePrime(glvqMuHat)
                / Math.max(dSum * dSum, LinearAlgebraicCalculations.NUMERIC_CUTOFF);

        double updateScalingFactor = this.costFunctionCalculator.update(dataPoint);
        double psiPlus = -updateScalingFactor * xsi * winningInformation.getDistanceOtherClass();
        double psiMinus = updateScalingFactor * xsi * winningInformation.getDistanceSameClass();

        Vector differenceSameClass = substract(embeddedSpaceVector, winningInformation.getWinnerSameClass());
        Vector differenceOtherClass = substract(embeddedSpaceVector, winningInformation.getWinnerOtherClass());

        // projection back to data space
        Vector deltaWPlus;
        Vector deltaWMinus;
        if (this.relevanceLearning) {
            deltaWPlus = multiply(multiply(differenceSameClass, this.scaledTransposedOmegaMatrix), psiPlus);
            deltaWMinus = multiply(multiply(differenceOtherClass, this.scaledTransposedOmegaMatrix), psiMinus);
        } else {
            deltaWPlus = multiply(differenceSameClass, -2.0 * psiPlus);
            deltaWMinus = multiply(differenceOtherClass, -2.0 * psiMinus);
        }

        // add the current update
        addPrototypeDelta(winningInformation.getIndexWinnerSameClass(), deltaWPlus);
        addPrototypeDelta(winningInformation.getIndexWinnerOtherClass(), deltaWMinus);

        // when relevance learning, then compute omega changes
        if (this.relevanceLearning) {
            // compute omega changes
            Matrix deltaDPlusDeltaOmega = this.omegaMatrix
                    .times(dyadicProduct(substract(dataPoint, winningInformation.getWinnerSameClass())));
            Matrix deltaDMinusDeltaOmega = this.omegaMatrix
                    .times(dyadicProduct(substract(dataPoint, winningInformation.getWinnerOtherClass())));

            // add to previously monitored changes
            addOmegaDelta(psiPlus, psiMinus, deltaDPlusDeltaOmega, deltaDMinusDeltaOmega);
        }

    }

    private void addOmegaDelta(double psiPlus, double psiMinus, Matrix deltaDPlusDeltaOmega,
            Matrix deltaDMinusDeltaOmega) {
        this.omegaDelta.plusEquals(deltaDPlusDeltaOmega.times(psiPlus).plus(deltaDMinusDeltaOmega.times(psiMinus)));
    }

    private void addPrototypeDelta(int index, Vector change) {
        this.prototypeDeltas.get(index).setValues(add(this.prototypeDeltas.get(index), change));
    }

    public OmegaMatrix getUpdatedOmegaMatrix() {
        if (!this.updateFinished) {
            finishUpdate();
        }
        return this.updatedOmegaMatrix;
    }

    /**
     * we cannot decide whether some incorporated data point is the last to
     * consider and the update is finished; rather, when other classes request
     * the update prototypes and omega matrix, this method will be invoked once
     * and, thus, ensure all delta are normalized.
     */
    private void finishUpdate() {

        // normalizes deltas
        normalizeDeltas();

        // calculate the updated omega matrix
        this.updatedOmegaMatrix = new OmegaMatrix(this.omegaMatrix.plus(this.omegaDelta.times(this.alphaO)));

        // calculate updated prototypes
        this.updatedPrototypes = new ArrayList<Prototype>();
        for (int prototypeIndex = 0; prototypeIndex < this.prototypes.size(); prototypeIndex++) {
            Prototype originalPrototype = this.prototypes.get(prototypeIndex);
            Prototype updatedPrototype = new Prototype(
                    scaledTranslate(originalPrototype, this.alphaW, this.prototypeDeltas.get(prototypeIndex)));
            this.updatedPrototypes.add(updatedPrototype);
        }

        this.updateFinished = true;
    }

    /**
     * normalizes the computed updates by the L2 norm
     */
    private void normalizeDeltas() {
        // prototypes
        // compute normalization factor
        double prototypeSum = 0;
        for (Vector prototypeDelta : this.prototypeDeltas) {
            for (double value : prototypeDelta.getValues()) {
                prototypeSum += value * value;
            }
        }
        double prototypeNormalizationFactor = 1 / Math.max(prototypeSum, LinearAlgebraicCalculations.NUMERIC_CUTOFF);

        // apply
        for (Vector prototypeDelta : this.prototypeDeltas) {
            prototypeDelta.setValues(multiply(prototypeDelta, prototypeNormalizationFactor));
        }

        // matrix
        // compute normalization factor
        if (this.relevanceLearning) {
            double omegaMatrixSum = 0;
            for (double d : this.omegaDelta.getRowPackedCopy()) {
                omegaMatrixSum += d * d;
            }
            double omegaMatrixNormalizationFactor = 1
                    / Math.max(omegaMatrixSum, LinearAlgebraicCalculations.NUMERIC_CUTOFF);
            // apply
            this.omegaDelta.timesEquals(omegaMatrixNormalizationFactor);
        }
    }

    public List<Prototype> getUpdatedPrototypes() {
        if (!this.updateFinished) {

            finishUpdate();
        }
        return this.updatedPrototypes;
    }
}
