package weka.classifiers.functions.gmlvq.core;

import weka.classifiers.functions.gmlvq.core.cost.*;
import weka.classifiers.functions.gmlvq.model.*;
import weka.classifiers.functions.gmlvq.model.Observer;
import weka.classifiers.functions.gmlvq.utilities.DataRandomizer;
import weka.classifiers.functions.gmlvq.utilities.LinearAlgebraicCalculations;
import weka.core.Instances;
import weka.core.matrix.EigenvalueDecomposition;
import weka.core.matrix.Matrix;

import java.io.Serializable;
import java.util.*;
import java.util.concurrent.ExecutionException;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * The implementation of the generalized matrix learning vector quantization.
 * Basically, this is a prototype-based, supervised learning procedure.<br />
 * <br />
 * The conventional LVQ was enriched a linear mapping rule provided by a matrix
 * (G<b>M</b>LVQ). This matrix has a dimension of
 * <code>dataDimension x omegaDimension</code>. The omega dimension can be set
 * to <code>2...dataDimension</code>. Depending on the set omega dimension each
 * data point and prototype will be mapped (resp. linearly transformed) to an
 * embedded data space. Within this data space distance between data points and
 * prototypes are computed and this information is used to compose the update
 * for each learning epoch. Setting the omega dimension to values significantly
 * smaller than the data dimension will drastically speed up the learning
 * process. As mapping to the embedded space of data points is still
 * computationally expensive, we 'cache' these mappings. By invoking
 * {@link DataPoint#getEmbeddedSpaceVector(OmegaMatrix)} one can retrieve the
 * {@link EmbeddedSpaceVector} for this data point according to the specified
 * mapping rule (provided by the {@link OmegaMatrix}). As the computation of the
 * embedding can be quite expensive, results are directly link to the data
 * points. So they are only calculated once and can then by recalled.
 * Subsequently, by calling
 * {@link EmbeddedSpaceVector#getWinningInformation(List)} one can access the
 * {@link WinningInformation} linked to each embedded space vector. These
 * information include the distance to the closest prototype of the same class
 * as the 'asked' data point as well as the distance to the closest prototype of
 * a different class. This information is crucial in composing the update of
 * each epoch as well as for the computation of {@link CostFunction}s.<br />
 * <br />
 * Also <b>G</b>MLVQ is capable of generalization, meaning various
 * {@link CostFunction}s can be optimized. Most notably, it is possible to
 * evaluate the success of each epoch by consulting the F-measure or
 * precision-recall values.<br />
 * <br />
 * Another key feature is the possibility of tracking the influence of
 * individual features within the input data which contribute the most to the
 * training process. This is realized by a lambda matrix (defined as
 * <code>lambda = omega * omega'</code>). This matrix can be visualized and will
 * contain the influence of features to the classification on its principal
 * axis. Other elements describe the correlation between the corresponding
 * features.<br />
 * <br />
 * This class takes care of the correct initialization of all parameters and
 * such of one GMLVQ run. To track input parameters, a internal builder is
 * employed. Most of the internal tasks are then delegated to the
 * {@link UpdateManager} which will direct the learning process.
 *
 * @author S
 */
public class GMLVQCore implements Serializable {

    public static final Logger LOGGER = Logger.getLogger(GMLVQCore.class.getName());
    private static final long serialVersionUID = 1L;

    static {
        ConsoleHandler consoleHandler = new ConsoleHandler();
        consoleHandler.setLevel(Level./* FINEST */INFO);
        LOGGER.setUseParentHandlers(false);
        LOGGER.setLevel(Level./* FINEST */INFO);
        LOGGER.addHandler(consoleHandler);
    }

    // required
    private List<DataPoint> dataPoints;
    // optional fields
    private int numberOfTotalEpochs;
    private int numberOfPrototypesPerClass;
    private int omegaDimension;
    private double learnRateChange;
    private double prototypeLearningRate;
    private double omegaLearningRate;
    private double dataPointRatioPerRound;
    private double sigmoidSigmaIntervalStart;
    private double sigmoidSigmaIntervalEnd;
    private double stopCriterion;
    private boolean matrixLearning;
    private boolean parallelExecution;
    private boolean visualization;
    private long seed;
    private int numberOfClasses;
    private int dataDimension;
    // this is initialized by GMLVQCore
    private DataRandomizer dataRandomizer;
    private OmegaMatrix omegaMatrix;
    private OmegaMatrix lambdaMatrix;
    private double lambdaMatrixScalingFactor;
    private Map<Double, Integer> prototypesPerClass;
    private List<Prototype> prototypes;
    private SigmoidFunction sigmoidFunction;
    private DefaultCostFunction costFunction;
    private ClassificationErrorFunction classificationErrorFunction;
    private UpdateManager updateManager;
    private GradientDescent gradientDescent;

    private GMLVQCore(Builder builder) throws InterruptedException, ExecutionException {
        this.dataPoints = builder.dataPoints;
        this.numberOfTotalEpochs = builder.numberOfEpochs;
        this.numberOfPrototypesPerClass = builder.numberOfPrototypesPerClass;
        this.prototypesPerClass = builder.prototypesPerClass;
        this.omegaDimension = builder.omegaDimension;

        this.learnRateChange = builder.learnRateChange;
        this.prototypeLearningRate = builder.prototypeLearningRate;
        this.omegaLearningRate = builder.omegaLearningRate;
        this.dataPointRatioPerRound = builder.dataPointRationPerRound;
        this.sigmoidSigmaIntervalStart = builder.sigmoidSigmaIntervalStart;
        this.sigmoidSigmaIntervalEnd = builder.sigmoidSigmaIntervalEnd;
        this.stopCriterion = builder.stopCriterion;

        this.matrixLearning = builder.matrixLearning;
        this.parallelExecution = builder.parallelExecution;
        this.visualization = builder.visualization;

        this.seed = builder.seed;

        this.numberOfClasses = builder.numberOfClasses;
        this.dataDimension = builder.dataDimension;

        this.dataRandomizer = new DataRandomizer(this.dataPoints.size(), this.dataPointRatioPerRound, this.seed);
        this.sigmoidFunction = new SigmoidFunction(this.sigmoidSigmaIntervalStart, this.sigmoidSigmaIntervalEnd,
                this.numberOfTotalEpochs);

        // initialize cost function calculator
        CostFunctionCalculator costFunctionCalculator = new CostFunctionCalculator(this.sigmoidFunction,
                builder.costFunctionBeta,
                builder.costFunctionWeights,
                builder.costFunctionToOptimize,
                builder.additionalCostFunctions
                        .toArray(new CostFunctionValue[0]));
        // register multiple costs functions
        // List<CostFunction> additionalCostFunctions = new
        // ArrayList<CostFunction>();
        // additionalCostFunctions.add(new
        // ClassificationErrorFunction(this.sigmoidFunction));

        this.gradientDescent = new GradientDescent(this.dataRandomizer, this.sigmoidFunction, costFunctionCalculator);
        initializeMatrices();
        initializePrototypes();
        if (this.visualization) {
            builder.observer.updatePrototypes(this.prototypes);
            builder.observer.updateLambdaMatrix(this.getLambdaMatrix());
        }

        // create the update manager with additional cost functions
        this.updateManager = new UpdateManager(this, costFunctionCalculator, builder.observer);
        // this.updateManager = new UpdateManager(this.dataPoints,
        // this.prototypes, this.omegaMatrix, this.sigmoidFunction,
        // this.costFunction, this.dataRandomizer, this.numberOfTotalEpochs,
        // builder.prototypeLearningRate,
        // builder.omegaLearningRate, this.learnRateChange, this.stopCriterion,
        // additionalCostFunctions);
    }

    public double getPrototypeLearningRate() {
        return this.prototypeLearningRate;
    }

    public double getOmegaLearningRate() {
        return this.omegaLearningRate;
    }

    public List<DataPoint> getDataPoints() {
        return this.dataPoints;
    }

    public int getNumberOfTotalEpochs() {
        return this.numberOfTotalEpochs;
    }

    public int getNumberOfPrototypesPerClass() {
        return this.numberOfPrototypesPerClass;
    }

    public int getOmegaDimension() {
        return this.omegaDimension;
    }

    public double getLearnRateChange() {
        return this.learnRateChange;
    }

    public double getDataPointRatioPerRound() {
        return this.dataPointRatioPerRound;
    }

    public double getSigmoidSigmaIntervalStart() {
        return this.sigmoidSigmaIntervalStart;
    }

    public double getSigmoidSigmaIntervalEnd() {
        return this.sigmoidSigmaIntervalEnd;
    }

    public double getStopCriterion() {
        return this.stopCriterion;
    }

    public boolean isMatrixLearning() {
        return this.matrixLearning;
    }

    public boolean isParallelExecution() {
        return this.parallelExecution;
    }

    public boolean isVisualization() {
        return this.visualization;
    }

    public long getSeed() {
        return this.seed;
    }

    public int getNumberOfClasses() {
        return this.numberOfClasses;
    }

    public int getDataDimension() {
        return this.dataDimension;
    }

    public DataRandomizer getDataRandomizer() {
        return this.dataRandomizer;
    }

    public OmegaMatrix getOmegaMatrix() {
        return this.omegaMatrix;
    }

    public OmegaMatrix getLambdaMatrix() {
        return this.lambdaMatrix;
    }

    public double getLambdaMatrixScalingFactor() {
        return this.lambdaMatrixScalingFactor;
    }

    public Map<Double, Integer> getPrototypesPerClass() {
        return this.prototypesPerClass;
    }

    public List<Prototype> getPrototypes() {
        return this.prototypes;
    }

    public SigmoidFunction getSigmoidFunction() {
        return this.sigmoidFunction;
    }

    public DefaultCostFunction getCostFunction() {
        return this.costFunction;
    }

    public ClassificationErrorFunction getClassificationErrorFunction() {
        return this.classificationErrorFunction;
    }

    public UpdateManager getUpdateManager() {
        return this.updateManager;
    }

    public GradientDescent getGradientDescent() {
        return this.gradientDescent;
    }

    /**
     * starts everything
     *
     * @throws ExecutionException
     * @throws InterruptedException
     */
    public void buildClassifier() throws InterruptedException, ExecutionException {

        boolean run = true;
        while (run) {
            ProposedUpdate proposedUpdate = this.gradientDescent.performStochasticGradientDescent(this.dataPoints,
                    this.prototypes,
                    this.omegaMatrix,
                    this.updateManager
                            .getPrototypeLearningRate(),
                    this.updateManager
                            .getOmegaLearningRate());
            run = this.updateManager.update(proposedUpdate);

        }

        // dispose thread pools
        this.gradientDescent.dispose();
    }

    public double classifyInstance(DataPoint dataPoint) {

        EmbeddedSpaceVector mappedDataPoint = dataPoint.getEmbeddedSpaceVector(this.omegaMatrix);

        // determine best matching unit
        double bmuDistance = Double.MAX_VALUE;
        int bmuIndex = -1;
        for (int prototypeIndex = 0; prototypeIndex < this.prototypes.size(); prototypeIndex++) {
            EmbeddedSpaceVector mappedPrototype = this.prototypes.get(prototypeIndex)
                    .getEmbeddedSpaceVector(this.omegaMatrix);
            double distance = LinearAlgebraicCalculations.calculateSquaredEuclideanDistance(mappedDataPoint,
                    mappedPrototype);
            if (distance < bmuDistance) {
                bmuDistance = distance;
                bmuIndex = prototypeIndex;
            }
        }

        // return BMU's class value
        return this.prototypes.get(bmuIndex).getClassLabel();
    }

    public double[] distributionForInstance(DataPoint dataPoint) {
        double[] distribution = new double[this.numberOfClasses];

        EmbeddedSpaceVector mappedDataPoint = dataPoint.getEmbeddedSpaceVector(this.omegaMatrix);
        final double originalClassLabel = mappedDataPoint.getClassLabel();
        // compute sigmoid sigma assuming data point is of each known class
        for (int classIndex = 0; classIndex < this.numberOfClasses; classIndex++) {
            // assign currently processed class label
            // TODO is there a more solid way to do this?
            mappedDataPoint.setClassLabel(classIndex);
            // we have to deregister all winning information, so it is updated
            // with regards to the new class label
            mappedDataPoint.deregisterAllWinnersBut(null);
            // compute winning information for the current class label
            WinningInformation winningInformation = mappedDataPoint.getWinningInformation(this.prototypes);
            // TODO duplicated code from DefaultCostFunction - however, this
            // dodges the whole parallel processing approach and the respective
            // overhead
            double dplus = winningInformation.getDistanceSameClass();
            double dminus = winningInformation.getDistanceOtherClass();
            double scalingFactor = Math.max(dplus + dminus, LinearAlgebraicCalculations.NUMERIC_CUTOFF);
            double sigmoidSigma = this.sigmoidFunction.evaluate((dminus - dplus) / scalingFactor);
            // TODO is the sigmoidFunction still valid here or did the
            // sigmoidSigma change?
            distribution[classIndex] = sigmoidSigma;
        }

        // normalize
        double sum = 0;
        for (int classIndex = 0; classIndex < this.numberOfClasses; classIndex++) {
            sum += distribution[classIndex];
        }
        for (int classIndex = 0; classIndex < this.numberOfClasses; classIndex++) {
            distribution[classIndex] /= sum;
        }

        // set class label again to initial value
        mappedDataPoint.setClassLabel(originalClassLabel);

        return distribution;
    }

    private void initializeMatrices() {

        // if matrix learning is enabled
        if (this.matrixLearning) {
            LOGGER.finest("initializing omega matrix with data dimension " + this.dataDimension
                    + " and omega dimension " + this.omegaDimension);
            if (this.dataDimension == this.omegaDimension) {
                LOGGER.finest("initializing omega matrix as identity matrix");
                this.omegaMatrix = new OmegaMatrix(Matrix.identity(this.omegaDimension, this.omegaDimension));
            } else {
                LOGGER.finest("initializing omega matrix with semi-random values");

                // chooses data points
                List<DataPoint> chosenTrainingData = this.dataRandomizer.generateRandomizedSubListOf(this.dataPoints,
                        DefaultSettings
                                .OMEGA_MATRIX_INITIALIZATION_AND_REGULARIZATION_NUMBER_OF_DATA_POINTS);

                double[][] dataMatrix = new double[chosenTrainingData.size()][];
                for (int index = 0; index < dataMatrix.length; index++) {
                    dataMatrix[index] = chosenTrainingData.get(index).getValues();
                }

                // compute covariance matrix and eigenvalue decomposition
                Matrix covarianceMatrix = LinearAlgebraicCalculations
                        .calculateCovarianceFromMeanVector(chosenTrainingData);
                EigenvalueDecomposition eigenvalueDecomposition = covarianceMatrix.eig();

                Matrix scaledEigenvalues = eigenvalueDecomposition.getD();
                for (int index = 0; index < scaledEigenvalues.getColumnDimension(); index++) {
                    // for (int index = scaledEigenvalues.getColumnDimension() -
                    // 1; index >= 0; index--) {
                    scaledEigenvalues.set(index, index, 1 / Math.max(scaledEigenvalues.get(index, index),
                            DefaultSettings
                                    .OMEGA_MATRIX_INITIALIZATION_MINIMAL_EXPECTED_VALUE));
                }
                this.omegaMatrix = new OmegaMatrix(eigenvalueDecomposition.getV().times(scaledEigenvalues).getMatrix(0,
                        this.omegaDimension -
                                1,
                        0,
                        this.dataDimension -
                                1));

                computeLambdaMatrix();
                normalizeOmegaMatrix();

            }

            // compute lambda matrix again if visualization is enabled
            if (this.visualization) {
                computeLambdaMatrix();
                // just for output we have to encapsulate the lambda matrix
                LOGGER.finest("initial lambda matrix:\n" + new OmegaMatrix(this.lambdaMatrix).toString());
            }

            LOGGER.finest("initial omega matrix:\n" + this.omegaMatrix.toString());
        } else {
            // create a more or less empty matrix, when no relevance learning is
            // happening
            this.omegaMatrix = new OmegaMatrix(new double[][]{{1}});
        }
    }

    private void computeLambdaMatrix() {
        this.lambdaMatrix = new OmegaMatrix(this.omegaMatrix.transpose().times(this.omegaMatrix));
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
     * initializes the appropriate number of prototypes for each class, if one
     * prototype ought to be placed the mean of the trainingData distribution is
     * chosen, else a dedicated method places the prototypes
     *
     * @throws Exception
     */
    private void initializePrototypes() {
        LOGGER.finest("initializing number of prototypes for each class");
        this.prototypes = new ArrayList<Prototype>();
        for (double classLabel : this.prototypesPerClass.keySet()) {
            List<DataPoint> dataPointsWithLabel = LinearAlgebraicCalculations
                    .collectDatapointsWithClassLabel(this.dataPoints, classLabel);
            int numberOfPrototypesToCreate = this.prototypesPerClass.get(classLabel);
            if (numberOfPrototypesToCreate == 1) {
                Prototype prototype = new Prototype(
                        LinearAlgebraicCalculations.createMeanVectorFromListOfVectors(dataPointsWithLabel), classLabel);
                LOGGER.finest("initializing one prototype for " + classLabel
                        + " (with centroid of the class values) as \n" + prototype);
                this.prototypes.add(prototype);
            } else {
                LOGGER.finest("initializing " + numberOfPrototypesToCreate + " prototypes for " + classLabel
                        + " (at the position of a datapoint of the same class)");
                initializeMultiplePrototypesForClass(dataPointsWithLabel, classLabel);
            }
        }
    }

    /**
     * initializes multiple prototypes by choosing the corresponding number of
     * prototypes, whereby no element ought to be chosen more than once
     *
     * @param dataPointsWithSameLabel the {@link Instances} of the according class label
     * @param classLabel              the label to be assigned to the prototypes
     */
    private void initializeMultiplePrototypesForClass(List<DataPoint> dataPointsWithSameLabel, double classLabel) {
        for (DataPoint randomDataPoint : this.dataRandomizer.generateRandomizedSubListOf(dataPointsWithSameLabel,
                this.prototypesPerClass.get(
                        classLabel))) {
            this.prototypes.add(new Prototype(randomDataPoint));
        }
    }

    public interface DefaultSettings {

        /**
         * the default number of epochs used for training
         */
        int DEFAULT_NUMBER_OF_EPOCHS = 2000;
        /**
         * the default number of prototypes used to represent each class
         */
        int DEFAULT_NUMBER_OF_PROTOTYPES_PER_CLASS = 1;
        /**
         * the default dimension of matrix omega
         */
        int DEFAULT_OMEGA_DIMENSION = 1;

        /**
         * number of data points used for initialization and regularization of
         * the omega matrix
         */
        int OMEGA_MATRIX_INITIALIZATION_AND_REGULARIZATION_NUMBER_OF_DATA_POINTS = 100;

        double OMEGA_MATRIX_INITIALIZATION_MINIMAL_EXPECTED_VALUE = 1.0E-4;

        double DEFAULT_LEARN_RATE_CHANGE = 0.01;
        /**
         * the default percentage of trainingData points used per round
         */
        double DEFAULT_DATA_POINT_RATIO_PER_ROUND = 0.1;
        /**
         * the default learning rate of the omega matrix
         */
        double DEFAULT_OMEGA_LEARNING_RATE = 1.0;
        /**
         * the default prototype learning rate
         */
        double DEFAULT_PROTOYPE_LEARNING_RATE = 1.0;
        double DEFAULT_SIGMOID_SIGMA_INTERVAL_START = 1.0;
        double DEFAULT_SIGMOID_SIGMA_INTERVAL_END = 10.0;
        /**
         * the default value of the stop criterion
         */
        double DEFAULT_STOP_CRITERION = 1E-9;

        /**
         * the default boolean if mode is GMLVQ (true) or GLVQ (false)
         */
        boolean DEFAULT_MATRIX_LEARNING = true;
        /**
         * {@code true} iff GMLVQ should be executed in parallel.
         **/
        boolean DEFAULT_PARALLEL_EXECUTION = true;
        /**
         * the default setting of matrix omega should be visualized
         */
        boolean DEFAULT_VISUALIZATION = true;
        CostFunctionValue DEFAULT_COST_FUNCTION = CostFunctionValue.DEFAULT_COST;
        CostFunctionValue DEFAULT_ADDITIONAL_COST_FUNCTION = CostFunctionValue.CLASSIFICATION_ERROR;
    }

    public static class Builder implements Serializable {

        private static final long serialVersionUID = 1L;

        // required
        private List<DataPoint> dataPoints;

        // optional fields
        private int numberOfEpochs = GMLVQCore.DefaultSettings.DEFAULT_NUMBER_OF_EPOCHS;
        private int numberOfPrototypesPerClass = GMLVQCore.DefaultSettings.DEFAULT_NUMBER_OF_PROTOTYPES_PER_CLASS;
        private int omegaDimension = GMLVQCore.DefaultSettings.DEFAULT_OMEGA_DIMENSION;

        private double learnRateChange = GMLVQCore.DefaultSettings.DEFAULT_LEARN_RATE_CHANGE;
        private double dataPointRationPerRound = GMLVQCore.DefaultSettings.DEFAULT_DATA_POINT_RATIO_PER_ROUND;
        private double omegaLearningRate = GMLVQCore.DefaultSettings.DEFAULT_OMEGA_LEARNING_RATE;
        private double prototypeLearningRate = GMLVQCore.DefaultSettings.DEFAULT_PROTOYPE_LEARNING_RATE;
        private double sigmoidSigmaIntervalStart = GMLVQCore.DefaultSettings.DEFAULT_SIGMOID_SIGMA_INTERVAL_START;
        private double sigmoidSigmaIntervalEnd = GMLVQCore.DefaultSettings.DEFAULT_SIGMOID_SIGMA_INTERVAL_END;
        private double stopCriterion = GMLVQCore.DefaultSettings.DEFAULT_STOP_CRITERION;

        private boolean matrixLearning = GMLVQCore.DefaultSettings.DEFAULT_MATRIX_LEARNING;
        private boolean parallelExecution = GMLVQCore.DefaultSettings.DEFAULT_PARALLEL_EXECUTION;
        private boolean visualization = GMLVQCore.DefaultSettings.DEFAULT_VISUALIZATION;

        // costs
        private CostFunctionValue costFunctionToOptimize = GMLVQCore.DefaultSettings.DEFAULT_COST_FUNCTION;
        private List<CostFunctionValue> additionalCostFunctions = new ArrayList<CostFunctionValue>();
        private double costFunctionBeta = CostFunctionCalculator.DEFAULT_BETA;
        private double[] costFunctionWeights = CostFunctionCalculator.DEFAULT_WEIGHTS;

        private Observer observer;
        private long seed = 0;
        // the following fields are set when build() is executed
        private int numberOfClasses;
        private Map<Double, Integer> prototypesPerClass;
        private int dataDimension;

        public int getNumberOfEpochs() {
            return this.numberOfEpochs;
        }

        public int getNumberOfPrototypesPerClass() {
            return this.numberOfPrototypesPerClass;
        }

        public int getOmegaDimension() {
            return this.omegaDimension;
        }

        public double getLearnRateChange() {
            return this.learnRateChange;
        }

        public double getDataPointRatioPerRound() {
            return this.dataPointRationPerRound;
        }

        public double getOmegaLearningRate() {
            return this.omegaLearningRate;
        }

        public double getPrototypeLearningRate() {
            return this.prototypeLearningRate;
        }

        public double getSigmoidSigmaIntervalStart() {
            return this.sigmoidSigmaIntervalStart;
        }

        public double getSigmoidSigmaIntervalEnd() {
            return this.sigmoidSigmaIntervalEnd;
        }

        public String getSigmoidSigmaInterval() {
            return this.sigmoidSigmaIntervalStart + "," + this.sigmoidSigmaIntervalEnd;
        }

        public double getStopCriterion() {
            return this.stopCriterion;
        }

        public boolean isMatrixLearning() {
            return this.matrixLearning;
        }

        public boolean isParallelExecution() {
            return this.parallelExecution;
        }

        public boolean isVisualization() {
            return this.visualization;
        }

        public long getSeed() {
            return this.seed;
        }

        public int getNumberOfClasses() {
            return this.numberOfClasses;
        }

        public Map<Double, Integer> getPrototypesPerClass() {
            return this.prototypesPerClass;
        }

        public int getDataDimension() {
            return this.dataDimension;
        }

        public CostFunctionValue getCostFunctionToOptimize() {
            return this.costFunctionToOptimize;
        }

        public List<CostFunctionValue> getAdditionalCostFunctions() {
            return this.additionalCostFunctions;
        }

        public double getCostFunctionBeta() {
            return this.costFunctionBeta;
        }

        public String getCostFunctionWeights() {
            return this.costFunctionWeights[0] + "," + this.costFunctionWeights[1];
        }

        public Builder numberOfEpochs(int numberOfEpochs) {
            this.numberOfEpochs = numberOfEpochs;
            return this;
        }

        public Builder numberOfPrototypesPerClass(int numberOfPrototypesPerClass) {
            this.numberOfPrototypesPerClass = numberOfPrototypesPerClass;
            return this;
        }

        public Builder omegaDimension(int omegaDimension) {
            this.omegaDimension = omegaDimension;
            return this;
        }

        public Builder learnRateChange(double learnRateChange) {
            this.learnRateChange = learnRateChange;
            return this;
        }

        public Builder dataPointRatioPerRound(double dataPointRationPerRound) {
            this.dataPointRationPerRound = dataPointRationPerRound;
            return this;
        }

        public Builder omegaLearningRate(double omegaLearningRate) {
            this.omegaLearningRate = omegaLearningRate;
            return this;
        }

        public Builder prototypeLearningRate(double prototypeLearningRate) {
            this.prototypeLearningRate = prototypeLearningRate;
            return this;
        }

        public Builder sigmoidSigmaInterval(String sigmoidSigmaIntervalString) {
            String[] split = sigmoidSigmaIntervalString.split(",");
            this.sigmoidSigmaIntervalStart = Double.parseDouble(split[0]);
            this.sigmoidSigmaIntervalEnd = Double.parseDouble(split[1]);
            return this;
        }

        public Builder sigmoidSigmaIntervalStart(double sigmoidSigmaIntervalStart) {
            this.sigmoidSigmaIntervalStart = sigmoidSigmaIntervalStart;
            return this;
        }

        public Builder sigmoidSigmaIntervalEnd(double sigmoidSigmaIntervalEnd) {
            this.sigmoidSigmaIntervalEnd = sigmoidSigmaIntervalEnd;
            return this;
        }

        public Builder stopCriterion(double stopCriterion) {
            this.stopCriterion = stopCriterion;
            return this;
        }

        public Builder matrixLearning(boolean matrixLearning) {
            this.matrixLearning = matrixLearning;
            return this;
        }

        public Builder parallelExecution(boolean parallelExecution) {
            this.parallelExecution = parallelExecution;
            return this;
        }

        public Builder visualization(boolean visualization) {
            this.visualization = visualization;
            return this;
        }

        public Builder seed(long seed) {
            this.seed = seed;
            return this;
        }

        public Builder costFunctionToOptimize(CostFunctionValue costFunctionToOptimize) {
            this.costFunctionToOptimize = costFunctionToOptimize;
            return this;
        }

        public Builder addAdditionalCostFunction(CostFunctionValue additionalCostFunctionValue) {
            this.additionalCostFunctions.add(additionalCostFunctionValue);
            return this;
        }

        public Builder costFunctionBeta(double costFunctionBeta) {
            this.costFunctionBeta = costFunctionBeta;
            return this;
        }

        public Builder costFunctionWeights(String costFunctionWeightsString) {
            String[] split = costFunctionWeightsString.split(",");
            this.costFunctionWeights[0] = Double.parseDouble(split[0]);
            this.costFunctionWeights[1] = Double.parseDouble(split[1]);
            return this;
        }

        public Builder observe(Observer observer) {
            this.observer = observer;
            return this;
        }

        /**
         * builds the classifier without showing live visualization
         *
         * @param dataPoints the training data points
         * @return a new GMLVQ instance
         * @throws InterruptedException
         * @throws ExecutionException
         */
        public GMLVQCore build(List<DataPoint> dataPoints) throws InterruptedException, ExecutionException {
            if (dataPoints == null) {
                throw new IllegalArgumentException("dataPoints cannot be null");
            }
            this.dataPoints = dataPoints;
            // extract and check the number of unique classes
            this.prototypesPerClass = new HashMap<Double, Integer>();
            for (DataPoint dataPoint : this.dataPoints) {
                this.prototypesPerClass.put(dataPoint.getClassLabel(), this.numberOfPrototypesPerClass);
            }
            this.numberOfClasses = this.prototypesPerClass.size();
            if (this.numberOfClasses < 2) {
                throw new IllegalArgumentException("number of classes cannot be smaller than 2");
            }
            // extract and check data dimension
            this.dataDimension = this.dataPoints.get(0).getDimension();
            if (this.dataDimension < this.omegaDimension) {
                throw new IllegalArgumentException("data dimension cannot be smaller than omega dimension");
            }
            // extract and check omega dimension
            if (this.omegaDimension == DefaultSettings.DEFAULT_OMEGA_DIMENSION) {
                this.omegaDimension = this.dataDimension;
            }
            // check sigmoid sigma interval
            if (this.sigmoidSigmaIntervalStart > this.sigmoidSigmaIntervalEnd) {
                throw new IllegalArgumentException("sigmoid sigma start cannot be larger than end value");
            }
            if (this.costFunctionToOptimize == null) {
                throw new IllegalArgumentException("cost function to optimize cannot be null");
            }
            if (this.numberOfClasses > 2 && anyCostFunctionRequiresConfusionMatrix()) {
                throw new IllegalArgumentException(
                        "cannot compute confusion-matrix-based cost functions for problems with " + this.numberOfClasses
                                + " classes");
            }

            // check for correct cost function weights
            if (this.costFunctionWeights[0] + this.costFunctionWeights[1] != 1.0) {
                throw new IllegalArgumentException(
                        "weights for confusion matrix based cost functions must sum up to 1.0 but are "
                                + Arrays.toString(this.costFunctionWeights));
            }

            return new GMLVQCore(this);
        }

        /**
         * builds the classifier and shows the live visualization
         *
         * @param dataPoints the training data points
         * @param instances  the training data in WEKA format (necessary to initialize visualizer)
         * @return a new GMLVQ instance
         * @throws Exception
         */
        public GMLVQCore buildAndShow(List<DataPoint> dataPoints, Instances instances) throws
                ExecutionException,
                InterruptedException {

            if (dataPoints == null) {
                throw new IllegalArgumentException("dataPoints cannot be null");
            }
            this.dataPoints = dataPoints;

            if (instances == null) {
                throw new IllegalArgumentException("WEKA instances cannot be null");
            }

            if (!visualization) {
                throw new IllegalArgumentException("use build(..) method if visualization is not wanted");
            }

            // extract and check the number of unique classes
            this.prototypesPerClass = new HashMap<Double, Integer>();
            for (DataPoint dataPoint : this.dataPoints) {
                this.prototypesPerClass.put(dataPoint.getClassLabel(), this.numberOfPrototypesPerClass);
            }
            this.numberOfClasses = this.prototypesPerClass.size();
            if (this.numberOfClasses < 2) {
                throw new IllegalArgumentException("number of classes cannot be smaller than 2");
            }
            // extract and check data dimension
            this.dataDimension = this.dataPoints.get(0).getDimension();
            if (this.dataDimension < this.omegaDimension) {
                throw new IllegalArgumentException("data dimension cannot be smaller than omega dimension");
            }
            // extract and check omega dimension
            if (this.omegaDimension == DefaultSettings.DEFAULT_OMEGA_DIMENSION) {
                this.omegaDimension = this.dataDimension;
            }
            // check sigmoid sigma interval
            if (this.sigmoidSigmaIntervalStart > this.sigmoidSigmaIntervalEnd) {
                throw new IllegalArgumentException("sigmoid sigma start cannot be larger than end value");
            }
            if (this.costFunctionToOptimize == null) {
                throw new IllegalArgumentException("cost function to optimize cannot be null");
            }
            if (this.numberOfClasses > 2 && this.anyCostFunctionRequiresConfusionMatrix()) {
                throw new IllegalArgumentException(
                        "cannot compute confusion-matrix-based cost functions for problems with " +
                                this.numberOfClasses
                                +
                                " classes");
            }

            // check for correct cost function weights
            if (this.costFunctionWeights[0] + this.costFunctionWeights[1] != 1.0) {
                throw new IllegalArgumentException(
                        "weights for confusion matrix based cost functions must sum up to 1.0 but are "
                                + Arrays.toString(this.costFunctionWeights));
            }

            // this is used to set up the Visualizer
            if (this.visualization) {
                int numberOfPrototypes = 0;
                for (int prototypes : this.prototypesPerClass.values()) {
                    numberOfPrototypes += prototypes;
                }
                final int finalNumberOfPrototypes = numberOfPrototypes;
                final Map<CostFunctionValue, Double> costFunctions = new HashMap<CostFunctionValue, Double>();
                costFunctions.put(this.costFunctionToOptimize, null);
                for (CostFunctionValue value : this.additionalCostFunctions) {
                    costFunctions.put(value, null);
                }
                this.observe(new GMLVQDefaultObserver(instances, finalNumberOfPrototypes, costFunctions));
            }

            return new GMLVQCore(this);
        }

        private boolean anyCostFunctionRequiresConfusionMatrix() {
            if (this.costFunctionToOptimize.requiresConfusionMatrix()) {
                return true;
            }
            for (CostFunctionValue additionalCostFunction : this.additionalCostFunctions) {
                if (additionalCostFunction.requiresConfusionMatrix()) {
                    return true;
                }
            }
            return false;
        }
    }
}
