package weka.classifiers.functions;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.functions.gmlvq.core.GMLVQCore;
import weka.classifiers.functions.gmlvq.core.GMLVQCore.Builder;
import weka.classifiers.functions.gmlvq.core.cost.CostFunctionCalculator;
import weka.classifiers.functions.gmlvq.core.cost.CostFunctionValue;
import weka.classifiers.functions.gmlvq.model.DataPoint;
import weka.classifiers.functions.gmlvq.model.Observer;
import weka.classifiers.functions.gmlvq.model.Prototype;
import weka.classifiers.functions.gmlvq.model.WekaModelConverter;
import weka.classifiers.functions.gmlvq.visualization.Visualizer;
import weka.core.*;
import weka.core.Capabilities.Capability;
import weka.core.matrix.Matrix;

import java.util.*;

/**
 * the adapter of {@link GMLVQCore} to weka's data structure, input options as
 * well as its GUI integration<br />
 *
 * @author S
 * @see {@link GMLVQCore} for details on GMLVQ's implementation
 */
public class GMLVQ extends AbstractClassifier
        implements TechnicalInformationHandler, Randomizable, AdditionalMeasureProducer, Observer {

    /**
     * The interface provides all default values and options essential for the
     * The interface provides all default values and options essential for the
     * algorithm.
     */
    public interface AlgorithmSettings {

        /**
         * the default number of epochs used for training
         */
        int DEFAULT_NUMBER_OF_EPOCHS = 2000;
        Option NUMBER_OF_EPOCHS_OPTION = new Option("\tnumber of maximal epochs before stop\n", "E", 1,
                "-E <number of maximal epochs>");

        /**
         * the default number of prototypes used to represent each class
         */
        int DEFAULT_NUMBER_OF_PROTOTYPES_PER_CLASS = 1;
        Option NUMBER_OF_PROTOTYPES_OPTION = new Option("\tnumber of prototypes per class\n", "P", 1,
                "-P <number of prototypes per class>");

        /**
         * the default value of the stop criterion
         */
        double DEFAULT_STOP_CRITERION = 1E-9;
        Option STOP_CRITERION_OPTION = new Option("\tstop criterion for change in cost function\n", "S", 1,
                "-S <stop criterion for change in cost function>");

        /**
         * the default setting of matrix omega should be visualized
         */
        boolean DEFAULT_VISUALIZATION = true;
        Option VISUALIZATION_OPTION = new Option("\tvisualization of relevance matrix during learning\n", "V", 0,
                "enable visualization during learning");

        /**
         * the default percentage of trainingData points used per round
         */
        double DEFAULT_DATA_POINT_RATIO_PER_ROUND = 0.1;
        Option DATA_POINTS_PER_ROUND_OPTION = new Option(
                "\tpercentage of trainingData points per round for pseudo batch\n", "R", 1,
                "-R <percentage of trainingData points per round>");

        double DEFAULT_SIGMOID_SIGMA_PERCENTAGE_CHANGE = 0.1;

        double DEFAULT_SIGMOID_SIGMA_INTERVAL_START = 1.0;

        double DEFAULT_SIGMOID_SIGMA_INTERVAL_END = 10.0;

        String DEFAULT_SIGMOID_SIGMA_INTERVAL = DEFAULT_SIGMOID_SIGMA_INTERVAL_START + ","
                + DEFAULT_SIGMOID_SIGMA_INTERVAL_END;
        Option SIGMOID_SIGMA_INTERVAL_OPTION = new Option("\tthe interval of the sigmoidFunction function\n", "I", 2,
                "-I <start,end>");
    }

    /**
     * The interface provides all default values and options essential for the
     * method in general.
     */
    public interface MethodSettings {

        /**
         * the default prototype learning rate
         */
        double DEFAULT_PROTOYPE_LEARNING_RATE = 1.0;
        Option PROTOYPE_LEARNING_RATE_OPTION = new Option("\tlearning rate of the prototypes\n", "W", 1,
                "-W <prototype learning rate>");

        /**
         * the default learning rate of the omega matrix
         */
        double DEFAULT_OMEGA_LEARNING_RATE = 1.0;
        Option OMEGA_LEARNING_RATE_OPTION = new Option(
                "\tlearning rate of the omega matrix used for relevance learning\n", "O", 1,
                "-O <omega learning rate>");

        /**
         * the default boolean if mode is GMLVQ (true) or GLVQ (false)
         */
        boolean DEFAULT_MATRIX_LEARNING = true;
        Option MATRIX_LEARNING_OPTION = new Option("\tis matrix learning\n", "M", 0, "enable matrix learning");

        /**
         * the default dimension of matrix omega
         */
        int DEFAULT_OMEGA_DIMENSION = 1;
        Option OMEGA_DIMENSION_OPTION = new Option("\tdimension of matrix omega\n", "D", 1,
                "-D <dimension of matrix omega>");

        double DEFAULT_LEARN_RATE_CHANGE = 0.01;
        Option LEARN_RATE_CHANGE_OPTION = new Option("\tthe amount the learning rate is changed\n", "L", 1,
                "-L <learning rate change>");

        /**
         * {@code true} iff GMLVQ shoud be executed in parallel.
         **/
        boolean DEFAULT_PARALLEL_EXECUTION = false;
        Option PARALLEL_EXECUTION_OPTION = new Option("\texecution in parallel\n", "X", 0,
                "enable parallel excecution");
    }

    /**
     * The interface provides all the cost function related settings.
     **/
    public interface CostFunctionsSettings {

        // declare cost functions to optimize
        Tag[] AVAILIABLE_COST_FUNCTIONS = new Tag[]{
                new Tag(CostFunctionValue.DEFAULT_COST.ordinal(), "default cost function"),
                new Tag(CostFunctionValue.CLASSIFICATION_ERROR.ordinal(), "classification error function"),
                new Tag(CostFunctionValue.FMEASURE.ordinal(), "F-measure-based cost function"),
                new Tag(CostFunctionValue.PRECISION_RECALL.ordinal(), "precision-recall cost function"),
                new Tag(CostFunctionValue.WEIGHTED_ACCURACY.ordinal(), "weighted accuracy-based cost function")};

        // declare additional cost functions
        Tag[] AVAILIABLE_ADDITIONAL_COST_FUNCTIONS = new Tag[]{
                new Tag(CostFunctionValue.NONE.ordinal(), "no additional cost function"),
                new Tag(CostFunctionValue.DEFAULT_COST.ordinal(), "default cost function"),
                new Tag(CostFunctionValue.CLASSIFICATION_ERROR.ordinal(), "classification error function"),
                new Tag(CostFunctionValue.FMEASURE.ordinal(), "F-measure-based cost function"),
                new Tag(CostFunctionValue.PRECISION_RECALL.ordinal(), "precision-recall cost function"),
                new Tag(CostFunctionValue.WEIGHTED_ACCURACY.ordinal(), "weighted accuracy-based cost function")};

        /**
         * the cost function to optimize
         */
        CostFunctionValue DEFAULT_COST_FUNCTION_TO_OPTIMIZE = CostFunctionValue.DEFAULT_COST;
        Option COST_FUNCTION_TO_OPTIMIZE_OPTION = new Option("\tcost function to optimize", "C", 1,
                "-C <cost function to optimize>");

        /**
         * the additional cost function that should be computed
         **/
        CostFunctionValue DEFAULT_ADDITIONAL_COST_FUNCTION = CostFunctionValue.NONE;
        Option ADDITIONAL_COST_FUNCTION_OPTION = new Option("\tadditional cost function to compute", "A", 1,
                "-A <additional cost function>");

        /**
         * the beta parameter used within confusion dependent cost functions
         * (currently only F-measure)
         **/
        Option COST_FUNCTION_BETA_OPTION = new Option("\tparameter used for F-measure calculation", "B", 1,
                "-B <beta>");

        /**
         * the weights used for confusion matrix based cost functions
         */
        Option COST_FUNCTION_WEIGHTS_OPTION = new Option("\tthe weights for confusion matrix based cost functions", "Y",
                1, "-Y <1st_class_weight,2nd_class_weight>");
    }

    private static final long serialVersionUID = 1L;

    public static boolean isRelevanceLearning(Matrix omegaMatrix) {
        return omegaMatrix.getColumnDimension() != 1 && omegaMatrix.getRowDimension() != 1;
    }

    /**
     * Main method for testing this class
     *
     * @param argv the commandline options
     */
    public static void main(String[] argv) {
        runClassifier(new GMLVQ(), argv);
    }

    private Builder builder;

    private int seed;

    private GMLVQCore gmlvqInstance;

    private transient Visualizer visualizer;

    public GMLVQ() {
        this.builder = new Builder();
    }

    @Override
    public void buildClassifier(Instances trainingData) throws Exception {

        getCapabilities().testWithFail(trainingData);

        final List<DataPoint> convertedTrainingData = WekaModelConverter.createDataPoints(trainingData);
        final Map<Double, String> classNamesForDouble = WekaModelConverter.extractClassLables(trainingData);
        final String[] attributeNames = WekaModelConverter.extractAttributeNames(trainingData);
        // has to happen before determining the number of prototypes
        if (this.builder.isVisualization()) {
            this.gmlvqInstance = this.builder.buildAndShow(convertedTrainingData, trainingData);
            this.builder = this.builder.observe(this);
        } else {
            this.gmlvqInstance = this.builder.build(convertedTrainingData);
        }
        int numberOfPrototypes = 0;
        for (int prototypes : this.builder.getPrototypesPerClass().values()) {
            numberOfPrototypes += prototypes;
        }
        final int finalNumberOfPrototypes = numberOfPrototypes;

        final Map<CostFunctionValue, Double> costFunctions = new HashMap<CostFunctionValue, Double>();
        costFunctions.put(this.builder.getCostFunctionToOptimize(), null);
        for (CostFunctionValue value : this.builder.getAdditionalCostFunctions()) {
            costFunctions.put(value, null);
        }

//        if (this.builder.isVisualization()) {
//            SwingUtilities.invokeAndWait(new Runnable() {
//
//                @Override
//                public void run() {
//                    GMLVQ.this.visualizer = new Visualizer(convertedTrainingData, classNamesForDouble, attributeNames,
//                            finalNumberOfPrototypes, costFunctions);
//                    GMLVQ.this.visualizer.setVisible(true);
//                }
//            });
//        }

        this.gmlvqInstance.buildClassifier();

    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        return this.gmlvqInstance.classifyInstance(WekaModelConverter.createDataPoint(instance));
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        return this.gmlvqInstance.distributionForInstance(WekaModelConverter.createDataPoint(instance));
    }

    /*
     * Methods to apply changes to user controllable options as well as tip
     * texts for the weka gui.
     */

    public String dataPointRatioPerRoundTipText() {
        return "fraction of the data that is used for each batch learning step";
    }

    @Override
    public Enumeration<String> enumerateMeasures() {
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();
        // attributes
        result.enable(Capability.NUMERIC_ATTRIBUTES);
//        result.enable(Capability.BINARY_ATTRIBUTES);
//        result.enable(Capability.NOMINAL_ATTRIBUTES);
        // class
        result.enable(Capability.NOMINAL_CLASS);
        return result;
    }

    public SelectedTag getCostFunctionToOptimize() {
        CostFunctionValue costFunctionToOptimize = this.builder.getCostFunctionToOptimize();
        SelectedTag selectedCostFunctionToOptimize = null;

        for (Tag tag : CostFunctionsSettings.AVAILIABLE_COST_FUNCTIONS) {
            if (tag.getID() == costFunctionToOptimize.ordinal()) {
                selectedCostFunctionToOptimize = new SelectedTag(tag.getID(),
                        CostFunctionsSettings.AVAILIABLE_COST_FUNCTIONS);
            }
        }

        return selectedCostFunctionToOptimize;
    }

    public SelectedTag getAdditionalCostFunction() {
        // FIXME this only works for one cost function value
        CostFunctionValue additionalCostFunction = this.builder.getAdditionalCostFunctions().isEmpty()
                ? CostFunctionValue.NONE : this.builder.getAdditionalCostFunctions().get(0);
        SelectedTag selectedCostFunctionToOptimize = null;

        for (Tag tag : CostFunctionsSettings.AVAILIABLE_ADDITIONAL_COST_FUNCTIONS) {
            if (tag.getID() == additionalCostFunction.ordinal()) {
                selectedCostFunctionToOptimize = new SelectedTag(tag.getID(),
                        CostFunctionsSettings.AVAILIABLE_ADDITIONAL_COST_FUNCTIONS);
            }
        }

        return selectedCostFunctionToOptimize;
    }

    public double getCostFunctionBeta() {
        return this.builder.getCostFunctionBeta();
    }

    public int getDataDimension() {
        return this.builder.getDataDimension();
    }

    public double getDataPointRatioPerRound() {
        return this.builder.getDataPointRatioPerRound();
    }

    public double getLearnRateChange() {
        return this.builder.getLearnRateChange();
    }

    @Override
    public double getMeasure(String measureName) {
        // TODO Auto-generated method stub
        return 0;
    }

    public int getNumberOfClasses() {
        return this.builder.getNumberOfClasses();
    }

    public int getNumberOfEpochs() {
        return this.builder.getNumberOfEpochs();
    }

    public int getNumberOfPrototypesPerClass() {
        return this.builder.getNumberOfPrototypesPerClass();
    }

    public int getOmegaDimension() {
        return this.builder.getOmegaDimension();
    }

    public double getOmegaLearningRate() {
        return this.builder.getOmegaLearningRate();
    }

    @Override
    public String[] getOptions() {

        Vector<String> commandLine = new Vector<String>();

        // algorithm settings
        commandLine.add("-" + AlgorithmSettings.NUMBER_OF_EPOCHS_OPTION.name());
        commandLine.add("" + this.builder.getNumberOfEpochs());
        commandLine.add("-" + AlgorithmSettings.NUMBER_OF_PROTOTYPES_OPTION.name());
        commandLine.add("" + this.builder.getNumberOfPrototypesPerClass());
        commandLine.add("-" + AlgorithmSettings.STOP_CRITERION_OPTION.name());
        commandLine.add("" + this.builder.getStopCriterion());
        if (this.builder.isVisualization()) {
            commandLine.add("-" + AlgorithmSettings.VISUALIZATION_OPTION.name());
        }
        commandLine.add("-" + AlgorithmSettings.DATA_POINTS_PER_ROUND_OPTION.name());
        commandLine.add("" + this.builder.getDataPointRatioPerRound());
        commandLine.add("-" + AlgorithmSettings.SIGMOID_SIGMA_INTERVAL_OPTION.name());
        commandLine.add("" + this.builder.getSigmoidSigmaInterval());

        // method settings
        commandLine.add("-" + MethodSettings.PROTOYPE_LEARNING_RATE_OPTION.name());
        commandLine.add("" + this.builder.getPrototypeLearningRate());
        commandLine.add("-" + MethodSettings.OMEGA_LEARNING_RATE_OPTION.name());
        commandLine.add("" + this.builder.getOmegaLearningRate());
        commandLine.add("-" + MethodSettings.OMEGA_DIMENSION_OPTION.name());
        commandLine.add("" + this.builder.getOmegaDimension());
        if (this.builder.isMatrixLearning()) {
            commandLine.add("-" + MethodSettings.MATRIX_LEARNING_OPTION.name());
        }
        commandLine.add("-" + MethodSettings.LEARN_RATE_CHANGE_OPTION.name());
        commandLine.add("" + this.builder.getLearnRateChange());
        if (this.builder.isParallelExecution()) {
            commandLine.add("-" + MethodSettings.PARALLEL_EXECUTION_OPTION.name());
        }

        // cost function settings
        commandLine.add("-" + CostFunctionsSettings.COST_FUNCTION_TO_OPTIMIZE_OPTION.name());
        commandLine.add("" + this.builder.getCostFunctionToOptimize().ordinal());
        commandLine.add("-" + CostFunctionsSettings.ADDITIONAL_COST_FUNCTION_OPTION.name());
        commandLine.add("" + (this.builder.getAdditionalCostFunctions().isEmpty() ? CostFunctionValue.NONE.ordinal()
                : this.builder.getAdditionalCostFunctions().get(0).ordinal()));
        commandLine.add("-" + CostFunctionsSettings.COST_FUNCTION_BETA_OPTION.name());
        commandLine.add("" + this.builder.getCostFunctionBeta());
        commandLine.add("-" + CostFunctionsSettings.COST_FUNCTION_WEIGHTS_OPTION.name());
        commandLine.add("" + this.builder.getCostFunctionWeights());
        Collections.addAll(commandLine, super.getOptions());

        return commandLine.toArray(new String[0]);

    }

    public double getPrototypeLearningRate() {
        return this.builder.getPrototypeLearningRate();
    }

    public Map<Double, Integer> getPrototypesPerClass() {
        return this.builder.getPrototypesPerClass();
    }

    @Override
    public int getSeed() {
        return (int) this.seed;
    }

    public String getSigmoidSigmaInterval() {
        return this.builder.getSigmoidSigmaInterval();
    }

    public String getCostFunctionWeights() {
        return this.builder.getCostFunctionWeights();
    }

    public double getStopCriterion() {
        return this.builder.getStopCriterion();
    }

    @Override
    public TechnicalInformation getTechnicalInformation() {
        // TODO add publication and stuff
        return null;
    }

    /**
     * Returns a string describing classifier
     *
     * @return a description suitable for displaying in the
     * explorer/experimenter gui
     */
    public String globalInfo() {
        return "An implementation of the GMLVQ and GLVQ algorithm.";
    }

    public boolean isMatrixLearning() {
        return this.builder.isMatrixLearning();
    }

    public boolean isParallelExecution() {
        return this.builder.isParallelExecution();
    }

    public boolean isVisualization() {
        return this.builder.isVisualization();
    }

    public String learnRateChangeTipText() {
        return "the change of the learning rate";
    }

    @Override
    public Enumeration<Option> listOptions() {

        Vector<Option> options = new Vector<Option>();

        // algorithm settings
        options.addElement(AlgorithmSettings.NUMBER_OF_EPOCHS_OPTION);
        options.addElement(AlgorithmSettings.NUMBER_OF_PROTOTYPES_OPTION);
        options.addElement(AlgorithmSettings.STOP_CRITERION_OPTION);
        options.addElement(AlgorithmSettings.VISUALIZATION_OPTION);
        options.addElement(AlgorithmSettings.DATA_POINTS_PER_ROUND_OPTION);
        options.addElement(AlgorithmSettings.SIGMOID_SIGMA_INTERVAL_OPTION);

        // method settings
        options.addElement(MethodSettings.MATRIX_LEARNING_OPTION);
        options.addElement(MethodSettings.OMEGA_DIMENSION_OPTION);
        options.addElement(MethodSettings.PROTOYPE_LEARNING_RATE_OPTION);
        options.addElement(MethodSettings.OMEGA_LEARNING_RATE_OPTION);
        options.addElement(MethodSettings.LEARN_RATE_CHANGE_OPTION);
        options.addElement(MethodSettings.PARALLEL_EXECUTION_OPTION);

        // cost function settings
        options.addElement(CostFunctionsSettings.COST_FUNCTION_TO_OPTIMIZE_OPTION);
        options.addElement(CostFunctionsSettings.ADDITIONAL_COST_FUNCTION_OPTION);
        options.addElement(CostFunctionsSettings.COST_FUNCTION_BETA_OPTION);
        options.addElement(CostFunctionsSettings.COST_FUNCTION_WEIGHTS_OPTION);

        // add options of super class
        options.addAll(Collections.list(super.listOptions()));

        return options.elements();
    }

    public String matrixLearningTipText() {
        return "specifies if matrix learning should be performed";
    }

    public String numberOfEpochsTipText() {
        return "the number of epochs to be calculated";
    }

    public String numberOfPrototypesPerClassTipText() {
        return "the default number of prototypes to be used to represent each class";
    }

    public String omegaDimensionTipText() {
        return "the explicit setting of the dimension of the omega matrix, if set to 1 data dimension will be used";
    }

    public String omegaLearningRateTipText() {
        return "learning rate used for the omega matrix";
    }

    public String parallelExecutionTipText() {
        return "determines wheter the the classifier is build in parallel or not";
    }

    public String prototypeLearningRateTipText() {
        return "the learning rate used for prototype learning";
    }

    public String additionalCostFunctionTipText() {
        return "choose an additional cost function to optimize";
    }

    public String costFunctionToOptimizeTipText() {
        return "choose what cost function is used to guide training";
    }

    public String costFunctionBetaTipText() {
        return "this parameter is currently only used by F-measure cost function";
    }

    public String costFunctionWeightsTipText() {
        return "this parameter is a weight vector used for confusion matrix based cost functions";
    }

    public String seedTipText() {
        return "seed to be used for the random number generator";
    }

    public String sigmoidSigmaIntervalTipText() {
        return "the interval of the sigmoid and sigmoid prime function that is used for calculation";
    }

    public String stopCriterionTipText() {
        return "the stop criterion when learning should stop";
    }

    public String visualizationTipText() {
        return "determines if the progress should be visualized";
    }

    public void setDataPointRatioPerRound(double dataPointRatioPerRound) {
        this.builder.dataPointRatioPerRound(dataPointRatioPerRound);
    }

    public void setLearnRateChange(double learnRateChange) {
        this.builder.learnRateChange(learnRateChange);

    }

    public void setMatrixLearning(boolean matrixLearning) {
        this.builder.matrixLearning(matrixLearning);

    }

    public void setNumberOfEpochs(int numberOfEpochs) {
        this.builder.numberOfEpochs(numberOfEpochs);

    }

    public void setNumberOfPrototypesPerClass(int numberOfPrototypesPerClass) {
        this.builder.numberOfPrototypesPerClass(numberOfPrototypesPerClass);
    }

    public void setOmegaDimension(int omegaDimension) {
        this.builder.omegaDimension(omegaDimension);
    }

    public void setOmegaLearningRate(double omegaLearningRate) {
        this.builder.omegaLearningRate(omegaLearningRate);
    }

    public void setCostFunctionToOptimize(SelectedTag costFunctionToOptimizeTag) {
        if (costFunctionToOptimizeTag.getTags() == CostFunctionsSettings.AVAILIABLE_COST_FUNCTIONS) {
            for (CostFunctionValue costFunctionValue : CostFunctionValue.values()) {
                if (costFunctionValue.ordinal() == costFunctionToOptimizeTag.getSelectedTag().getID()) {
                    this.builder.costFunctionToOptimize(costFunctionValue);
                }
            }
        }
    }

    public void setAdditionalCostFunction(SelectedTag additionalCostFunctionTag) {
        if (additionalCostFunctionTag.getTags() == CostFunctionsSettings.AVAILIABLE_ADDITIONAL_COST_FUNCTIONS) {
            for (CostFunctionValue costFunctionValue : CostFunctionValue.values()) {
                if (costFunctionValue.ordinal() == additionalCostFunctionTag.getSelectedTag().getID()
                        && costFunctionValue != CostFunctionValue.NONE) {
                    this.builder.addAdditionalCostFunction(costFunctionValue);
                }
            }
        }
    }

    public void setCostFunctionBeta(double costFunctionBeta) {
        this.builder.costFunctionBeta(costFunctionBeta);
    }

    @Override
    public void setOptions(String[] options) throws Exception {

        // algorithm settings
        String numberOfEpochsString = Utils.getOption(AlgorithmSettings.NUMBER_OF_EPOCHS_OPTION.name().charAt(0),
                options);
        if (numberOfEpochsString.length() != 0) {
            this.builder.numberOfEpochs(Integer.parseInt(numberOfEpochsString));
        } else {
            this.builder.numberOfEpochs(AlgorithmSettings.DEFAULT_NUMBER_OF_EPOCHS);
        }

        String numberOfPrototypesString = Utils
                .getOption(AlgorithmSettings.NUMBER_OF_PROTOTYPES_OPTION.name().charAt(0), options);
        if (numberOfPrototypesString.length() != 0) {
            this.builder.numberOfPrototypesPerClass(Integer.parseInt(numberOfPrototypesString));
        } else {
            this.builder.numberOfPrototypesPerClass(AlgorithmSettings.DEFAULT_NUMBER_OF_PROTOTYPES_PER_CLASS);
        }

        String stopCriterionString = Utils.getOption(AlgorithmSettings.STOP_CRITERION_OPTION.name().charAt(0), options);
        if (stopCriterionString.length() != 0) {
            this.builder.stopCriterion(Double.parseDouble(stopCriterionString));
        } else {
            this.builder.stopCriterion(AlgorithmSettings.DEFAULT_STOP_CRITERION);
        }

        this.builder.visualization = Utils.getFlag(AlgorithmSettings.VISUALIZATION_OPTION.name().charAt(0), options);

        String dataPointsPerRoundString = Utils
                .getOption(AlgorithmSettings.DATA_POINTS_PER_ROUND_OPTION.name().charAt(0), options);
        if (dataPointsPerRoundString.length() != 0) {
            this.builder.dataPointRatioPerRound(Double.parseDouble(dataPointsPerRoundString));
        } else {
            this.builder.dataPointRatioPerRound(AlgorithmSettings.DEFAULT_DATA_POINT_RATIO_PER_ROUND);
        }

        String sigmoidSigmaIntervalString = Utils
                .getOption(AlgorithmSettings.SIGMOID_SIGMA_INTERVAL_OPTION.name().charAt(0), options);
        if (sigmoidSigmaIntervalString.length() != 0) {
            this.builder.sigmoidSigmaInterval(sigmoidSigmaIntervalString);
        } else {
            this.builder.sigmoidSigmaInterval(AlgorithmSettings.DEFAULT_SIGMOID_SIGMA_INTERVAL);
            // initialize with sigmoid sigma default values - not needed anymore
            // - probably
            // this.embeddedSpaceCalculator.setSigmoidSigmaInterval(AlgorithmSettings.DEFAULT_SIGMOID_SIGMA_INTERVAL_START,
            // AlgorithmSettings.DEFAULT_SIGMOID_SIGMA_INTERVAL_END);
        }

        // method settings
        String prototypeLearningRateString = Utils
                .getOption(MethodSettings.PROTOYPE_LEARNING_RATE_OPTION.name().charAt(0), options);
        if (prototypeLearningRateString.length() != 0) {
            this.builder.prototypeLearningRate(Double.parseDouble(prototypeLearningRateString));
        } else {
            this.builder.prototypeLearningRate(MethodSettings.DEFAULT_PROTOYPE_LEARNING_RATE);
        }

        String omegaLearningRateString = Utils.getOption(MethodSettings.OMEGA_LEARNING_RATE_OPTION.name().charAt(0),
                options);
        if (omegaLearningRateString.length() != 0) {
            this.builder.omegaLearningRate(Double.parseDouble(omegaLearningRateString));
        } else {
            this.builder.omegaLearningRate(MethodSettings.DEFAULT_OMEGA_LEARNING_RATE);
        }

        this.builder.matrixLearning(Utils.getFlag(MethodSettings.MATRIX_LEARNING_OPTION.name().charAt(0), options));

        String omegaDimensionString = Utils.getOption(MethodSettings.OMEGA_DIMENSION_OPTION.name().charAt(0), options);
        if (omegaDimensionString.length() != 0) {
            this.builder.omegaDimension(Integer.parseInt(omegaDimensionString));
        } else {
            this.builder.omegaDimension(MethodSettings.DEFAULT_OMEGA_DIMENSION);
        }

        String learnRateChangeString = Utils.getOption(MethodSettings.LEARN_RATE_CHANGE_OPTION.name().charAt(0),
                options);
        if (omegaDimensionString.length() != 0) {
            this.builder.learnRateChange(Double.parseDouble(learnRateChangeString));
        } else {
            this.builder.learnRateChange(MethodSettings.DEFAULT_OMEGA_DIMENSION);
        }

        this.builder
                .parallelExecution(Utils.getFlag(MethodSettings.PARALLEL_EXECUTION_OPTION.name().charAt(0), options));

        // cost function settings
        String costFunctionToOptimizeString = Utils
                .getOption(CostFunctionsSettings.COST_FUNCTION_TO_OPTIMIZE_OPTION.name().charAt(0), options);
        if (costFunctionToOptimizeString.length() != 0) {
            SelectedTag costFunctionToOptimizeTag = new SelectedTag(Integer.parseInt(costFunctionToOptimizeString),
                    CostFunctionsSettings.AVAILIABLE_COST_FUNCTIONS);
            setCostFunctionToOptimize(costFunctionToOptimizeTag);
        } else {
            this.builder.costFunctionToOptimize(CostFunctionsSettings.DEFAULT_COST_FUNCTION_TO_OPTIMIZE);
        }

        String additionalCostFunctionString = Utils
                .getOption(CostFunctionsSettings.ADDITIONAL_COST_FUNCTION_OPTION.name().charAt(0), options);
        if (additionalCostFunctionString.length() != 0) {
            SelectedTag additionalCostFunctionTag = new SelectedTag(Integer.parseInt(additionalCostFunctionString),
                    CostFunctionsSettings.AVAILIABLE_ADDITIONAL_COST_FUNCTIONS);
            setAdditionalCostFunction(additionalCostFunctionTag);
        }

        String costFunctionBetaString = Utils
                .getOption(CostFunctionsSettings.COST_FUNCTION_BETA_OPTION.name().charAt(0), options);
        if (costFunctionBetaString.length() != 0) {
            this.builder.costFunctionBeta(Double.parseDouble(costFunctionBetaString));
        } else {
            this.builder.costFunctionBeta(CostFunctionCalculator.DEFAULT_BETA);
        }

        super.setOptions(options);
        // Utils.checkForRemainingOptions(options);
    }

    public void setParallelExecution(boolean parallelExecution) {
        this.builder.parallelExecution(parallelExecution);

    }

    public void setPrototypeLearningRate(double prototypeLearningRate) {
        this.builder.prototypeLearningRate(prototypeLearningRate);

    }

    @Override
    public void setSeed(int seed) {
        this.seed = seed;
        this.builder.seed((long) this.seed);
    }

    public void setSigmoidSigmaInterval(String sigmoidSigmaIntervalString) {
        this.builder.sigmoidSigmaInterval(sigmoidSigmaIntervalString);

    }

    public void setCostFunctionWeights(String costFunctionWeightsString) {
        this.builder.costFunctionWeights(costFunctionWeightsString);
    }

    public void setStopCriterion(double stopCriterion) {
        this.builder.stopCriterion(stopCriterion);

    }

    public void setVisualization(boolean visualization) {
        this.builder.visualization = visualization;

    }

    @Override
    public void updatePrototypes(List<Prototype> prototypes) {
        this.visualizer.updatePrototypes(prototypes);
    }

    @Override
    public void updateCostFunctions(Map<CostFunctionValue, Double> currentCostValues) {
        this.visualizer.updateCostFunctions(currentCostValues);
    }

    @Override
    public void updateLambdaMatrix(Matrix lambdaMatrix) {
        this.visualizer.updateLambdaMatrix(lambdaMatrix);
    }
}
