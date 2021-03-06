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
import weka.classifiers.functions.gmlvq.visualization.VisualizationSingleton;
import weka.classifiers.functions.gmlvq.visualization.Visualizer;
import weka.core.*;
import weka.core.Capabilities.Capability;
import weka.core.matrix.Matrix;

import javax.swing.*;
import java.util.*;

/**
 * the adapter of {@link GMLVQCore} to weka's data structure, input options as
 * well as its GUI integration<br />
 * see {@link GMLVQCore} for details on GMLVQ's implementation
 *
 * @author S
 */
public class GMLVQ extends AbstractClassifier implements TechnicalInformationHandler, Observer {

    /**
     * The interface provides all default values and options essential for the
     * algorithm.
     */
    public interface AlgorithmSettings {

        /**
         * the default number of epochs used for training
         */
        int DEFAULT_NUMBER_OF_EPOCHS = 500;
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
        double DEFAULT_DATA_POINT_RATIO_PER_ROUND = 0.75;
        Option DATA_POINTS_PER_ROUND_OPTION = new Option(
                "\tpercentage of data points per round for pseudo batch\n", "R", 1,
                "-R <percentage of data points per round>");

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
                new Tag(CostFunctionValue.CLASSIFICATION_ACCURACY.ordinal(), "classification accuracy function"),
                new Tag(CostFunctionValue.FMEASURE.ordinal(), "F-measure-based cost function"),
                new Tag(CostFunctionValue.PRECISION_RECALL.ordinal(), "precision-recall cost function"),
                new Tag(CostFunctionValue.WEIGHTED_ACCURACY.ordinal(), "weighted accuracy-based cost function")};

        // declare additional cost functions
        Tag[] AVAILIABLE_ADDITIONAL_COST_FUNCTIONS = new Tag[]{
                new Tag(CostFunctionValue.NONE.ordinal(), "no additional cost function"),
                new Tag(CostFunctionValue.DEFAULT_COST.ordinal(), "default cost function"),
                new Tag(CostFunctionValue.CLASSIFICATION_ACCURACY.ordinal(), "classification accuracy function"),
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

        /**
         * additional classification functions
         */
        Option VISUALIZE_CLASSIFICATION_ACCURACY = new Option("visualize the classification accuracy", "VC", 0, "visualize the classification accuracy");
        Option VISUALIZE_WEIGHTED_ACCURACY = new Option("visualize the weighted accuracy", "VA", 0, "visualize the weighted accuracy");
        Option VISUALIZE_FMEASURE = new Option("visualize the f-measure", "VF", 0, "visualize the f measure");
        Option VISUALIZE_PRECISION_RECALL = new Option("visualize precision recall", "VP", 0, "visualize precision recall");
        Option VISUALIZE_DEFAULT_COST = new Option("visualize GMLVQ default cost function", "VD", 0, "visualize the GMLVQ default cost function");
    }

    private static final long serialVersionUID = 1L;

    private Builder builder;

    private GMLVQCore gmlvqInstance;

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
            this.builder = this.builder.observe(this);
        }
        this.gmlvqInstance = this.builder.build(convertedTrainingData);
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

        if (this.builder.isVisualization()) {
            SwingUtilities.invokeAndWait(() -> {
                VisualizationSingleton.addVisualization(new Visualizer(gmlvqInstance, convertedTrainingData, classNamesForDouble, attributeNames, finalNumberOfPrototypes, costFunctions));
                VisualizationSingleton.showVisualizations();
            });
            updatePrototypes(this.gmlvqInstance.getPrototypes());
            updateLambdaMatrix(this.gmlvqInstance.getLambdaMatrix());
        }

        this.gmlvqInstance.buildClassifier();

    }

    @Override
    public double classifyInstance(Instance instance) {
        return this.gmlvqInstance.classifyInstance(WekaModelConverter.createDataPoint(instance));
    }

    @Override
    public double[] distributionForInstance(Instance instance) {
        return this.gmlvqInstance.distributionForInstance(WekaModelConverter.createDataPoint(instance));
    }

    @Override
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();
        // attributes
        result.enable(Capability.NUMERIC_ATTRIBUTES);
        // class
        result.enable(Capability.NOMINAL_CLASS);
        return result;
    }

    public SelectedTag get_1_costFunctionToOptimize() {
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

    public double get_2_costFunctionBeta() {
        return this.builder.getCostFunctionBeta();
    }

    public int getDataDimension() {
        return this.builder.getDataDimension();
    }

    public double get_2_dataPointRatioPerRound() {
        return this.builder.getDataPointRatioPerRound();
    }

    public int getNumberOfClasses() {
        return this.builder.getNumberOfClasses();
    }

    public int get_1_numberOfEpochs() {
        return this.builder.getNumberOfEpochs();
    }

    public int get_1_numberOfPrototypesPerClass() {
        return this.builder.getNumberOfPrototypesPerClass();
    }

    public int get_2_omegaDimension() {
        return this.builder.getOmegaDimension();
    }

    public double get_2_omegaLearningRate() {
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

        // cost function settings
        commandLine.add("-" + CostFunctionsSettings.COST_FUNCTION_TO_OPTIMIZE_OPTION.name());
        commandLine.add("" + this.builder.getCostFunctionToOptimize().ordinal());
        if (this.builder.isVisualizingClassificationAccuracy()) {
            commandLine.add("-" + CostFunctionsSettings.VISUALIZE_CLASSIFICATION_ACCURACY.name());
        }
        if (this.builder.isVisualizingWeightedAccuracy()) {
            commandLine.add("-" + CostFunctionsSettings.VISUALIZE_WEIGHTED_ACCURACY.name());
        }
        if (this.builder.isVisualizingFMeasure()) {
            commandLine.add("-" + CostFunctionsSettings.VISUALIZE_FMEASURE.name());
        }
        if (this.builder.isVisualizingPrecisionRecall()) {
            commandLine.add("-" + CostFunctionsSettings.VISUALIZE_PRECISION_RECALL.name());
        }
        if (this.builder.isVisualizingDefaultCost()) {
            commandLine.add("-" + CostFunctionsSettings.VISUALIZE_DEFAULT_COST.name());
        }

        commandLine.add("-" + CostFunctionsSettings.COST_FUNCTION_BETA_OPTION.name());
        commandLine.add("" + this.builder.getCostFunctionBeta());
        commandLine.add("-" + CostFunctionsSettings.COST_FUNCTION_WEIGHTS_OPTION.name());
        commandLine.add("" + this.builder.getCostFunctionWeights());
        Collections.addAll(commandLine, super.getOptions());

        return commandLine.toArray(new String[0]);

    }

    public double get_2_prototypeLearningRate() {
        return this.builder.getPrototypeLearningRate();
    }

    public Map<Double, Integer> getPrototypesPerClass() {
        return this.builder.getPrototypesPerClass();
    }

    public String get_2_sigmoidSigmaInterval() {
        return this.builder.getSigmoidSigmaInterval();
    }

    public String get_2_costFunctionWeights() {
        return this.builder.getCostFunctionWeights();
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

    public boolean is_2_matrixLearning() {
        return this.builder.isMatrixLearning();
    }

    public boolean is_2_parallelExecution() {
        return this.builder.isParallelExecution();
    }

    public boolean is_1_visualization() {
        return this.builder.isVisualization();
    }

    public boolean is_3_visualizeClassificationAccuracy() {
        return this.builder.isVisualizingClassificationAccuracy();
    }

    public boolean is_3_visualizeWeightedAccuracy() {
        return this.builder.isVisualizingWeightedAccuracy();
    }

    public boolean is_3_visualizeFMeasure() {
        return this.builder.isVisualizingFMeasure();
    }

    public boolean is_3_visualizePrecisionRecall() {
        return this.builder.isVisualizingPrecisionRecall();
    }

    public boolean is_3_visualizeDefaultCost() {
        return this.builder.isVisualizingDefaultCost();
    }

    public static boolean isRelevanceLearning(Matrix omegaMatrix) {
        return omegaMatrix.getColumnDimension() != 1 && omegaMatrix.getRowDimension() != 1;
    }

    @Override
    public Enumeration<Option> listOptions() {

        Vector<Option> options = new Vector<Option>();

        // algorithm settings
        options.addElement(AlgorithmSettings.NUMBER_OF_EPOCHS_OPTION);
        options.addElement(AlgorithmSettings.NUMBER_OF_PROTOTYPES_OPTION);
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
        options.addElement(CostFunctionsSettings.VISUALIZE_CLASSIFICATION_ACCURACY);
        options.addElement(CostFunctionsSettings.VISUALIZE_WEIGHTED_ACCURACY);
        options.addElement(CostFunctionsSettings.VISUALIZE_FMEASURE);
        options.addElement(CostFunctionsSettings.VISUALIZE_PRECISION_RECALL);
        options.addElement(CostFunctionsSettings.VISUALIZE_DEFAULT_COST);

        options.addElement(CostFunctionsSettings.COST_FUNCTION_BETA_OPTION);
        options.addElement(CostFunctionsSettings.COST_FUNCTION_WEIGHTS_OPTION);

        // add options of super class
        options.addAll(Collections.list(super.listOptions()));

        return options.elements();
    }

    public String _2_matrixLearningTipText() {
        return "if enabled a mapping matrix is adapted beside the prototypes";
    }

    public String _1_numberOfEpochsTipText() {
        return "number of epochs/rounds to be performed for training";
    }

    public String _1_numberOfPrototypesPerClassTipText() {
        return "number of prototypes to be used to represent each class  (if only one number is given, each class gets this number of prototypes";
    }

    public String _2_omegaDimensionTipText() {
        return "explicit setting of the dimension of the mapping matrix, if set to 1 data dimension will be used";
    }

    public String _2_omegaLearningRateTipText() {
        return "learning rate used for learning of the mapping matrix";
    }

    public String _2_parallelExecutionTipText() {
        return "determines whether the the classifier is build in parallel or not";
    }

    public String _2_prototypeLearningRateTipText() {
        return "learning rate used for prototype learning";
    }

    public String _1_costFunctionToOptimizeTipText() {
        return "choose what cost function is used to guide training";
    }

    public String _2_costFunctionBetaTipText() {
        return "parameter of the F-measure";
    }

    public String _2_costFunctionWeightsTipText() {
        return "vector with weights of the importance of each class";
    }

    public String seedTipText() {
        return "seed to be used for the random number generator";
    }

    public String _2_sigmoidSigmaIntervalTipText() {
        return "interval of the parameter of the sigmoid/Fermit  function  which is part of the cost function";
    }

    public String _1_visualizationTipText() {
        return "determines if the progress should be visualized";
    }

    public String _2_dataPointRatioPerRoundTipText() {
        return "percentage of data which are used to perform one update step in one epoch";
    }

    public String _3_visualizeClassificationAccuracyTipText() {
        return "calculate and display classification accuracy";
    }

    public String _3_visualizeWeightedAccuracyTipText() {
        return "calculate and display weighted accuracy";
    }

    public String _3_visualizeFMeasureTipText() {
        return "calculate and display f-measure";
    }

    public String _3_visualizePrecisionRecallTipText() {
        return "calculate and display precision recall";
    }

    public String _3_visualizeDefaultCostTipText() {
        return "calculate and display default cost";
    }

    public void set_2_dataPointRatioPerRound(double dataPointRatioPerRound) {
        this.builder.dataPointRatioPerRound(dataPointRatioPerRound);
    }

    public void set_2_matrixLearning(boolean matrixLearning) {
        this.builder.matrixLearning(matrixLearning);

    }

    public void set_1_numberOfEpochs(int numberOfEpochs) {
        this.builder.numberOfEpochs(numberOfEpochs);

    }

    public void set_1_numberOfPrototypesPerClass(int numberOfPrototypesPerClass) {
        this.builder.numberOfPrototypesPerClass(numberOfPrototypesPerClass);
    }

    public void set_2_omegaDimension(int omegaDimension) {
        this.builder.omegaDimension(omegaDimension);
    }

    public void set_2_omegaLearningRate(double omegaLearningRate) {
        this.builder.omegaLearningRate(omegaLearningRate);
    }

    public void set_3_visualizeClassificationAccuracy(boolean visualize) {
        this.builder.visualizeClassificationAccuracy(visualize);
    }

    public void set_3_visualizeWeightedAccuracy(boolean visualize) {
        this.builder.visualizeWeightedAccuracy(visualize);
    }

    public void set_3_visualizeFMeasure(boolean visualize) {
        this.builder.visualizeFMeasure(visualize);
    }

    public void set_3_visualizePrecisionRecall(boolean visualize) {
        this.builder.visualizePrecisionRecall(visualize);
    }

    public void set_3_visualizeDefaultCost(boolean visualize) {
        this.builder.visualizeDefaultCost(visualize);
    }

    public void set_1_costFunctionToOptimize(SelectedTag costFunctionToOptimizeTag) {
        if (costFunctionToOptimizeTag.getTags() == CostFunctionsSettings.AVAILIABLE_COST_FUNCTIONS) {
            for (CostFunctionValue costFunctionValue : CostFunctionValue.values()) {
                if (costFunctionValue.ordinal() == costFunctionToOptimizeTag.getSelectedTag().getID()) {
                    this.builder.costFunctionToOptimize(costFunctionValue);
                }
            }
        }
    }

    public void setCostFunctionToOptimize(CostFunctionValue costFunctionValue) {
        this.builder.costFunctionToOptimize(costFunctionValue);
    }

    public void addAdditionalCostFunction(CostFunctionValue costFunctionValue) {
        if (costFunctionValue != CostFunctionValue.NONE) {
            this.builder.addAdditionalCostFunction(costFunctionValue);
        }
    }

    public void set_2_costFunctionBeta(double costFunctionBeta) {
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

        this.builder.visualization(Utils.getFlag(AlgorithmSettings.VISUALIZATION_OPTION.name().charAt(0), options));

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

        this.builder
                .parallelExecution(Utils.getFlag(MethodSettings.PARALLEL_EXECUTION_OPTION.name().charAt(0), options));

        // cost function settings
        String costFunctionToOptimizeString = Utils
                .getOption(CostFunctionsSettings.COST_FUNCTION_TO_OPTIMIZE_OPTION.name().charAt(0), options);
        if (costFunctionToOptimizeString.length() != 0) {
            SelectedTag costFunctionToOptimizeTag = new SelectedTag(Integer.parseInt(costFunctionToOptimizeString),
                    CostFunctionsSettings.AVAILIABLE_COST_FUNCTIONS);
            set_1_costFunctionToOptimize(costFunctionToOptimizeTag);
        } else {
            this.builder.costFunctionToOptimize(CostFunctionsSettings.DEFAULT_COST_FUNCTION_TO_OPTIMIZE);
        }

        String costFunctionBetaString = Utils
                .getOption(CostFunctionsSettings.COST_FUNCTION_BETA_OPTION.name().charAt(0), options);
        if (costFunctionBetaString.length() != 0) {
            this.builder.costFunctionBeta(Double.parseDouble(costFunctionBetaString));
        } else {
            this.builder.costFunctionBeta(CostFunctionCalculator.DEFAULT_BETA);
        }

        String costFunctionWeightString = Utils
                .getOption(CostFunctionsSettings.COST_FUNCTION_WEIGHTS_OPTION.name().charAt(0), options);
        if(costFunctionWeightString.length() != 0) {
            this.builder.costFunctionWeights(costFunctionWeightString);
        } else {
            this.builder.costFunctionWeights(CostFunctionCalculator.DEFAULT_WEIGHTS);
        }

        this.builder
                .visualizeDefaultCost(Utils.getFlag(CostFunctionsSettings.VISUALIZE_DEFAULT_COST.name(), options));
        this.builder
                .visualizeFMeasure(Utils.getFlag(CostFunctionsSettings.VISUALIZE_FMEASURE.name(), options));
        this.builder
                .visualizePrecisionRecall(Utils.getFlag(CostFunctionsSettings.VISUALIZE_PRECISION_RECALL.name(), options));
        this.builder
                .visualizeWeightedAccuracy(Utils.getFlag(CostFunctionsSettings.VISUALIZE_WEIGHTED_ACCURACY.name(), options));
        this.builder
                .visualizeClassificationAccuracy(Utils.getFlag(CostFunctionsSettings.VISUALIZE_CLASSIFICATION_ACCURACY.name(), options));

        super.setOptions(options);
    }

    public void set_2_ParallelExecution(boolean parallelExecution) {
        this.builder.parallelExecution(parallelExecution);

    }

    public void set_2_prototypeLearningRate(double prototypeLearningRate) {
        this.builder.prototypeLearningRate(prototypeLearningRate);

    }

    public void set_2_sigmoidSigmaInterval(String sigmoidSigmaIntervalString) {
        this.builder.sigmoidSigmaInterval(sigmoidSigmaIntervalString);

    }

    public void set_2_costFunctionWeights(String costFunctionWeightsString) {
        this.builder.costFunctionWeights(costFunctionWeightsString);
    }

    public void set_1_visualization(boolean visualization) {
        this.builder.visualization(visualization);

    }

    @Override
    public void updatePrototypes(List<Prototype> prototypes) {
        VisualizationSingleton.getLastVisualizalizer().updatePrototypes(prototypes);
    }

    @Override
    public void updateCostFunctions(Map<CostFunctionValue, Double> currentCostValues) {
        VisualizationSingleton.getLastVisualizalizer().updateCostFunctions(currentCostValues);
    }

    @Override
    public void updateLambdaMatrix(Matrix lambdaMatrix) {
        if (is_2_matrixLearning()) {
            VisualizationSingleton.getLastVisualizalizer().updateLambdaMatrix(lambdaMatrix);
        }
    }
}
