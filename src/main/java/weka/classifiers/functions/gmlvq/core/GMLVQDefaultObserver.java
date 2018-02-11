package weka.classifiers.functions.gmlvq.core;

import weka.classifiers.functions.gmlvq.core.cost.CostFunctionValue;
import weka.classifiers.functions.gmlvq.model.DataPoint;
import weka.classifiers.functions.gmlvq.model.Observer;
import weka.classifiers.functions.gmlvq.model.Prototype;
import weka.classifiers.functions.gmlvq.model.WekaModelConverter;
import weka.classifiers.functions.gmlvq.visualization.VisualizationSingleton;
import weka.classifiers.functions.gmlvq.visualization.Visualizer;
import weka.core.Instances;
import weka.core.matrix.Matrix;

import javax.swing.*;
import java.lang.reflect.InvocationTargetException;
import java.util.List;
import java.util.Map;

/**
 * Provides a light-weight, neutral, non-Weka implementation of the
 * {@link Observer} interface, so a {@link Visualizer} can be available even
 * when {@link GMLVQCore} was invoked from <i>Tests</i> or operated upon by
 * directly using its API.<br />
 *
 * @author S
 */
public class GMLVQDefaultObserver implements Observer {

    public GMLVQDefaultObserver(GMLVQCore gmlvqCore, Instances trainingData, int numberOfPrototypes,
                                Map<CostFunctionValue, Double> currentCostValues) {

        final List<DataPoint> convertedTrainingData = WekaModelConverter.createDataPoints(trainingData);
        final Map<Double, String> classNamesForDouble = WekaModelConverter.extractClassLables(trainingData);
        final String[] attributeNames = WekaModelConverter.extractAttributeNames(trainingData);
        final int finalNumberOfPrototypes = numberOfPrototypes;
        final Map<CostFunctionValue, Double> finalCurrentCostValues = currentCostValues;

        try {
            SwingUtilities.invokeAndWait(new Runnable() {
                @Override
                public void run() {
                    VisualizationSingleton.addVisualization(new Visualizer(gmlvqCore, convertedTrainingData,
                            classNamesForDouble,
                            attributeNames,
                            finalNumberOfPrototypes,
                            finalCurrentCostValues));
                    VisualizationSingleton.showVisualizations();
                }
            });
        } catch (InterruptedException | InvocationTargetException e) {
            GMLVQCore.LOGGER.warning("failed to initialize visualizer " + e.getMessage());
        }
    }

    @Override
    public void updateCostFunctions(Map<CostFunctionValue, Double> currentCostValues) {
        VisualizationSingleton.getLastVisualizalizer().updateCostFunctions(currentCostValues);
    }

    @Override
    public void updateLambdaMatrix(Matrix lambdaMatrix) {
        VisualizationSingleton.getLastVisualizalizer().updateLambdaMatrix(lambdaMatrix);
    }

    @Override
    public void updatePrototypes(List<Prototype> prototypes) {
        VisualizationSingleton.getLastVisualizalizer().updatePrototypes(prototypes);
    }

}
