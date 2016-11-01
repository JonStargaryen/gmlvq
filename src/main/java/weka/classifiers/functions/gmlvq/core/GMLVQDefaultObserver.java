package weka.classifiers.functions.gmlvq.core;

import weka.classifiers.functions.gmlvq.core.cost.CostFunctionValue;
import weka.classifiers.functions.gmlvq.model.DataPoint;
import weka.classifiers.functions.gmlvq.model.Observer;
import weka.classifiers.functions.gmlvq.model.Prototype;
import weka.classifiers.functions.gmlvq.model.WekaModelConverter;
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

    private Visualizer visualizer;

    public GMLVQDefaultObserver(Instances trainingData, int numberOfPrototypes,
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
                    GMLVQDefaultObserver.this.visualizer = new Visualizer(convertedTrainingData,
                                                                          classNamesForDouble,
                                                                          attributeNames,
                                                                          finalNumberOfPrototypes,
                                                                          finalCurrentCostValues);
                    GMLVQDefaultObserver.this.visualizer.setVisible(true);
                }
            });
        } catch (InterruptedException e) {
            GMLVQCore.LOGGER.warning("failed to initialize visualizer " + e.getMessage());
        } catch (InvocationTargetException e) {
            GMLVQCore.LOGGER.warning("failed to initialize visualizer " + e.getMessage());
        }
    }

    @Override
    public void updateCostFunctions(Map<CostFunctionValue, Double> currentCostValues) {
        this.visualizer.updateCostFunctions(currentCostValues);

    }

    @Override
    public void updateLambdaMatrix(Matrix lambdaMatrix) {
        this.visualizer.updateLambdaMatrix(lambdaMatrix);
    }

    @Override
    public void updatePrototypes(List<Prototype> prototypes) {
        this.visualizer.updatePrototypes(prototypes);
    }

}