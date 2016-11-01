package weka.classifiers.functions.gmlvq;

import org.junit.Before;
import org.junit.Test;
import weka.classifiers.functions.gmlvq.core.GMLVQCore;
import weka.classifiers.functions.gmlvq.core.cost.CostFunctionValue;
import weka.classifiers.functions.gmlvq.model.DataPoint;
import weka.classifiers.functions.gmlvq.model.WekaModelConverter;
import weka.core.Instances;

import java.util.List;

public class GMLVQVisualizerTest {

    // private List<DataPoint> dataPoints;
    private Instances instances;
    //    private GMLVQ glmvq;
    private List<DataPoint> dataPoints;
    private GMLVQCore gmlvq;

    @Before
    public void setup() throws Exception {
        instances = TestUtils.loadDataset("hds.arff");
        dataPoints = WekaModelConverter.createDataPoints(instances);
        gmlvq = new GMLVQCore.Builder().numberOfEpochs(1000)
                                       .numberOfPrototypesPerClass(5)
                                       .dataPointRatioPerRound(0.8)
                                       .costFunctionToOptimize(CostFunctionValue.CLASSIFICATION_ERROR)
                                       .buildAndShow(dataPoints, instances);
    }

    @Test
    public void shouldRunClassifier() throws Exception {
        long startTime = System.currentTimeMillis();
        this.gmlvq.buildClassifier();
        long endTime = System.currentTimeMillis();
        System.out.println("computation took " + ((double) endTime - startTime) / 1000 + " s");
        // infinite sleep
        Thread.sleep(Long.MAX_VALUE);
    }
}