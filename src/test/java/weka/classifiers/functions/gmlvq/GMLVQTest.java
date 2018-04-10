package weka.classifiers.functions.gmlvq;

import org.junit.Before;
import org.junit.Test;
import weka.classifiers.functions.GMLVQ;
import weka.classifiers.functions.gmlvq.core.cost.CostFunctionValue;
import weka.classifiers.functions.gmlvq.model.DataPoint;
import weka.classifiers.functions.gmlvq.model.WekaModelConverter;
import weka.core.Instances;

import java.util.List;

import static org.junit.Assert.fail;

public class GMLVQTest {

    private static final String dataset = TestUtils.Datasets.TECATOR_D;
    private Instances instances;
    private List<DataPoint> dataPoints;
    private GMLVQ gmlvq;

    @Before
    public void setup() {
        try {

            // load data set and convert to internal data structure
            this.instances = TestUtils.loadDataset(dataset);
            this.dataPoints = WekaModelConverter.createDataPoints(this.instances);

            // create GMLVQ instance with requested parameters
            gmlvq = new GMLVQ();
            gmlvq.set_2_matrixLearning(true);
            gmlvq.set_1_visualization(true);
            gmlvq.set_2_dataPointRatioPerRound(0.1);
            gmlvq.setCostFunctionToOptimize(CostFunctionValue.DEFAULT_COST);
            gmlvq.addAdditionalCostFunction(CostFunctionValue.PRECISION_RECALL);
            gmlvq.addAdditionalCostFunction(CostFunctionValue.FMEASURE);

        } catch (Exception e) {
            e.printStackTrace();
            fail("could not set up test, as:\n" + e.getMessage());
        }

    }

    @Test
    public void shouldRunClassifier() {
        try {
            gmlvq.buildClassifier(instances);
            Thread.sleep(1000*60);
        } catch (Exception e) {
            fail("could not run GMLVQ, as:\n" + e.getMessage());
        }
    }
}
