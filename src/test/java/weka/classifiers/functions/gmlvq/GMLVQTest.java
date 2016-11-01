package weka.classifiers.functions.gmlvq;

import static org.junit.Assert.fail;

import java.util.List;

import org.junit.Before;
import org.junit.Test;

import weka.classifiers.functions.gmlvq.core.GMLVQCore;
import weka.classifiers.functions.gmlvq.core.cost.CostFunctionValue;
import weka.classifiers.functions.gmlvq.model.DataPoint;
import weka.classifiers.functions.gmlvq.model.WekaModelConverter;
import weka.core.Instances;

public class GMLVQTest {

    private static final String dataset = TestUtils.Datasets.TECATOR_D;
    private Instances instances;
    private List<DataPoint> dataPoints;
    private GMLVQCore gmlvq;

    @Before
    public void setup() {
        try {

            // load data set and convert to internal data structure
            this.instances = TestUtils.loadDataset(dataset);
            this.dataPoints = WekaModelConverter.createDataPoints(this.instances);

            // create GMLVQ instance with requested parameters
            this.gmlvq = new GMLVQCore.Builder()/* .matrixLearning(false) */
                    .costFunctionToOptimize(CostFunctionValue.WEIGHTED_ACCURACY).numberOfEpochs(2500)
                    .dataPointRatioPerRound(1.0).build(this.dataPoints);
        } catch (Exception e) {
            e.printStackTrace();
            fail("could not set up test, as:\n" + e.getMessage());
        }
    }

    @Test
    public void shouldRunClassifier() {
        try {
            this.gmlvq.buildClassifier();
        } catch (Exception e) {
            fail("could not run GMLVQ, as:\n" + e.getMessage());
        }
    }
}
