package weka.classifiers.functions.gmlvq;

import org.junit.Before;
import org.junit.Test;

import weka.classifiers.functions.GMLVQ;
import weka.core.Instances;

public class GMLVQDrawStuffTest {

    // private List<DataPoint> dataPoints;
    private Instances instances;
    private GMLVQ glmvq;

    @Before
    public void setup() throws Exception {
        this.instances = TestUtils.loadDataset("polyhedral_pmid_27296169.arff");
        // this.dataPoints = WekaModelConverter.createDataPoints(instances);
        this.glmvq = new GMLVQ();
        this.glmvq.setDataPointRatioPerRound(1.0);
        this.glmvq.setNumberOfPrototypesPerClass(1);
        // this.glmvq.setOmegaDimension(5);
        this.glmvq.setVisualization(true);
    }

    @Test
    public void shouldRunClassifier() throws Exception {
        long startTime = System.currentTimeMillis();
        this.glmvq.buildClassifier(this.instances);
        long endTime = System.currentTimeMillis();
        System.out.println("computation took " + ((double) endTime - startTime) / 1000 + " s");
    }
}