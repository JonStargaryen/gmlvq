package weka.classifiers.functions.gmlvq;

import static org.junit.Assert.assertArrayEquals;

import java.util.ArrayList;

import org.junit.Before;
import org.junit.Test;

import weka.classifiers.functions.gmlvq.model.DataPoint;
import weka.classifiers.functions.gmlvq.utilities.LinearAlgebraicCalculations;

public class LinearAlgebraicCalculationsTest {

    private ArrayList<DataPoint> dataPoints;

    private static final double[][] covarianceOctave = new double[][] { { 2.5000e-02, 7.5000e-03, 1.7500e-03 },
            { 7.5000e-03, 7.0000e-03, 1.3500e-03 }, { 1.7500e-03, 1.3500e-03, 4.3000e-04 } };

    @Before
    public void setup() {

        this.dataPoints = new ArrayList<DataPoint>();
        this.dataPoints.add(new DataPoint(new double[] { 4.00000, 2.00000, 0.60000 }, 0.0));
        this.dataPoints.add(new DataPoint(new double[] { 4.20000, 2.10000, 0.59000 }, 0.0));
        this.dataPoints.add(new DataPoint(new double[] { 3.90000, 2.00000, 0.58000 }, 0.0));
        this.dataPoints.add(new DataPoint(new double[] { 4.30000, 2.10000, 0.62000 }, 0.0));
        this.dataPoints.add(new DataPoint(new double[] { 4.10000, 2.20000, 0.63000 }, 0.0));

    }

    @Test
    public void calculateCovariance() {

        double[][] covariance = LinearAlgebraicCalculations.calculateCovarianceFromMeanVector(this.dataPoints)
                .getArray();

        for (int i = 0; i < covarianceOctave.length; i++) {
            assertArrayEquals(covarianceOctave[i], covariance[i], 1.0E-9);
        }
    }
}
