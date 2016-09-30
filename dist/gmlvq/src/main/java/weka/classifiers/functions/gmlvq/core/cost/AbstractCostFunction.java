package weka.classifiers.functions.gmlvq.core.cost;

import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

import weka.classifiers.functions.gmlvq.core.GradientDescent;
import weka.classifiers.functions.gmlvq.core.SigmoidFunction;
import weka.classifiers.functions.gmlvq.model.DataPoint;
import weka.classifiers.functions.gmlvq.model.OmegaMatrix;
import weka.classifiers.functions.gmlvq.model.Prototype;
import weka.classifiers.functions.gmlvq.model.WinningInformation;
import weka.classifiers.functions.gmlvq.utilities.DataRandomizer;

/**
 * An abstract implementation of the {@link CostFunction} interface. Much like
 * the {@link GradientDescent} class, this one provides the possibility of
 * performing operations in multiple threads and merge the results.
 *
 * @author S
 *
 */
public abstract class AbstractCostFunction implements CostFunction {

    private static final long serialVersionUID = 1L;

    /**
     * the number of processors that can be used for parallel calculation <br/>
     *
     * <b>NOTE<b/>: sometimes it may be useful to divide this by 2 if
     * hyper-threading "simulates" doubled amount of processors
     */
    private static final int processsors = Runtime.getRuntime().availableProcessors();

    private transient ExecutorService executorService;
    protected SigmoidFunction sigmoidFunction;

    public AbstractCostFunction(SigmoidFunction sigmoidFunction) {

        this.executorService = Executors.newFixedThreadPool(processsors);
        this.sigmoidFunction = sigmoidFunction;
    }

    /**
     *
     * @param dataPoints
     * @param prototypes
     * @param omegaMatrix
     * @return
     * @throws ExecutionException
     * @throws InterruptedException
     */
    @Override
    public double evaluate(List<DataPoint> dataPoints, List<Prototype> prototypes, OmegaMatrix omegaMatrix)
            throws InterruptedException, ExecutionException {

        // parallel job creation
        Set<Future<Double>> results = new HashSet<Future<Double>>();
        for (List<DataPoint> partion : DataRandomizer.partition(dataPoints, processsors)) {

            results.add(this.executorService.submit(new CostCalculator(partion, prototypes, omegaMatrix)));
        }

        double error = 0;
        // we have to wait for the results
        for (Future<Double> result : results) {
            error += result.get();
        }

        return error / dataPoints.size();
    }

    /**
     * internally compute
     *
     * @param winningInformation
     * @return
     */
    protected abstract double evaluateWinningInformation(WinningInformation winningInformation);

    private class CostCalculator implements Callable<Double> {

        private List<DataPoint> dataPoints;
        private List<Prototype> prototypes;
        private OmegaMatrix omegaMatrix;

        public CostCalculator(List<DataPoint> dataPoints, List<Prototype> prototypes, OmegaMatrix omegaMatrix) {

            this.dataPoints = dataPoints;
            this.prototypes = prototypes;
            this.omegaMatrix = omegaMatrix;
        }

        @Override
        public Double call() {

            double error = 0;
            for (DataPoint dataPoint : this.dataPoints) {
                WinningInformation winningInformation = dataPoint.getEmbeddedSpaceVector(this.omegaMatrix)
                        .getWinningInformation(this.prototypes);
                error += evaluateWinningInformation(winningInformation);
            }
            return error;
        }
    }

    @Override
    public void dispose() throws InterruptedException {

        this.executorService.shutdown();
        this.executorService.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
    }

}
