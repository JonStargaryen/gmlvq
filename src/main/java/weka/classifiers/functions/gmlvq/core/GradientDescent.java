package weka.classifiers.functions.gmlvq.core;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

import weka.classifiers.functions.gmlvq.core.cost.CostFunctionCalculator;
import weka.classifiers.functions.gmlvq.model.DataPoint;
import weka.classifiers.functions.gmlvq.model.OmegaMatrix;
import weka.classifiers.functions.gmlvq.model.Prototype;
import weka.classifiers.functions.gmlvq.utilities.DataRandomizer;

/**
 * This class wraps the stochastic gradient descent of GMLVQ. Actually, it is
 * not performing any computations, but rather delegates and abstracts the
 * underlying methods to be performed in multiple threads if wanted.
 *
 * @author S
 *
 */
public class GradientDescent implements Serializable, Disposable {

    private static final long serialVersionUID = 1L;

    /**
     * the number of processors that can be used for parallel calculation <br/>
     *
     * <b>NOTE<b/>: sometimes it may be useful to divide this by 2 if
     * hyper-threading "simulates" doubled amount of processors
     */
    private static final int processsors = Runtime.getRuntime().availableProcessors();

    private DataRandomizer dataRandomizer;
    private SigmoidFunction sigmoidFunction;

    private transient ExecutorService executorService;

    private CostFunctionCalculator costFunctionCalculator;

    public GradientDescent(DataRandomizer dataRandomizer, SigmoidFunction sigmoidFunction,
            CostFunctionCalculator costFunctionCalculator) {
        this.dataRandomizer = dataRandomizer;
        this.sigmoidFunction = sigmoidFunction;
        this.costFunctionCalculator = costFunctionCalculator;

        this.executorService = Executors.newFixedThreadPool(processsors);
    }

    /**
     * performs the stochastic gradient descent on the given data points and
     * will result in a proposed update which will either be rejected or
     * accepted and subsequently used in the next epoch
     *
     * @param trainingData
     *            all training data, however commonly only a subpopulation will
     *            actually be used in this batch learning step
     * @param prototypes
     *            the current prototypes
     * @param omegaMatrix
     *            the current mapping rule
     * @param alphaW
     *            the prototype learning rate to employ
     * @param alphaO
     *            the omega learning rate to employ
     * @return the composed update with updated prototypes and updated omega
     *         matrix
     * @throws InterruptedException
     * @throws ExecutionException
     */
    public ProposedUpdate performStochasticGradientDescent(List<DataPoint> trainingData, List<Prototype> prototypes,
            OmegaMatrix omegaMatrix, double alphaW, double alphaO) throws InterruptedException, ExecutionException {

        List<DataPoint> chosenDataPoints = this.dataRandomizer.generateRandomizedSubListOf(trainingData);

        // parallel job creation
        Set<Future<?>> results = new HashSet<Future<?>>();
        List<ProposedUpdate> proposedUpdates = new ArrayList<ProposedUpdate>();
        // split data into partitions so no thread is bored
        for (List<DataPoint> partion : DataRandomizer.partition(chosenDataPoints, processsors)) {

            // init object to accumulate potential changes over the course of
            // the batch
            ProposedUpdate proposedUpdate = new ProposedUpdate(prototypes, this.sigmoidFunction, omegaMatrix, alphaW,
                    alphaO, this.costFunctionCalculator);

            proposedUpdates.add(proposedUpdate);

            results.add(this.executorService.submit(new UpdateCalculator(partion, proposedUpdate)));
        }

        // we have to wait for the results
        for (Future<?> result : results) {
            result.get();
        }

        // here we sum up single results
        return new ProposedUpdate(prototypes, this.sigmoidFunction, omegaMatrix, alphaW, alphaO, proposedUpdates,
                this.costFunctionCalculator);
    }

    private class UpdateCalculator implements Runnable {

        private List<DataPoint> dataPoints;
        private ProposedUpdate proposedUpdate;

        public UpdateCalculator(List<DataPoint> dataPoints, ProposedUpdate proposedUpdate) {
            this.dataPoints = dataPoints;
            this.proposedUpdate = proposedUpdate;
        }

        @Override
        public void run() {
            for (DataPoint dataPoint : this.dataPoints) {
                // accumulate updates of prototypes and matrix
                this.proposedUpdate.incorporate(dataPoint);
            }
        }
    }

    @Override
    public void dispose() throws InterruptedException {

        this.executorService.shutdown();
        this.executorService.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
    }
}
