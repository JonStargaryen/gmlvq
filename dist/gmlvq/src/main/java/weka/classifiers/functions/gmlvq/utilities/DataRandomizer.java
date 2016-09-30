package weka.classifiers.functions.gmlvq.utilities;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import weka.classifiers.functions.gmlvq.core.GMLVQCore;

/**
 * A collection of convenience methods to create subsets of data points or to
 * split them into partition of equal size in order to provide an arbitrary
 * number of threads with data to process.
 *
 * @author S
 *
 */
public class DataRandomizer implements Serializable {

    private static final long serialVersionUID = 1L;

    private final long seed;
    private final Random random;

    private final int numberOfTrainingData;

    public int getNumberOfTrainingData() {
        return this.numberOfTrainingData;
    }

    private int fractionOfTrainingData;
    private double ratio;

    public double getRatio() {
        return this.ratio;
    }

    public DataRandomizer(int numberOfTrainingData, double ratio) {
        this(numberOfTrainingData, ratio, 0);
    }

    public DataRandomizer(int numberOfTrainingData, double ratio, long seed) {
        this.numberOfTrainingData = numberOfTrainingData;
        this.ratio = ratio;
        this.fractionOfTrainingData = (int) (numberOfTrainingData * ratio);
        if (this.fractionOfTrainingData == 0) {
            GMLVQCore.LOGGER
                    .warning("data ratio too small for number of training data, ensured that 1 data point is selected");
            this.fractionOfTrainingData = 1;
        }
        this.seed = seed;
        if (seed == 0) {
            this.random = new Random();
        } else {
            this.random = new Random(this.seed);
        }
    }

    /**
     * returns the specified number of elements at random from the given list
     *
     * @param originalList
     *            what list to select from?
     * @param lengthOfSublist
     *            how many elements to select?
     * @return a random selection of elements
     */
    public <T> List<T> generateRandomizedSubListOf(List<T> originalList, int lengthOfSublist) {
        // no need to shuffle when we choose all data points - also easier to
        // debug
        if (this.ratio != 1.0) {
            Collections.shuffle(originalList, this.random);
        }
        return originalList.subList(0, lengthOfSublist);
    }

    /**
     * @see DataRandomizer#generateRandomizedSubListOf(List, int)
     */
    public <T> List<T> generateRandomizedSubListOf(List<T> originalList) {
        return generateRandomizedSubListOf(originalList, this.fractionOfTrainingData);
    }

    public long getSeed() {
        return this.seed;
    }

    public Random getRandom() {
        return this.random;
    }

    /**
     * distributes a set of object evenly to subset of equal (+-1) size
     *
     * @param list
     * @param numberOfPartitions
     *            how many partitions to create?
     * @return the sublists contained in a list
     */
    public static <E> List<List<E>> partition(List<E> list, int numberOfPartitions) {
        List<List<E>> partitions = new ArrayList<List<E>>();
        // init partitions
        for (int i = 0; i < numberOfPartitions; i++) {
            partitions.add(new ArrayList<E>(list.size() / numberOfPartitions + 1));
        }

        // distribute objects fairly among all partitions
        for (int i = 0; i < list.size(); i++) {
            partitions.get(i % numberOfPartitions).add(list.get(i));
        }

        return partitions;
    }
}
