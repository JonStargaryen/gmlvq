package weka.classifiers.functions.gmlvq;

import java.io.InputStream;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;

public final class TestUtils {

    interface Datasets {

        String COFFEE = "coffee.arff";
        String HDS = "hds.arff";
        String IRIS = "iris.arff";
        String IRIS_2_CLASSES = "iris_2classes.arff";
        String TECATOR_D = "tecatorD.arff";
    }

    private TestUtils() {

    }

    public static Instances loadDataset(String filename, boolean normalize) throws Exception {
        InputStream stream = Thread.currentThread().getContextClassLoader().getResourceAsStream(filename);
        Instances instances = DataSource.read(stream);
        // System.out.println("loaded " + instances.size() + " instances");
        // class index is supposed to be the last attribute entry
        instances.setClassIndex(instances.numAttributes() - 1);

        // normalize externally
        if (normalize) {
            Normalize normalizeFilter = new Normalize();
            normalizeFilter.setInputFormat(instances);
            instances = Filter.useFilter(instances, normalizeFilter);
        }

        return instances;
    }

    public static Instances loadDataset(String filename) throws Exception {
        return loadDataset(filename, false);
    }
}
