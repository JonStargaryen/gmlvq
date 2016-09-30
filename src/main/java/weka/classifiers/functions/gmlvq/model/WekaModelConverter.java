package weka.classifiers.functions.gmlvq.model;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Converts the internal data structure to WEKA format and vice versa.
 *
 * @author fkaiser
 *
 */
public final class WekaModelConverter {

    private WekaModelConverter() {

    }

    public static DataPoint createDataPoint(Instance instance) {
        return new DataPoint(Arrays.copyOf(instance.toDoubleArray(), instance.numAttributes() - 1), instance.classValue());
    }

    public static List<DataPoint> createDataPoints(Instances dataset) {
        List<DataPoint> modelDataset = new ArrayList<DataPoint>();
        for (Instance instance : dataset) {
            modelDataset.add(createDataPoint(instance));
        }
        return modelDataset;
    }

    /**
     * extracts all attribute names of WEKA instances and converts them to a
     * string array
     *
     * @param data
     * @return attribute names as string arrays
     */
    public static String[] extractAttributeNames(Instances data) {
        String[] attributeNames = new String[data.numAttributes() - 1];
        int i = 0;
        Enumeration<Attribute> attributes = data.enumerateAttributes();
        while (attributes.hasMoreElements() && i < data.numAttributes() - 1) {
            Attribute attribute = attributes.nextElement();
            attributeNames[i] = attribute.name();
            i++;
        }
        return attributeNames;
    }

    public static Map<Double, String> extractClassLables(Instances data) {
        Map<Double, String> distinctClasses = new HashMap<Double, String>();
        for (Instance instance : data) {
            double key = instance.value(data.classAttribute());
            String value = instance.classAttribute().value((int) key);
            distinctClasses.put(key, value);
        }
        return distinctClasses;
    }
}
