package weka.classifiers.functions.gmlvq.visualization;

import java.io.Serializable;
import java.util.Comparator;
import java.util.Map;

public class HashMapValueComparator<K, V extends Comparable<? super V>>
        implements Serializable, Comparator<Map.Entry<K, V>> {

    private static final long serialVersionUID = 1L;

    @Override
    public int compare(Map.Entry<K, V> e1, Map.Entry<K, V> e2) {
        int res = e1.getValue().compareTo(e2.getValue()) * -1;
        return res != 0 ? res : 1;
    }
}
