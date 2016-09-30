package weka.classifiers.functions.gmlvq.visualization;

import java.awt.BorderLayout;
import java.awt.Component;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Map.Entry;
import java.util.SortedSet;
import java.util.TreeSet;

import javax.swing.JComponent;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JTable;
import javax.swing.JTextPane;
import javax.swing.table.DefaultTableModel;
import javax.swing.table.TableCellRenderer;

import weka.core.matrix.Matrix;

public class FeatureImpactPanel extends JPanel {

    private static final long serialVersionUID = 1L;

    private String[] attributeNames;
    private Matrix lambdaMatrix;
    private SortedSet<Entry<String, Double>> featureImportance;
    DefaultTableModel tableModel;

    ColorScale multiColorScale;

    public FeatureImpactPanel(String[] attributeNames) {
        this.attributeNames = attributeNames;
        initializeInterface();
    }

    private void initializeInterface() {
        this.setLayout(new BorderLayout());

        JTextPane descriptionPane = new JTextPane();
        descriptionPane.setText("This table contains the 15 features from the main diagonal with the largest values."
                + "Therefore, the first entry in the table is the feature that influences the classification and"
                + " in turn the separation of the classes the most. The next entry has the second largest"
                + " influence, and so on.");
        descriptionPane.setEditable(false);
        descriptionPane.setBackground(this.getBackground());
        // TODO maybe make this fancy'er
        // descriptionPane.setContentType("text/html");
        this.add(descriptionPane, BorderLayout.NORTH);

        this.tableModel = new DefaultTableModel();
        this.tableModel.setDataVector(new String[][] { { "Pre", "pare", "ing" } },
                new String[] { "color", "feature", "value" });
        JTable featureTable = new JTable(this.tableModel) {

            private static final long serialVersionUID = 1L;

            @Override
            public Component prepareRenderer(TableCellRenderer renderer, int rowIndex, int columnIndex) {
                JComponent component = (JComponent) super.prepareRenderer(renderer, rowIndex, columnIndex);
                if (columnIndex == 0) {
                    component.setBackground(FeatureImpactPanel.this.multiColorScale
                            .getColor(Float.valueOf(getValueAt(rowIndex, 2).toString())));
                } else {
                    component.setBackground(FeatureImpactPanel.this.getBackground());
                }
                return component;
            }
        };

        this.add(new JScrollPane(featureTable), BorderLayout.CENTER);
    }

    static <K, V extends Comparable<? super V>> SortedSet<Map.Entry<K, V>> entriesSortedByValues(Map<K, V> map) {
        SortedSet<Map.Entry<K, V>> sortedEntries = new TreeSet<Map.Entry<K, V>>(new HashMapValueComparator<K, V>());
        sortedEntries.addAll(map.entrySet());
        return sortedEntries;
    }

    public Matrix getLambdaMatrix() {
        return this.lambdaMatrix;
    }

    public void setLambdaMatrix(Matrix lambdaMatrix) {
        this.lambdaMatrix = lambdaMatrix;
        this.extractAndSortFeatures();
        this.multiColorScale = new ColorScale.Builder(this.featureImportance.last().getValue().floatValue(),
                this.featureImportance.first().getValue().floatValue()).build();
        String[] cols = { "color", "feature", "value" };
        Iterator<Entry<String, Double>> it = this.featureImportance.iterator();
        int maxFeatureIndex = 15;
        int currentFeatureIndex = 0;

        String[][] data = new String[maxFeatureIndex][3];

        while (it.hasNext() && currentFeatureIndex < maxFeatureIndex) {
            Entry<String, Double> current = it.next();
            data[currentFeatureIndex][0] = "   ";
            data[currentFeatureIndex][1] = current.getKey();
            data[currentFeatureIndex][2] = String.valueOf(current.getValue());
            currentFeatureIndex++;
        }

        this.tableModel.setDataVector(data, cols);
    }

    private void extractAndSortFeatures() {
        Map<String, Double> featureValues = new HashMap<String, Double>();
        for (int diagonalIndex = 0; diagonalIndex < this.lambdaMatrix.getRowDimension(); diagonalIndex++) {
            featureValues.put(this.attributeNames[diagonalIndex], this.lambdaMatrix.get(diagonalIndex, diagonalIndex));
        }
        this.featureImportance = entriesSortedByValues(featureValues);
    }

}
