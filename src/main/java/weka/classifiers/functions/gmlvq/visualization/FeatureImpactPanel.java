package weka.classifiers.functions.gmlvq.visualization;

import weka.classifiers.functions.gmlvq.utilities.LinearAlgebraicCalculations;
import weka.core.matrix.Matrix;

import javax.swing.*;
import javax.swing.table.DefaultTableModel;
import javax.swing.table.TableCellRenderer;
import java.awt.*;
import java.util.*;
import java.util.Map.Entry;

public class FeatureImpactPanel extends JPanel {

    private static final long serialVersionUID = 1L;

    private String[] attributeNames;
    private Matrix lambdaMatrix;
    private SortedSet<Entry<String, Double>> featureImportance;
    private DefaultTableModel tableModel;

    private ColorScale colorScale;

    public FeatureImpactPanel(String[] attributeNames, ColorScale colorScale) {
        this.attributeNames = attributeNames;
        this.colorScale = colorScale;
        initializeInterface(colorScale);
    }

    private void initializeInterface(ColorScale colorScale) {
        this.setLayout(new BorderLayout());
        this.colorScale = colorScale;
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
                    component.setBackground(FeatureImpactPanel.this.colorScale
                            .getColor(Float.valueOf(getValueAt(rowIndex, 2).toString())));
                } else {
                    component.setBackground(FeatureImpactPanel.this.getBackground());
                }
                return component;
            }

            /**
             * Prohibit editing of cells by the user.
             */
            @Override
            public boolean isCellEditable(int row, int column) {
                return false;
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

        double[] minAndMaxValues = LinearAlgebraicCalculations.getMinAndMaxValuesFromMatrix(lambdaMatrix.copy());
        float minValue = (float) minAndMaxValues[LinearAlgebraicCalculations.MINIMAL_INDEX];
        float maxValue = (float) minAndMaxValues[LinearAlgebraicCalculations.MAXIMAL_INDEX];
        this.colorScale = new ColorScale.Builder(minValue, maxValue).build();

        String[] cols = { "color", "feature", "value" };
        Iterator<Entry<String, Double>> it = this.featureImportance.iterator();
        int maxFeatureIndex =  attributeNames.length < 15 ? attributeNames.length : 15;
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
