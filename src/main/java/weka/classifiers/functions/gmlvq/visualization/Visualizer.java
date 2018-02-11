package weka.classifiers.functions.gmlvq.visualization;

import java.awt.BorderLayout;
import java.io.File;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.List;
import java.util.Map;

import javax.swing.JButton;
import javax.swing.JCheckBox;
import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JTabbedPane;
import javax.swing.JToolBar;
import javax.swing.filechooser.FileNameExtensionFilter;

import weka.classifiers.functions.gmlvq.core.GMLVQCore;
import weka.classifiers.functions.gmlvq.core.cost.CostFunctionValue;
import weka.classifiers.functions.gmlvq.model.DataPoint;
import weka.classifiers.functions.gmlvq.model.Prototype;
import weka.classifiers.functions.gmlvq.utilities.LinearAlgebraicCalculations;
import weka.core.matrix.Matrix;

public class Visualizer extends JTabbedPane {

    private static final long serialVersionUID = 1L;

    private VisualizerMouseAdapter mouseAdapter;
    private RunDetailsPanel runDetailsPanel;
    private LambdaMatrixPanel panelLambdaMatrix;
    private CostFunctionChartPanel panelCostFunctionChart;
    private FeatureImpactPanel panelFeatureInfluence;
    private FeatureAnalysisPanel panelFeatureAnalysis;

    private ColorScale colorScale;

    // gui components
    private JCheckBox checkBoxShowScale;
    private JButton buttonExport;

    public Visualizer(GMLVQCore gmlvqCore, List<DataPoint> dataPoints, Map<Double, String> classNamesForDouble, String[] attributeNames,
                      int numberOfPrototypes, Map<CostFunctionValue, Double> currentCostValues) {

        this.runDetailsPanel = new RunDetailsPanel(gmlvqCore);
        this.panelLambdaMatrix = new LambdaMatrixPanel(attributeNames, this.colorScale);
        this.panelFeatureInfluence = new FeatureImpactPanel(attributeNames, this.colorScale);
        this.mouseAdapter = new VisualizerMouseAdapter(this);
        this.panelCostFunctionChart = new CostFunctionChartPanel(currentCostValues);

        this.panelFeatureAnalysis = new FeatureAnalysisPanel(this.mouseAdapter, dataPoints, classNamesForDouble, attributeNames, numberOfPrototypes);
        initializeInterface();
    }

    private void initializeInterface() {

        JPanel panelLambdaMatrix = new JPanel();
        addTab("Lambda matrix", panelLambdaMatrix);
        panelLambdaMatrix.setLayout(new BorderLayout(0, 0));
        panelLambdaMatrix.add(this.panelLambdaMatrix);

        JToolBar toolBarMatrix = new JToolBar();
        toolBarMatrix.setFloatable(false);
        panelLambdaMatrix.add(toolBarMatrix, BorderLayout.NORTH);

        this.checkBoxShowScale = new JCheckBox("Show scale");
        this.checkBoxShowScale.setName("SHOW_SCALE");
        this.checkBoxShowScale.setSelected(true);
        this.checkBoxShowScale.addMouseListener(this.mouseAdapter);
        toolBarMatrix.add(this.checkBoxShowScale);

        this.buttonExport = new JButton("Export");
        this.buttonExport.setName("EXPORT_LAMBDA_MATRIX");
        this.buttonExport.addMouseListener(this.mouseAdapter);
        toolBarMatrix.add(this.buttonExport);

        addTab("Cost funktions", this.panelCostFunctionChart);
        addTab("Feature Influence", this.panelFeatureInfluence);
        addTab("Feature Analysis", this.panelFeatureAnalysis);
        addTab("Run details", this.runDetailsPanel);
        setSelectedIndex(0);

    }

    public void switchScale() {
        this.panelLambdaMatrix.getRenderer().setShowScale(this.checkBoxShowScale.isSelected());
    }

    public void saveLamdaMatrixToSVG() {

        JFileChooser fileChooser = new JFileChooser();
        fileChooser.setFileFilter(new FileNameExtensionFilter("scalable vector graphics", "svg"));
        fileChooser.setSelectedFile(new File("lambda_matrix.svg"));

        if (fileChooser.showSaveDialog(null) == JFileChooser.APPROVE_OPTION) {
            File file = fileChooser.getSelectedFile();
            if (file != null && file.exists()) {
                int response = JOptionPane.showConfirmDialog(null,
                        "The file already exists. Do you want to replace the existing file?", "Overwrite file",
                        JOptionPane.YES_NO_OPTION, JOptionPane.WARNING_MESSAGE);
                if (response != JOptionPane.YES_OPTION) {
                    return;
                }
            }
            // request export
            this.panelLambdaMatrix.exportLambdaMatrixToSVG(file);
        }
    }

    public FeatureAnalysisPanel getFeatureAnalysisPanel() {
        return this.panelFeatureAnalysis;
    }

    public void updatePrototypes(List<Prototype> prototypes) {
        this.panelFeatureAnalysis.setPrototypes(prototypes);

    }

    public void updateCostFunctions(Map<CostFunctionValue, Double> currentCostValues) {
        this.panelCostFunctionChart.addLeatestValues(currentCostValues);
    }

    public void updateLambdaMatrix(Matrix lambdaMatrix) {
        double[] minAndMaxValues = LinearAlgebraicCalculations.getMinAndMaxValuesFromMatrix(lambdaMatrix.copy());
        float minValue = (float) minAndMaxValues[LinearAlgebraicCalculations.MINIMAL_INDEX];
        float maxValue = (float) minAndMaxValues[LinearAlgebraicCalculations.MAXIMAL_INDEX];
        this.colorScale = new ColorScale.Builder(minValue, maxValue).build();
        this.panelLambdaMatrix.setLambdaMatrix(lambdaMatrix);
        this.panelLambdaMatrix.repaint();
        this.panelFeatureInfluence.setLambdaMatrix(lambdaMatrix);
        this.panelFeatureInfluence.repaint();
    }

}
