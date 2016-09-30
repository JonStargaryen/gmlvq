package weka.classifiers.functions.gmlvq.visualization;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.GridLayout;
import java.awt.Insets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import javax.swing.DefaultListModel;
import javax.swing.JButton;
import javax.swing.JLabel;
import javax.swing.JList;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JTree;
import javax.swing.UIManager;
import javax.swing.border.TitledBorder;
import javax.swing.tree.DefaultMutableTreeNode;
import javax.swing.tree.DefaultTreeModel;
import javax.swing.tree.TreeSelectionModel;

import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.CategoryAxis;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.plot.CategoryPlot;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.renderer.category.BarRenderer;
import org.jfree.chart.renderer.category.CategoryItemRenderer;
import org.jfree.chart.renderer.category.LineAndShapeRenderer;
import org.jfree.data.category.DefaultCategoryDataset;

import weka.classifiers.functions.gmlvq.model.DataPoint;
import weka.classifiers.functions.gmlvq.model.Prototype;
import weka.classifiers.functions.gmlvq.utilities.LinearAlgebraicCalculations;

public class FeatureAnalysisPanel extends JPanel {

    private static final long serialVersionUID = 1L;

    private VisualizerMouseAdapter mouseAdapter;
    private ChartPanel chartPanel;
    private JFreeChart chart;
    private DefaultCategoryDataset chartTrainingData;
    private List<DefaultCategoryDataset> chartPrototypeData;

    private JList<String> listHideByClass;
    private DefaultListModel<String> listModelHideByClass;
    private JList<String> listShowByClass;
    private DefaultListModel<String> listModelShowByClass;
    private JList<String> listHideByAttribute;
    private DefaultListModel<String> listModelHideByAttribute;
    private JList<String> listShowByAttribute;
    private DefaultListModel<String> listModelShowByAttribute;
    private JList<String> listShowingPrototypes;
    private DefaultListModel<String> listModelShowingPrototypes;
    private JTree treePrototypes;

    private Map<String, Color> classColors = new HashMap<String, Color>();

    private List<DataPoint> dataPoints;
    private List<Prototype> prototypes;
    private Map<Double, String> classNamesForDouble;
    private String[] attributeNames;
    private int numberOfPrototypes;

    private boolean prototypesInitialized = false;

    public FeatureAnalysisPanel(VisualizerMouseAdapter mouseAdapter, List<DataPoint> datapoints,
            Map<Double, String> classNamesForDouble, String[] attributeNames, int numberOfPrototypes) {
        this.mouseAdapter = mouseAdapter;
        this.dataPoints = datapoints;
        this.classNamesForDouble = classNamesForDouble;
        this.attributeNames = attributeNames;
        this.numberOfPrototypes = numberOfPrototypes;

        this.listModelShowByClass = new DefaultListModel<String>();
        this.listModelHideByClass = new DefaultListModel<String>();
        this.listModelShowByAttribute = new DefaultListModel<String>();
        this.listModelHideByAttribute = new DefaultListModel<String>();
        this.listModelShowingPrototypes = new DefaultListModel<String>();

        initializeTrainingData();
        fillLists();
        initializeChart();
        initializeGUI();

    }

    private void initializeGUI() {

        GridBagLayout gbl_panel = new GridBagLayout();
        gbl_panel.columnWidths = new int[] { 0, 0 };
        gbl_panel.rowHeights = new int[] { 0, 0, 0, 0 };
        gbl_panel.columnWeights = new double[] { 1.0, Double.MIN_VALUE };
        gbl_panel.rowWeights = new double[] { 0.0, 0.0, 1.0, Double.MIN_VALUE };
        this.setLayout(gbl_panel);

        JPanel panelData = new JPanel();
        panelData.setBorder(new TitledBorder(null, "Data", TitledBorder.LEADING, TitledBorder.TOP, null, null));
        GridBagConstraints gbc_panelData = new GridBagConstraints();
        gbc_panelData.insets = new Insets(0, 0, 5, 0);
        gbc_panelData.fill = GridBagConstraints.BOTH;
        gbc_panelData.gridx = 0;
        gbc_panelData.gridy = 0;
        this.add(panelData, gbc_panelData);
        panelData.setLayout(new GridLayout(1, 0, 0, 0));

        JPanel panelDataByClass = new JPanel();
        panelDataByClass
                .setBorder(new TitledBorder(null, "By Class", TitledBorder.LEADING, TitledBorder.TOP, null, null));
        panelData.add(panelDataByClass);
        GridBagLayout gbl_panelDataByClass = new GridBagLayout();
        gbl_panelDataByClass.columnWidths = new int[] { 0, 0, 0, 0 };
        gbl_panelDataByClass.rowHeights = new int[] { 0, 0, 0, 0, 0, 0 };
        gbl_panelDataByClass.columnWeights = new double[] { 1.0, 0.0, 1.0, Double.MIN_VALUE };
        gbl_panelDataByClass.rowWeights = new double[] { 0.0, 0.0, 0.0, 0.0, 1.0, Double.MIN_VALUE };
        panelDataByClass.setLayout(gbl_panelDataByClass);

        JLabel lblClassShowing = new JLabel("Showing");
        GridBagConstraints gbc_lblClassShowing = new GridBagConstraints();
        gbc_lblClassShowing.anchor = GridBagConstraints.NORTH;
        gbc_lblClassShowing.insets = new Insets(0, 0, 5, 5);
        gbc_lblClassShowing.gridx = 0;
        gbc_lblClassShowing.gridy = 0;
        panelDataByClass.add(lblClassShowing, gbc_lblClassShowing);

        JLabel lblClassHiding = new JLabel("Hiding");
        GridBagConstraints gbc_lblClassHiding = new GridBagConstraints();
        gbc_lblClassHiding.anchor = GridBagConstraints.NORTH;
        gbc_lblClassHiding.insets = new Insets(0, 0, 5, 0);
        gbc_lblClassHiding.gridx = 2;
        gbc_lblClassHiding.gridy = 0;
        panelDataByClass.add(lblClassHiding, gbc_lblClassHiding);

        JButton btnClassHideAll = new JButton(">>");
        btnClassHideAll.setName("CLASS_HIDE_ALL");
        btnClassHideAll.addMouseListener(this.mouseAdapter);
        GridBagConstraints gbc_btnClassHideAll = new GridBagConstraints();
        gbc_btnClassHideAll.anchor = GridBagConstraints.NORTHWEST;
        gbc_btnClassHideAll.insets = new Insets(0, 0, 5, 5);
        gbc_btnClassHideAll.gridx = 1;
        gbc_btnClassHideAll.gridy = 1;
        panelDataByClass.add(btnClassHideAll, gbc_btnClassHideAll);

        JButton btnClassHideOne = new JButton(">");
        btnClassHideOne.setName("CLASS_HIDE_ONE");
        btnClassHideOne.addMouseListener(this.mouseAdapter);
        GridBagConstraints gbc_btnClassHideOne = new GridBagConstraints();
        gbc_btnClassHideOne.anchor = GridBagConstraints.NORTH;
        gbc_btnClassHideOne.insets = new Insets(0, 0, 5, 5);
        gbc_btnClassHideOne.gridx = 1;
        gbc_btnClassHideOne.gridy = 2;
        panelDataByClass.add(btnClassHideOne, gbc_btnClassHideOne);

        JButton btnClassShowOne = new JButton("<");
        btnClassShowOne.setName("CLASS_SHOW_ONE");
        btnClassShowOne.addMouseListener(this.mouseAdapter);
        GridBagConstraints gbc_btnClassShowOne = new GridBagConstraints();
        gbc_btnClassShowOne.anchor = GridBagConstraints.NORTH;
        gbc_btnClassShowOne.insets = new Insets(0, 0, 5, 5);
        gbc_btnClassShowOne.gridx = 1;
        gbc_btnClassShowOne.gridy = 3;
        panelDataByClass.add(btnClassShowOne, gbc_btnClassShowOne);

        this.listShowByClass = new JList<String>(this.listModelShowByClass);
        JScrollPane scrollShowByClass = new JScrollPane();
        scrollShowByClass.setViewportView(this.listShowByClass);
        GridBagConstraints gbc_listShowByClass = new GridBagConstraints();
        gbc_listShowByClass.gridheight = 4;
        gbc_listShowByClass.insets = new Insets(0, 0, 0, 5);
        gbc_listShowByClass.fill = GridBagConstraints.BOTH;
        gbc_listShowByClass.gridx = 0;
        gbc_listShowByClass.gridy = 1;
        panelDataByClass.add(scrollShowByClass, gbc_listShowByClass);

        JButton btnClassShowAll = new JButton("<<");
        btnClassShowAll.setName("CLASS_SHOW_ALL");
        btnClassShowAll.addMouseListener(this.mouseAdapter);
        GridBagConstraints gbc_btnClassShowAll = new GridBagConstraints();
        gbc_btnClassShowAll.anchor = GridBagConstraints.NORTH;
        gbc_btnClassShowAll.insets = new Insets(0, 0, 0, 5);
        gbc_btnClassShowAll.gridx = 1;
        gbc_btnClassShowAll.gridy = 4;
        panelDataByClass.add(btnClassShowAll, gbc_btnClassShowAll);

        this.listHideByClass = new JList<String>(this.listModelHideByClass);
        JScrollPane scrollHideByClass = new JScrollPane();
        scrollHideByClass.setViewportView(this.listHideByClass);
        GridBagConstraints gbc_listHideByClass = new GridBagConstraints();
        gbc_listHideByClass.gridheight = 4;
        gbc_listHideByClass.fill = GridBagConstraints.BOTH;
        gbc_listHideByClass.gridx = 2;
        gbc_listHideByClass.gridy = 1;
        panelDataByClass.add(scrollHideByClass, gbc_listHideByClass);

        JPanel panelDataByAttribute = new JPanel();
        panelDataByAttribute.setBorder(new TitledBorder(UIManager.getBorder("TitledBorder.border"), "By Attribute",
                TitledBorder.LEADING, TitledBorder.TOP, null, new Color(0, 0, 0)));
        panelData.add(panelDataByAttribute);
        GridBagLayout gbl_panelDataByAttribute = new GridBagLayout();
        gbl_panelDataByAttribute.columnWidths = new int[] { 0, 0, 0, 0 };
        gbl_panelDataByAttribute.rowHeights = new int[] { 0, 0, 0, 0, 0, 0 };
        gbl_panelDataByAttribute.columnWeights = new double[] { 1.0, 0.0, 1.0, Double.MIN_VALUE };
        gbl_panelDataByAttribute.rowWeights = new double[] { 0.0, 0.0, 0.0, 0.0, 1.0, Double.MIN_VALUE };
        panelDataByAttribute.setLayout(gbl_panelDataByAttribute);

        JLabel lblAttributeShowing = new JLabel("Showing");
        GridBagConstraints gbc_lblAttributeShowing = new GridBagConstraints();
        gbc_lblAttributeShowing.anchor = GridBagConstraints.NORTH;
        gbc_lblAttributeShowing.insets = new Insets(0, 0, 5, 5);
        gbc_lblAttributeShowing.gridx = 0;
        gbc_lblAttributeShowing.gridy = 0;
        panelDataByAttribute.add(lblAttributeShowing, gbc_lblAttributeShowing);

        JLabel lblAttributeHiding = new JLabel("Hiding");
        GridBagConstraints gbc_lblAttributeHiding = new GridBagConstraints();
        gbc_lblAttributeHiding.anchor = GridBagConstraints.NORTH;
        gbc_lblAttributeHiding.insets = new Insets(0, 0, 5, 0);
        gbc_lblAttributeHiding.gridx = 2;
        gbc_lblAttributeHiding.gridy = 0;
        panelDataByAttribute.add(lblAttributeHiding, gbc_lblAttributeHiding);

        this.listShowByAttribute = new JList<String>(this.listModelShowByAttribute);
        JScrollPane scrollShowByAttribute = new JScrollPane();
        scrollShowByAttribute.setViewportView(this.listShowByAttribute);
        GridBagConstraints gbc_listShowByAttribute = new GridBagConstraints();
        gbc_listShowByAttribute.fill = GridBagConstraints.BOTH;
        gbc_listShowByAttribute.gridheight = 4;
        gbc_listShowByAttribute.insets = new Insets(0, 0, 0, 5);
        gbc_listShowByAttribute.gridx = 0;
        gbc_listShowByAttribute.gridy = 1;
        panelDataByAttribute.add(scrollShowByAttribute, gbc_listShowByAttribute);

        JButton btnAttributeHideAll = new JButton(">>");
        btnAttributeHideAll.setName("ATTRIBUTE_HIDE_ALL");
        btnAttributeHideAll.addMouseListener(this.mouseAdapter);
        GridBagConstraints gbc_btnAttributeHideAll = new GridBagConstraints();
        gbc_btnAttributeHideAll.anchor = GridBagConstraints.NORTHWEST;
        gbc_btnAttributeHideAll.insets = new Insets(0, 0, 5, 5);
        gbc_btnAttributeHideAll.gridx = 1;
        gbc_btnAttributeHideAll.gridy = 1;
        panelDataByAttribute.add(btnAttributeHideAll, gbc_btnAttributeHideAll);

        this.listHideByAttribute = new JList<String>(this.listModelHideByAttribute);
        JScrollPane scrollHideByAttribute = new JScrollPane();
        scrollHideByAttribute.setViewportView(this.listHideByAttribute);
        GridBagConstraints gbc_listHideByAttribute = new GridBagConstraints();
        gbc_listHideByAttribute.fill = GridBagConstraints.BOTH;
        gbc_listHideByAttribute.gridheight = 4;
        gbc_listHideByAttribute.gridx = 2;
        gbc_listHideByAttribute.gridy = 1;
        panelDataByAttribute.add(this.listHideByAttribute, gbc_listHideByAttribute);

        JButton btnAttributeHideOne = new JButton(">");
        btnAttributeHideOne.setName("ATTRIBUTE_HIDE_ONE");
        btnAttributeHideOne.addMouseListener(this.mouseAdapter);
        GridBagConstraints gbc_btnAttributeHideOne = new GridBagConstraints();
        gbc_btnAttributeHideOne.anchor = GridBagConstraints.NORTH;
        gbc_btnAttributeHideOne.insets = new Insets(0, 0, 5, 5);
        gbc_btnAttributeHideOne.gridx = 1;
        gbc_btnAttributeHideOne.gridy = 2;
        panelDataByAttribute.add(btnAttributeHideOne, gbc_btnAttributeHideOne);

        JButton btnAttributeShowOne = new JButton("<");
        btnAttributeShowOne.setName("ATTRIBUTE_SHOW_ONE");
        btnAttributeShowOne.addMouseListener(this.mouseAdapter);
        GridBagConstraints gbc_btnAttributeShowOne = new GridBagConstraints();
        gbc_btnAttributeShowOne.anchor = GridBagConstraints.NORTH;
        gbc_btnAttributeShowOne.insets = new Insets(0, 0, 5, 5);
        gbc_btnAttributeShowOne.gridx = 1;
        gbc_btnAttributeShowOne.gridy = 3;
        panelDataByAttribute.add(btnAttributeShowOne, gbc_btnAttributeShowOne);

        JButton btnAttributeShowAll = new JButton("<<");
        btnAttributeShowAll.setName("ATTRIBUTE_SHOW_ALL");
        btnAttributeShowAll.addMouseListener(this.mouseAdapter);
        GridBagConstraints gbc_btnAttributeShowAll = new GridBagConstraints();
        gbc_btnAttributeShowAll.anchor = GridBagConstraints.NORTH;
        gbc_btnAttributeShowAll.insets = new Insets(0, 0, 0, 5);
        gbc_btnAttributeShowAll.gridx = 1;
        gbc_btnAttributeShowAll.gridy = 4;
        panelDataByAttribute.add(btnAttributeShowAll, gbc_btnAttributeShowAll);

        JPanel panelPrototypes = new JPanel();
        panelPrototypes
                .setBorder(new TitledBorder(null, "Prototype", TitledBorder.LEADING, TitledBorder.TOP, null, null));
        GridBagConstraints gbc_panelPrototypes = new GridBagConstraints();
        gbc_panelPrototypes.insets = new Insets(0, 0, 5, 0);
        gbc_panelPrototypes.fill = GridBagConstraints.BOTH;
        gbc_panelPrototypes.gridx = 0;
        gbc_panelPrototypes.gridy = 1;
        this.add(panelPrototypes, gbc_panelPrototypes);
        GridBagLayout gbl_panelPrototypes = new GridBagLayout();
        gbl_panelPrototypes.columnWidths = new int[] { 0, 0, 0, 0 };
        gbl_panelPrototypes.rowHeights = new int[] { 0, 0, 0, 0 };
        gbl_panelPrototypes.columnWeights = new double[] { 1.0, 0.0, 1.0, Double.MIN_VALUE };
        gbl_panelPrototypes.rowWeights = new double[] { 0.0, 0.0, 0.0, Double.MIN_VALUE };
        panelPrototypes.setLayout(gbl_panelPrototypes);

        JLabel lblAvailablePrototypes = new JLabel("Available Prototypes");
        GridBagConstraints gbc_lblAvailablePrototypes = new GridBagConstraints();
        gbc_lblAvailablePrototypes.insets = new Insets(0, 0, 5, 5);
        gbc_lblAvailablePrototypes.gridx = 0;
        gbc_lblAvailablePrototypes.gridy = 0;
        panelPrototypes.add(lblAvailablePrototypes, gbc_lblAvailablePrototypes);

        JLabel lblShowing = new JLabel("Showing");
        GridBagConstraints gbc_lblShowing = new GridBagConstraints();
        gbc_lblShowing.insets = new Insets(0, 0, 5, 0);
        gbc_lblShowing.gridx = 2;
        gbc_lblShowing.gridy = 0;
        panelPrototypes.add(lblShowing, gbc_lblShowing);

        JButton btnShowPrototype = new JButton("Show");
        btnShowPrototype.setName("PROTOTYPE_SHOW");
        btnShowPrototype.addMouseListener(this.mouseAdapter);
        GridBagConstraints gbc_btnShowPrototype = new GridBagConstraints();
        gbc_btnShowPrototype.insets = new Insets(0, 0, 5, 5);
        gbc_btnShowPrototype.gridx = 1;
        gbc_btnShowPrototype.gridy = 1;
        panelPrototypes.add(btnShowPrototype, gbc_btnShowPrototype);

        DefaultMutableTreeNode root = new DefaultMutableTreeNode("Prototypes");
        this.treePrototypes = new JTree(root);
        this.treePrototypes.getSelectionModel().setSelectionMode(TreeSelectionModel.SINGLE_TREE_SELECTION);
        JScrollPane scrollTreePrototypes = new JScrollPane();
        scrollTreePrototypes.setMinimumSize(new Dimension(100, 100));
        scrollTreePrototypes.setViewportView(this.treePrototypes);
        GridBagConstraints gbc_treePrototypes = new GridBagConstraints();
        gbc_treePrototypes.gridheight = 2;
        gbc_treePrototypes.insets = new Insets(0, 0, 0, 5);
        gbc_treePrototypes.fill = GridBagConstraints.BOTH;
        gbc_treePrototypes.gridx = 0;
        gbc_treePrototypes.gridy = 1;
        panelPrototypes.add(scrollTreePrototypes, gbc_treePrototypes);

        JButton btnRemovePrototype = new JButton("Remove");
        btnRemovePrototype.setName("PROTOTYPE_REMOVE");
        btnRemovePrototype.addMouseListener(this.mouseAdapter);
        GridBagConstraints gbc_btnRemovePrototype = new GridBagConstraints();
        gbc_btnRemovePrototype.anchor = GridBagConstraints.NORTH;
        gbc_btnRemovePrototype.insets = new Insets(0, 0, 0, 5);
        gbc_btnRemovePrototype.gridx = 1;
        gbc_btnRemovePrototype.gridy = 2;
        panelPrototypes.add(btnRemovePrototype, gbc_btnRemovePrototype);

        this.listShowingPrototypes = new JList<String>(this.listModelShowingPrototypes);
        GridBagConstraints gbc_listShowingPrototypes = new GridBagConstraints();
        gbc_listShowingPrototypes.gridheight = 2;
        gbc_listShowingPrototypes.fill = GridBagConstraints.BOTH;
        gbc_listShowingPrototypes.gridx = 2;
        gbc_listShowingPrototypes.gridy = 1;
        panelPrototypes.add(this.listShowingPrototypes, gbc_listShowingPrototypes);

        this.chartPanel = new ChartPanel(this.chart);
        GridBagConstraints gbc_panelChart = new GridBagConstraints();
        gbc_panelChart.fill = GridBagConstraints.BOTH;
        gbc_panelChart.gridx = 0;
        gbc_panelChart.gridy = 2;
        this.add(this.chartPanel, gbc_panelChart);
    }

    public void initializeTrainingData() {
        this.chartTrainingData = new DefaultCategoryDataset();
        for (Double classKey : this.classNamesForDouble.keySet()) {
            List<DataPoint> dataPointsWithLabel = LinearAlgebraicCalculations
                    .collectDatapointsWithClassLabel(this.dataPoints, classKey);
            double[] valuesForClass = LinearAlgebraicCalculations
                    .createMeanVectorFromListOfVectors(dataPointsWithLabel);
            for (int i = 0; i < valuesForClass.length; i++) {
                this.chartTrainingData.addValue(valuesForClass[i], this.classNamesForDouble.get(classKey),
                        this.attributeNames[i]);
            }
        }
    }

    private void initializeChart() {

        // Training Data
        final BarRenderer trainingRenderer = new BarRenderer();
        trainingRenderer.setShadowVisible(false);
        final CategoryPlot plot = new CategoryPlot();
        plot.setDataset(this.chartTrainingData);
        plot.setRenderer(trainingRenderer);
        for (String classLabel : this.classNamesForDouble.values()) {
            // set color to prepared class color
            trainingRenderer.setSeriesPaint(this.chartTrainingData.getRowIndex(classLabel),
                    this.classColors.get(classLabel));
        }

        // Prototype renderer
        this.chartPrototypeData = new ArrayList<DefaultCategoryDataset>();
        for (int i = 0; i < this.numberOfPrototypes; i++) {
            this.chartPrototypeData.add(new DefaultCategoryDataset());
            // set color to prepared class color
            CategoryItemRenderer chartRenderer = new LineAndShapeRenderer();
            plot.setRenderer(i + 1, chartRenderer);
            plot.setDataset(i + 1, this.chartPrototypeData.get(i));
        }

        plot.setDomainAxis(new CategoryAxis("Class"));
        plot.setRangeAxis(new NumberAxis("Value"));
        plot.setOrientation(PlotOrientation.VERTICAL);

        this.chart = new JFreeChart(plot);

    }

    private void fillLists() {
        // fill classes
        Random rand = new Random();
        for (String classLabel : this.classNamesForDouble.values()) {
            this.listModelShowByClass.addElement(classLabel);
            // assign random color
            float r = rand.nextFloat();
            float g = rand.nextFloat();
            float b = rand.nextFloat();
            this.classColors.put(classLabel, new Color(r, g, b));

        }
        // fill attributes
        for (String attribute : this.attributeNames) {
            this.listModelShowByAttribute.addElement(attribute);
        }
    }

    public void setPrototypes(List<Prototype> prototypes) {
        this.prototypes = prototypes;
        for (int prototypeIndex = 0; prototypeIndex < prototypes.size(); prototypeIndex++) {
            DefaultCategoryDataset series = this.chartPrototypeData.get(prototypeIndex);
            String prototypeClassLabel = this.classNamesForDouble
                    .get(this.prototypes.get(prototypeIndex).getClassLabel());

            ((CategoryPlot) this.chart.getPlot()).getRenderer(prototypeIndex + 1).setSeriesPaint(0,
                    this.classColors.get(prototypeClassLabel));

            for (int valueIndex = 0; valueIndex < prototypes.get(prototypeIndex).getDimension(); valueIndex++) {
                series.setValue(prototypes.get(prototypeIndex).getValue(valueIndex), prototypeClassLabel,
                        this.attributeNames[valueIndex]);
            }
        }
        this.chart.fireChartChanged();
        // this.chartPanel.revalidate();
        this.chartPanel.repaint();

        if (!this.prototypesInitialized) {
            // fill tree
            DefaultMutableTreeNode root = new DefaultMutableTreeNode("Prototypes");
            for (String classLabel : this.classNamesForDouble.values()) {
                DefaultMutableTreeNode classNode = new DefaultMutableTreeNode(classLabel);
                for (int i = 0; i < this.prototypes.size(); i++) {
                    if (this.classNamesForDouble.get(this.prototypes.get(i).getClassLabel()).equals(classLabel)) {
                        DefaultMutableTreeNode prototypeNode = new DefaultMutableTreeNode("Prototype " + i);
                        classNode.add(prototypeNode);
                        this.listModelShowingPrototypes.addElement("Prototype " + i);
                    }

                }
                root.add(classNode);
            }
            DefaultTreeModel model = (DefaultTreeModel) this.treePrototypes.getModel();
            model.setRoot(root);
            model.reload(root);
            this.prototypesInitialized = true;
        }

    }

    public void moveAll(DefaultListModel<String> source, DefaultListModel<String> target, String componentName) {
        List<String> tempList = new ArrayList<String>();
        // create temporary list
        for (int i = 0; i < source.getSize(); i++) {
            String classLabel = source.getElementAt(i);
            tempList.add(classLabel);
        }
        for (String currentLabel : tempList) {
            moveOne(source, target, currentLabel, componentName);
        }
    }

    public void moveOne(DefaultListModel<String> source, DefaultListModel<String> target, String element,
            String componentName) {
        if (element != null) {
            target.addElement(element);
            source.removeElement(element);
        }
        if (componentName.contains("SHOW")) {
            if (componentName.contains("CLASS")) {
                // SHOW CLASS
                showClass(element);
            } else {
                // SHOW ATTRIBUTE
                showAttribute(element);
            }
        } else {
            if (componentName.contains("CLASS")) {
                // HIDE CLASS
                hideClass(element);
            } else {
                // HIDE ATTRIBUTE
                hideAttribute(element);
            }
        }
    }

    public void showClass(String classLabel) {
        ((CategoryPlot) this.chart.getPlot()).getRenderer(0)
                .setSeriesVisible(this.chartTrainingData.getRowIndex(classLabel), true);
    }

    public void hideClass(String classLabel) {
        ((CategoryPlot) this.chart.getPlot()).getRenderer(0)
                .setSeriesVisible(this.chartTrainingData.getRowIndex(classLabel), false);
    }

    public void hideAttribute(String attributeLabel) {
        this.chartTrainingData.removeColumn(this.chartTrainingData.getColumnIndex(attributeLabel));
    }

    public void showAttribute(String attributeLabel) {
        int labelIndex = -1;
        for (int i = 0; i < this.attributeNames.length; i++) {
            if (this.attributeNames[i].equals(attributeLabel)) {
                labelIndex = i;
                break;
            }
        }
        for (Double classKey : this.classNamesForDouble.keySet()) {
            List<DataPoint> dataPointsWithLabel = LinearAlgebraicCalculations
                    .collectDatapointsWithClassLabel(this.dataPoints, classKey);
            double[] valuesForClass = LinearAlgebraicCalculations
                    .createMeanVectorFromListOfVectors(dataPointsWithLabel);
            this.chartTrainingData.addValue(valuesForClass[labelIndex], this.classNamesForDouble.get(classKey),
                    this.attributeNames[labelIndex]);
        }
    }

    public void showProtoype(DefaultMutableTreeNode nodeToShow) {
        String label = (String) nodeToShow.getUserObject();
        Pattern pattern = Pattern.compile("Prototype (\\d+)");
        Matcher matcher = pattern.matcher(label);
        if (matcher.matches()) {
            int rendererIndex = Integer.valueOf(matcher.group(1));
            ((CategoryPlot) this.chart.getPlot()).getRenderer(rendererIndex + 1).setSeriesVisible(0, true);
            this.listModelShowingPrototypes.addElement(label);
        }
    }

    public void hideProtoype(String element) {
        Pattern pattern = Pattern.compile("Prototype (\\d+)");
        Matcher matcher = pattern.matcher(element);
        if (matcher.matches()) {
            int rendererIndex = Integer.valueOf(matcher.group(1));
            ((CategoryPlot) this.chart.getPlot()).getRenderer(rendererIndex + 1).setSeriesVisible(0, false);
            this.listModelShowingPrototypes.removeElement(element);
        }
    }

    public DefaultListModel<String> getListModelHideByClass() {
        return this.listModelHideByClass;
    }

    public DefaultListModel<String> getListModelShowByClass() {
        return this.listModelShowByClass;
    }

    public DefaultListModel<String> getListModelHideByAttribute() {
        return this.listModelHideByAttribute;
    }

    public DefaultListModel<String> getListModelShowByAttribute() {
        return this.listModelShowByAttribute;
    }

    public JList<String> getListShowByClass() {
        return this.listShowByClass;
    }

    public JList<String> getListHideByClass() {
        return this.listHideByClass;
    }

    public JList<String> getListHideByAttribute() {
        return this.listHideByAttribute;
    }

    public JList<String> getListShowByAttribute() {
        return this.listShowByAttribute;
    }

    public DefaultListModel<String> getListModelShowingPrototypes() {
        return this.listModelShowingPrototypes;
    }

    public void setListModelShowingPrototypes(DefaultListModel<String> listModelShowingPrototypes) {
        this.listModelShowingPrototypes = listModelShowingPrototypes;
    }

    public JList<String> getListShowingPrototypes() {
        return this.listShowingPrototypes;
    }

    public void setListShowingPrototypes(JList<String> listShowingPrototypes) {
        this.listShowingPrototypes = listShowingPrototypes;
    }

    public JTree getTreePrototypes() {
        return this.treePrototypes;
    }

    public void setTreePrototypes(JTree treePrototypes) {
        this.treePrototypes = treePrototypes;
    }

}
