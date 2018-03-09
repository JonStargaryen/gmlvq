package weka.classifiers.functions.gmlvq.visualization;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.block.BlockBorder;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import weka.classifiers.functions.gmlvq.core.cost.CostFunctionValue;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ItemEvent;
import java.awt.event.ItemListener;
import java.util.*;
import java.util.List;

public class CostFunctionChartPanel extends JPanel implements ItemListener {

    private static final long serialVersionUID = 1L;

    private JFreeChart chart;
    private XYSeriesCollection chartDataset;
    private XYLineAndShapeRenderer chartRenderer;

    private List<CostFunctionValue> costFunctions;
    private Map<String, Color> costFunctionColors = new HashMap<String, Color>();

    public CostFunctionChartPanel(Map<CostFunctionValue, Double> currentCostValues) {
        this.costFunctions = new ArrayList<CostFunctionValue>();
        for (CostFunctionValue costFuntion : currentCostValues.keySet()) {
            this.costFunctions.add(costFuntion);
        }
        initializeChart();
        initializeInterface();
    }

    private void initializeChart() {
        // initialize series collections
        this.chartDataset = new XYSeriesCollection();

        // setup all cost functions
        Random rand = new Random();
        for (CostFunctionValue costFunction : this.costFunctions) {
            // add cost function
            String costFunctionName = costFunction.toString();
            XYSeries series = new XYSeries(costFunctionName);
            this.chartDataset.addSeries(series);
            // assign random color
            float r = rand.nextFloat();
            float g = rand.nextFloat();
            float b = rand.nextFloat();
            this.costFunctionColors.put(costFunctionName, new Color(r, g, b));
        }
        // initialize options
        this.chart = ChartFactory.createXYLineChart("Cost Functions", // title
                "epoch", // x axis label
                "cost value", // y axis label
                this.chartDataset, // data
                PlotOrientation.VERTICAL, // orientation
                true, // include legend
                true, // tooltips
                false // urls
        );

        // chart and plot options
        this.chart.setBackgroundPaint(Color.white);
        this.chart.setBorderVisible(false);
        this.chart.getLegend().setFrame(BlockBorder.NONE);
        this.chart.getRenderingHints().put(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        XYPlot plot = this.chart.getXYPlot();
        plot.setBackgroundPaint(Color.white);
        plot.setDomainGridlinePaint(Color.white);
        plot.setRangeGridlinePaint(Color.white);

        // setup renderer
        this.chartRenderer = new XYLineAndShapeRenderer();
        this.chartRenderer.setBaseShapesVisible(false);
        plot.setRenderer(this.chartRenderer);
        // set colors for series
        for (int i = 0; i < this.chartDataset.getSeriesCount(); i++) {
            this.chartRenderer.setSeriesStroke(i, new BasicStroke(2f));
            this.chartRenderer.setSeriesPaint(i, this.costFunctionColors.get(this.chartDataset.getSeriesKey(i)));
        }

        // range axis options
        NumberAxis rangeAxis = (NumberAxis) plot.getRangeAxis();
        rangeAxis.setStandardTickUnits(NumberAxis.createStandardTickUnits(Locale.US));
        rangeAxis.setLabelFont(new Font(Font.SANS_SERIF, Font.BOLD, 12));
        rangeAxis.setRange(0.0, 1.0);

        // domain axis options
        NumberAxis domainAxis = (NumberAxis) plot.getDomainAxis();
        domainAxis.setStandardTickUnits(NumberAxis.createIntegerTickUnits(Locale.US));
        domainAxis.setLabelFont(new Font(Font.SANS_SERIF, Font.BOLD, 12));
    }

    private void initializeInterface() {
        // setup layout
        this.setLayout(new BorderLayout());
        ChartPanel chartPanel = new ChartPanel(this.chart);
        this.add(chartPanel, BorderLayout.CENTER);

        JPanel topPanel = new JPanel();
        topPanel.setLayout(new BorderLayout());
        this.add(topPanel, BorderLayout.NORTH);

        JTextPane descriptionPane = new JTextPane();
        descriptionPane.setText("The following cost functions are evaluated during the learning process. Only one" +
                " is optimized (see Run details panel), the others are calculated and evaluated for visualization purposes. The displayed value" +
                " is only valid for the current (sub)set of data points, the cross-validated values are displayed in the WEKA log.");
        descriptionPane.setEditable(false);
        descriptionPane.setBackground(this.getBackground());
        topPanel.add(descriptionPane, BorderLayout.NORTH);

        // setup tool bar
        JToolBar toolBar = new JToolBar();
        toolBar.setFloatable(false);
        topPanel.add(toolBar, BorderLayout.SOUTH);

        // add check boxes for each cost function
        for (CostFunctionValue costFunction : this.costFunctions) {
            JCheckBox checkBox = new JCheckBox(costFunction.toString());
            checkBox.setName(costFunction.name());
            checkBox.setSelected(true);
            checkBox.addItemListener(this);
            toolBar.add(checkBox);
        }

    }

    public void addLatestValues(Map<CostFunctionValue, Double> currentCostValues) {
        // add additional cost function values
        for (CostFunctionValue costFunctionValue : currentCostValues.keySet()) {
            if (!costFunctionValue.equals(CostFunctionValue.COST_FUNCTION_VALUE_TO_OPTIMIZE)) {
                XYSeries series = this.chartDataset.getSeries(costFunctionValue.toString());
                series.addOrUpdate(series.getItemCount(), currentCostValues.get(costFunctionValue).doubleValue());
            }
        }
    }

    @Override
    public void itemStateChanged(ItemEvent itemEvent) {
        // get toogled check box
        String componentName = ((JCheckBox) itemEvent.getItem()).getName();
        // switch display state
        if (itemEvent.getStateChange() == ItemEvent.SELECTED) {
            this.chartRenderer.setSeriesVisible(this.chartDataset.getSeriesIndex(componentName), true);
        } else {
            this.chartRenderer.setSeriesVisible(this.chartDataset.getSeriesIndex(componentName), false);
        }
    }


}
