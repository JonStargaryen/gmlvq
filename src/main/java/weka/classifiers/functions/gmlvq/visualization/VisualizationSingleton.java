package weka.classifiers.functions.gmlvq.visualization;

import javax.swing.*;
import java.awt.*;
import java.util.ArrayList;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

public class VisualizationSingleton extends JFrame {

    /**
     * singleton instance
     */
    private static VisualizationSingleton instance;

    /**
     * visualizations
     */
    private Map<Integer, Visualizer> visualizations;

    private AtomicInteger counter;

    private JToolBar toolbar;
    private JTabbedPane tabbedPane;

    private static synchronized VisualizationSingleton getInstance () {
        if (VisualizationSingleton.instance == null) {
            VisualizationSingleton.instance = new VisualizationSingleton();
        }
        return VisualizationSingleton.instance;
    }

    public static int addVisualization(Visualizer visualizer) {
        int current = getInstance().counter.incrementAndGet();
        getInstance().visualizations.put(current, visualizer);
        getInstance().tabbedPane.addTab("V"+current, visualizer);
        return current;
    }

    public static Visualizer getLastVisualizalizer() {
        return getInstance().visualizations.get(getInstance().counter.get());
    }

    public static void showVisualizations() {
        getInstance().setVisible(true);
    }

    private VisualizationSingleton() {
        visualizations = new ConcurrentHashMap<>();
        counter = new AtomicInteger(0);
        initializeInterface();
    }

    private void initializeInterface() {
        this.setTitle("GMLVQ Live Visualization");
        this.setDefaultCloseOperation(DISPOSE_ON_CLOSE);
        this.setSize(800, 600);

        toolbar = new JToolBar();
        Button closeButton = new Button("Clear stale visualizations");
        toolbar.add(closeButton);
        closeButton.addActionListener(actionListener -> {
            ArrayList<Integer> visualizersToRemove = new ArrayList<>();
            for (Map.Entry<Integer, Visualizer> visualizer : getInstance().visualizations.entrySet()) {
                if (visualizer.getValue() != getLastVisualizalizer()) {
                    visualizersToRemove.add(visualizer.getKey());
                }
            }
            for (Integer key : visualizersToRemove) {
                getInstance().visualizations.remove(key);
                for (int tabIndex = 0; tabIndex < getInstance().tabbedPane.getTabCount(); tabIndex++) {
                    if (getInstance().tabbedPane.getTitleAt(tabIndex).matches("V"+key)) {
                        getInstance().tabbedPane.removeTabAt(tabIndex);
                        break;
                    }
                }
            }
        });

        getContentPane().add(toolbar, BorderLayout.NORTH);

        tabbedPane = new JTabbedPane(JTabbedPane.LEFT);
        getContentPane().add(tabbedPane, BorderLayout.CENTER);
    }




}
