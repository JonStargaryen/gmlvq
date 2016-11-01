package weka.classifiers.functions.gmlvq.visualization;

import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;

import javax.swing.tree.DefaultMutableTreeNode;

public class VisualizerMouseAdapter extends MouseAdapter {

    private Visualizer parent;

    public VisualizerMouseAdapter(Visualizer parent) {
        this.parent = parent;
    }

    @Override
    public void mouseClicked(MouseEvent e) {
        String componentName = e.getComponent().getName();
        // Debug correct assignment
        if (componentName.equals("SHOW_SCALE")) {
            this.parent.switchScale();
        } else if (componentName.equals("EXPORT_LAMBDA_MATRIX")) {
            this.parent.saveLamdaMatrixToSVG();
        } else if (componentName.equals("CLASS_HIDE_ALL")) {
            this.parent.getFeatureAnalysisPanel().moveAll(
                    this.parent.getFeatureAnalysisPanel().getListModelShowByClass(),
                    this.parent.getFeatureAnalysisPanel().getListModelHideByClass(), componentName);
        } else if (componentName.equals("CLASS_SHOW_ALL")) {
            this.parent.getFeatureAnalysisPanel().moveAll(
                    this.parent.getFeatureAnalysisPanel().getListModelHideByClass(),
                    this.parent.getFeatureAnalysisPanel().getListModelShowByClass(), componentName);
        } else if (componentName.equals("CLASS_SHOW_ONE")) {
            int index = this.parent.getFeatureAnalysisPanel().getListHideByClass().getSelectedIndex();
            String classLabel = this.parent.getFeatureAnalysisPanel().getListModelHideByClass().getElementAt(index);
            this.parent.getFeatureAnalysisPanel().moveOne(
                    this.parent.getFeatureAnalysisPanel().getListModelHideByClass(),
                    this.parent.getFeatureAnalysisPanel().getListModelShowByClass(), classLabel, componentName);
        } else if (componentName.equals("CLASS_HIDE_ONE")) {
            int index = this.parent.getFeatureAnalysisPanel().getListShowByClass().getSelectedIndex();
            String classLabel = this.parent.getFeatureAnalysisPanel().getListModelShowByClass().getElementAt(index);
            this.parent.getFeatureAnalysisPanel().moveOne(
                    this.parent.getFeatureAnalysisPanel().getListModelShowByClass(),
                    this.parent.getFeatureAnalysisPanel().getListModelHideByClass(), classLabel, componentName);
        } else if (componentName.equals("ATTRIBUTE_HIDE_ALL")) {
            this.parent.getFeatureAnalysisPanel().moveAll(
                    this.parent.getFeatureAnalysisPanel().getListModelShowByAttribute(),
                    this.parent.getFeatureAnalysisPanel().getListModelHideByAttribute(), componentName);
        } else if (componentName.equals("ATTRIBUTE_SHOW_ALL")) {
            this.parent.getFeatureAnalysisPanel().moveAll(
                    this.parent.getFeatureAnalysisPanel().getListModelHideByAttribute(),
                    this.parent.getFeatureAnalysisPanel().getListModelShowByAttribute(), componentName);
        } else if (componentName.equals("ATTRIBUTE_SHOW_ONE")) {
            int index = this.parent.getFeatureAnalysisPanel().getListHideByAttribute().getSelectedIndex();
            String attributeLabel = this.parent.getFeatureAnalysisPanel().getListModelHideByAttribute()
                    .getElementAt(index);
            this.parent.getFeatureAnalysisPanel().moveOne(
                    this.parent.getFeatureAnalysisPanel().getListModelHideByAttribute(),
                    this.parent.getFeatureAnalysisPanel().getListModelShowByAttribute(), attributeLabel, componentName);
        } else if (componentName.equals("ATTRIBUTE_HIDE_ONE")) {
            int index = this.parent.getFeatureAnalysisPanel().getListShowByAttribute().getSelectedIndex();
            String attributeLabel = this.parent.getFeatureAnalysisPanel().getListModelShowByAttribute()
                    .getElementAt(index);
            this.parent.getFeatureAnalysisPanel().moveOne(
                    this.parent.getFeatureAnalysisPanel().getListModelShowByAttribute(),
                    this.parent.getFeatureAnalysisPanel().getListModelHideByAttribute(), attributeLabel, componentName);
            this.parent.getFeatureAnalysisPanel().hideAttribute(attributeLabel);
        } else if (componentName.equals("PROTOTYPE_REMOVE")) {
            int index = this.parent.getFeatureAnalysisPanel().getListShowingPrototypes().getSelectedIndex();
            String prototypeLabel = this.parent.getFeatureAnalysisPanel().getListModelShowingPrototypes()
                    .getElementAt(index);
            this.parent.getFeatureAnalysisPanel().hideProtoype(prototypeLabel);
        } else if (componentName.equals("PROTOTYPE_SHOW")) {
            DefaultMutableTreeNode node = (DefaultMutableTreeNode) this.parent.getFeatureAnalysisPanel()
                    .getTreePrototypes().getLastSelectedPathComponent();
            if (node == null) {
                return;
            }
            this.parent.getFeatureAnalysisPanel().showProtoype(node);
        }
    }

}
