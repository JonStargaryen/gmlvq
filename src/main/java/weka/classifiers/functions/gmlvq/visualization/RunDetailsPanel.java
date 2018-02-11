package weka.classifiers.functions.gmlvq.visualization;

import weka.classifiers.functions.gmlvq.core.GMLVQCore;

import javax.swing.*;
import java.awt.*;

public class RunDetailsPanel extends JPanel {

    public RunDetailsPanel(GMLVQCore gmlvqCore) {
        initializeInterface(gmlvqCore);
    }

    private void initializeInterface(GMLVQCore gmlvqCore) {
        this.setLayout(new BorderLayout());
        JTextPane descriptionPane = new JTextPane();
        descriptionPane.setText(gmlvqCore.getDetailString());
        descriptionPane.setEditable(false);
        this.add(descriptionPane, BorderLayout.CENTER);
    }
}
