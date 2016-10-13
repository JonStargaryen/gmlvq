package weka.classifiers.functions.gmlvq.visualization;

import java.awt.Color;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Point;
import java.awt.event.MouseEvent;
import java.io.File;
import java.io.IOException;
import java.text.DecimalFormat;

import javax.swing.JPanel;
import javax.swing.ToolTipManager;

import org.jfree.graphics2d.svg.SVGGraphics2D;
import org.jfree.graphics2d.svg.SVGUtils;

import weka.core.matrix.Matrix;

public class LambdaMatrixPanel extends JPanel {

    private static final long serialVersionUID = 1L;

    // attributes
    private String[] attributeNames;
    private MatrixRenderer renderer;
    private Matrix lambdaMatrix;
    private DecimalFormat decimalFormat = new DecimalFormat("0.000");

    // sizes
    private int elementSize;
    private int horizontalMargin;
    private int verticalMargin;

    public LambdaMatrixPanel(String[] attributeNames, ColorScale colorScale) {
        this.attributeNames = attributeNames;
        this.setRenderer(new MatrixRenderer(this, colorScale));
        this.setBackground(Color.WHITE);
        // register to show tooltips
        ToolTipManager.sharedInstance().registerComponent(this);
    }

    @Override
    public String getToolTipText(MouseEvent event) {
        Point p = new Point(event.getX(), event.getY());
        String t = buildToolTipForMatrixElement(p);
        if (t != null) {
            return t;
        }
        return super.getToolTipText(event);
    }

    /**
     * creates a string for a tool tip based on the position of the given point
     *
     * @param p
     * @return
     */
    private String buildToolTipForMatrixElement(Point p) {
        if (isPointInMatrix(p)) {
            int rowIndex = determineRowAttributeIndex(p);
            int columnIndex = determineColumnAttributeIndex(p);
            if (rowIndex == columnIndex) {
                return this.attributeNames[rowIndex] + " "
                        + this.decimalFormat.format(this.lambdaMatrix.get(rowIndex, columnIndex));
            }
            return this.attributeNames[rowIndex] + " : " + this.attributeNames[columnIndex] + " "
                    + this.decimalFormat.format(this.lambdaMatrix.get(rowIndex, columnIndex));
        }
        return null;
    }

    /**
     * determines whether the point is in the matrix
     *
     * @param p
     * @return
     */
    private boolean isPointInMatrix(Point p) {
        final int leftMostPosition = this.horizontalMargin / 2;
        final int rightMostPosition = this.horizontalMargin / 2 + this.attributeNames.length * this.elementSize;
        final int topMostPoint = this.verticalMargin / 2;
        final int matrixBottomBoder = this.verticalMargin / 2 + this.attributeNames.length * this.elementSize;
        if (p.getX() > leftMostPosition && p.getX() < rightMostPosition && p.getY() > topMostPoint
                && p.getY() < matrixBottomBoder) {
            return true;
        }
        return false;
    }

    /**
     * determines the row index based on the given point
     *
     * @param p
     * @return
     */
    private int determineRowAttributeIndex(Point p) {
        return (int) ((p.getY() - this.verticalMargin / 2) / this.elementSize);
    }

    /**
     * determines the column index based on the given point
     *
     * @param p
     * @return
     */
    private int determineColumnAttributeIndex(Point p) {
        return (int) ((p.getX() - this.horizontalMargin / 2) / this.elementSize);
    }

    public void exportLambdaMatrixToSVG(File file) {
        SVGGraphics2D g2 = new SVGGraphics2D(300, 200);
        g2.setBackground(Color.WHITE);
        this.getRenderer().draw(g2, this.getLambdaMatrix());
        String svgElement = g2.getSVGElement();
        try {
            SVGUtils.writeToSVG(file, svgElement);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    public void paintComponent(Graphics g) {
        super.paintComponent(g);
        Graphics2D graphics = (Graphics2D) g;
        if (this.lambdaMatrix != null) {
            this.getRenderer().draw(graphics, this.getLambdaMatrix());
        } else {
            graphics.setColor(Color.BLACK);
            graphics.drawString("Initializing ...", 10, 25);
        }

    }

    public Matrix getLambdaMatrix() {
        return this.lambdaMatrix;
    }

    public void setLambdaMatrix(Matrix lambdaMatrix) {
        this.lambdaMatrix = lambdaMatrix;
    }

    public MatrixRenderer getRenderer() {
        return this.renderer;
    }

    public void setRenderer(MatrixRenderer renderer) {
        this.renderer = renderer;
    }

    public int getElementSize() {
        return this.elementSize;
    }

    public void setElementSize(int elementSize) {
        this.elementSize = elementSize;
    }

    public int getHorizontalMargin() {
        return this.horizontalMargin;
    }

    public void setHorizontalMargin(int horizontalMargin) {
        this.horizontalMargin = horizontalMargin;
    }

    public int getVerticalMargin() {
        return this.verticalMargin;
    }

    public void setVerticalMargin(int verticalMargin) {
        this.verticalMargin = verticalMargin;
    }

}
