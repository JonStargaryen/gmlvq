package weka.classifiers.functions.gmlvq.visualization;

import weka.classifiers.functions.gmlvq.utilities.LinearAlgebraicCalculations;
import weka.core.matrix.Matrix;

import java.awt.*;
import java.io.Serializable;
import java.text.DecimalFormat;

public class MatrixRenderer implements Serializable {

    private static final long serialVersionUID = 1L;

    private LambdaMatrixPanel parent;
    private Matrix lambdaMatrix;
    // private Graphics2D graphics;
    private ColorScale colorScale;

    // options
    private boolean showScale;

    // matrix
    private int matrixDrawWidth;
    private int matrixDrawHeight;
    private int matrixElementSize;
    private int matrixMarginHorizontal;
    private int matrixMarginVertical;

    private int rowDimension;
    private int columnDimension;

    private DecimalFormat decimalFormat = new DecimalFormat("0.000");

    public MatrixRenderer(LambdaMatrixPanel lambdaMatrixPanel, ColorScale colorScale) {
        this.parent = lambdaMatrixPanel;
        this.colorScale = colorScale;
        // this.graphics = (Graphics2D) this.parent.getGraphics();
        this.showScale = true;
    }

    public void setLambdaMatrix(Matrix lambdaMatrix) {
        this.lambdaMatrix = lambdaMatrix;
    }

    private void drawLambdaMatrix(Graphics2D graphics) {

        for (int rowIndex = 0; rowIndex < this.lambdaMatrix.getRowDimension(); rowIndex++) {
            for (int colIndex = 0; colIndex < this.lambdaMatrix.getColumnDimension(); colIndex++) {
                double matrixElementValue = this.lambdaMatrix.get(rowIndex, colIndex);
                Color matrixElementColor = this.colorScale.getColor(new Double(matrixElementValue).floatValue());
                graphics.setPaint(matrixElementColor);
                graphics.fillRect(this.matrixMarginHorizontal / 2 + this.matrixElementSize * rowIndex,
                        this.matrixMarginVertical / 2 + this.matrixElementSize * colIndex, this.matrixElementSize,
                        this.matrixElementSize);
            }
        }

    }

    private void drawScale(Graphics2D graphics) {

        int legendScaleElementHeight = this.rowDimension * this.matrixElementSize / 100;
        int legendScaleDrawWidth = 30;
        int legendScaleHorizontalMargin = this.matrixDrawHeight - 100 * legendScaleElementHeight;
        int xStart = this.matrixMarginHorizontal * 2 / 3 + this.matrixElementSize * this.rowDimension;
        graphics.setFont(new Font("Courier New", Font.CENTER_BASELINE, 16));

        float currentValue = this.colorScale.getMaximalValue();

        for (int scaleElementIndex = 0; scaleElementIndex < 100; scaleElementIndex++) {
            currentValue -= (this.colorScale.getMaximalValue() - this.colorScale.getMinimalValue()) / 100f;
            graphics.setPaint(this.colorScale.getColor(currentValue));
            graphics.fillRect(xStart, legendScaleHorizontalMargin / 2 + scaleElementIndex * legendScaleElementHeight,
                    legendScaleDrawWidth, legendScaleElementHeight);
            // draw tick text
            if (scaleElementIndex == 0 || scaleElementIndex == 99) {
                graphics.setPaint(Color.BLACK);
                graphics.drawString(this.decimalFormat.format(currentValue), xStart + legendScaleDrawWidth + 10,
                        legendScaleHorizontalMargin / 2 + (scaleElementIndex + 1) * legendScaleElementHeight);
            }
        }

        // draw zero tick = horizontal marign + scale height *
        double positionFactor = this.colorScale.getMaximalValue()
                / (this.colorScale.getMaximalValue() - this.colorScale.getMinimalValue());
        int zeroPosition = (int) (legendScaleHorizontalMargin / 2.0 + legendScaleElementHeight * 100 * positionFactor);
        graphics.setPaint(Color.BLACK);
        graphics.drawString(this.decimalFormat.format(0.0), xStart + legendScaleDrawWidth + 10, zeroPosition);
    }

    private void initialize() {
        // initialize the color scale
        double[] minAndMaxValues = LinearAlgebraicCalculations.getMinAndMaxValuesFromMatrix(this.lambdaMatrix/*.copy()*/);
        float minValue = (float) minAndMaxValues[LinearAlgebraicCalculations.MINIMAL_INDEX];
        float maxValue = (float) minAndMaxValues[LinearAlgebraicCalculations.MAXIMAL_INDEX];
        this.colorScale = new ColorScale.Builder(minValue, maxValue).build();

        // determine ideal matrix size
        this.matrixDrawWidth = (int) (this.parent.getWidth() * 0.9);
        this.matrixDrawHeight = (int) (this.parent.getHeight() * 0.95);
        this.columnDimension = this.lambdaMatrix.getColumnDimension();
        this.rowDimension = this.lambdaMatrix.getRowDimension();

        // determine ideal size of the matrix elements
        int smallestDimension = this.matrixDrawWidth > this.matrixDrawHeight ? this.matrixDrawHeight
                : this.matrixDrawWidth;
        this.matrixElementSize = smallestDimension
                / (this.rowDimension >= this.columnDimension ? this.columnDimension : this.rowDimension);

        // determine margins
        this.matrixMarginHorizontal = this.matrixDrawWidth - this.columnDimension * this.matrixElementSize;
        this.matrixMarginVertical = this.matrixDrawHeight - this.columnDimension * this.matrixElementSize;

        this.parent.setElementSize(this.matrixElementSize);
        this.parent.setHorizontalMargin(this.matrixMarginHorizontal);
        this.parent.setVerticalMargin(this.matrixMarginVertical);

    }

    public void draw(Graphics2D graphics, Matrix lambdaMatrix) {
        this.lambdaMatrix = lambdaMatrix;
        redraw(graphics);
    }

    public void redraw(Graphics2D graphics) {
        graphics.clearRect(0, 0, this.parent.getWidth(), this.parent.getHeight());
        graphics.setBackground(Color.WHITE);
        if (this.lambdaMatrix != null) {
            initialize();
            drawLambdaMatrix(graphics);
            if (this.showScale) {
                drawScale(graphics);
            }
        }
    }

    public boolean isShowScale() {
        return this.showScale;
    }

    public int getMatrixDrawWidth() {
        return matrixDrawWidth;
    }

    public int getMatrixDrawHeight() {
        return matrixDrawHeight;
    }

    public void setShowScale(boolean showScale) {
        this.showScale = showScale;
        if (this.parent.getGraphics() != null) {
            this.redraw((Graphics2D) this.parent.getGraphics());
        }
    }

}
