package weka.classifiers.functions.gmlvq.visualization;

import java.awt.Color;
import java.io.Serializable;

/**
 * This object provides a color scale between the given minimal and maximal
 * values.
 *
 * @author Christoph Leberecht
 *
 */
public class ColorScale implements Serializable {

    private static final long serialVersionUID = 1L;

    // from green to red over yellow
    private static final float defaultMinimalHue = 1.0f / 3.0f;
    private static final float defaultMaximalHue = 0.0f;
    // somewhat pastell colors
    private static final float defaultSaturation = 0.6f;
    private static final float defaultBrightness = 0.9f;

    private final float minimalValue;
    private final float maximalValue;
    private final float scalingFactor;

    private final float minimalHue;

    private final float saturation;
    private final float brightness;

    public static class Builder {

        private float minimalValue;
        private float maximalValue;
        private float scalingFactor;
        private float offset;

        private float minimalHue = defaultMinimalHue;
        private float maximalHue = defaultMaximalHue;

        private float saturation = defaultSaturation;
        private float brightness = defaultBrightness;

        public Builder(float minimalValue, float maximalValue) {
            if (minimalValue > maximalValue) {
                throw new IllegalArgumentException("The minimal value hue has to be smaller than the maximal value.");
            }

            if (minimalValue < 0.0f) {
                this.offset = this.minimalValue;
                this.minimalValue = 0.0f;
                this.maximalValue += this.offset;
            } else {
                this.offset = -this.minimalValue;
                this.minimalValue = 0.0f;
                this.maximalValue += this.offset;
            }
            this.minimalValue = minimalValue;
            this.maximalValue = maximalValue;
        }

        public Builder minimalHue(float minimalHue) {
            if (minimalHue < 0.0f || minimalHue > 360.0f) {
                throw new IllegalArgumentException("The value for hues has to be between 0.0 and 360.0");
            }
            this.minimalHue = minimalHue;
            return this;
        }

        public Builder minimalHue(Color minimalHueColor) {
            this.minimalHue = Color.RGBtoHSB(minimalHueColor.getRed(), minimalHueColor.getGreen(),
                    minimalHueColor.getBlue(), null)[0];
            return this;
        }

        public Builder maximalHue(float maximalHue) {
            if (maximalHue < 0.0f || maximalHue > 360.0f) {
                throw new IllegalArgumentException("The value for hues has to be between 0.0 and 360.0");
            }
            this.maximalHue = maximalHue;
            return this;
        }

        public Builder maximalHue(Color maximalHueColor) {
            this.maximalHue = Color.RGBtoHSB(maximalHueColor.getRed(), maximalHueColor.getGreen(),
                    maximalHueColor.getBlue(), null)[0];
            return this;
        }

        public Builder saturation(float saturation) {
            if (saturation < 0.0f || saturation > 1.0f) {
                throw new IllegalArgumentException("The value for saturation has to be between 0.0 and 1.0");
            }
            this.saturation = saturation;
            return this;
        }

        public Builder brightness(float brightness) {
            if (brightness < 0.0f || brightness > 1.0f) {
                throw new IllegalArgumentException("The value for brightness has to be between 0.0 and 1.0");
            }
            this.brightness = brightness;
            return this;
        }

        public ColorScale build() {
            // scale min and max values
            this.scalingFactor = (this.maximalHue - this.minimalHue) / (this.maximalValue - this.minimalValue);
            return new ColorScale(this);
        }

    }

    private ColorScale(Builder builder) {
        this.minimalValue = builder.minimalValue;
        this.maximalValue = builder.maximalValue;
        this.scalingFactor = builder.scalingFactor;
        this.minimalHue = builder.minimalHue;
        this.saturation = builder.saturation;
        this.brightness = builder.brightness;
    }

    /**
     * Gets the color as specified by this gradient for this value between the
     * minimal and maximal value.
     *
     * @param value
     * @return
     */
    public Color getColor(float value) {
        final float requestedHue = (value - this.getMinimalValue()) * this.scalingFactor + this.minimalHue;
        return Color.getHSBColor(requestedHue, this.saturation, this.brightness);
    }

    public float getMinimalValue() {
        return this.minimalValue;
    }

    public float getMaximalValue() {
        return this.maximalValue;
    }

}
