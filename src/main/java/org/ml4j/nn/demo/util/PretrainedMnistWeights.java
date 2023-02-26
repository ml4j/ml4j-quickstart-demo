package org.ml4j.nn.demo.util;

import java.util.Arrays;

import org.ml4j.MatrixFactory;
import org.ml4j.nn.axons.BiasFormatImpl;
import org.ml4j.nn.axons.BiasVector;
import org.ml4j.nn.axons.BiasVectorImpl;
import org.ml4j.nn.axons.FeaturesVectorFormat;
import org.ml4j.nn.axons.FeaturesVectorOrientation;
import org.ml4j.nn.axons.WeightsFormatImpl;
import org.ml4j.nn.axons.WeightsMatrix;
import org.ml4j.nn.axons.WeightsMatrixImpl;
import org.ml4j.nn.axons.WeightsMatrixOrientation;
import org.ml4j.nn.neurons.format.features.Dimension;
import org.ml4j.util.SerializationHelper;

public class PretrainedMnistWeights {
	
	private WeightsMatrix layer1Weights;
	private BiasVector layer1Biases;
	private WeightsMatrix layer3Weights;
	private BiasVector layer3Biases;
	private WeightsMatrix layer4Weights;
	private BiasVector layer4Biases;
	private WeightsMatrix layer5Weights;
	private BiasVector layer5Biases;
	
	public PretrainedMnistWeights(MatrixFactory matrixFactory) {

	    SerializationHelper helper = new SerializationHelper(
	    		PretrainedMnistWeights.class.getClassLoader(), "pretrainedweights");
		
		layer1Weights = new WeightsMatrixImpl(
				matrixFactory.createMatrixFromRowsByRowsArray(6, 81,
						helper.deserialize(float[].class, "layer1Weights")),
				new WeightsFormatImpl(
						Arrays.asList(Dimension.INPUT_DEPTH, Dimension.FILTER_HEIGHT, Dimension.FILTER_WIDTH),
						Arrays.asList(Dimension.OUTPUT_DEPTH), WeightsMatrixOrientation.ROWS_SPAN_OUTPUT_DIMENSIONS));

		layer1Biases = new BiasVectorImpl(
				matrixFactory.createMatrixFromRowsByRowsArray(6, 1, helper.deserialize(float[].class, "layer1Biases")), 
						new BiasFormatImpl(Dimension.OUTPUT_DEPTH, FeaturesVectorOrientation.COLUMN_VECTOR));

		layer3Weights = new WeightsMatrixImpl(
				matrixFactory.createMatrixFromRowsByRowsArray(400, 600,
						helper.deserialize(float[].class, "layer3Weights")),
				new WeightsFormatImpl(Arrays.asList(Dimension.INPUT_FEATURE), Arrays.asList(Dimension.OUTPUT_FEATURE), WeightsMatrixOrientation.ROWS_SPAN_OUTPUT_DIMENSIONS));

		layer3Biases = new BiasVectorImpl(matrixFactory.createMatrixFromRowsByRowsArray(400, 1,
				helper.deserialize(float[].class, "layer3Biases")), 
				FeaturesVectorFormat.DEFAULT_BIAS_FORMAT);

		layer4Weights = new WeightsMatrixImpl(
				matrixFactory.createMatrixFromRowsByRowsArray(100, 400,
						helper.deserialize(float[].class, "layer4Weights")),
				new WeightsFormatImpl(Arrays.asList(Dimension.INPUT_FEATURE), Arrays.asList(Dimension.OUTPUT_FEATURE), 
						WeightsMatrixOrientation.ROWS_SPAN_OUTPUT_DIMENSIONS));

		layer4Biases = new BiasVectorImpl(matrixFactory.createMatrixFromRowsByRowsArray(100, 1,
				helper.deserialize(float[].class, "layer4Biases")),
				FeaturesVectorFormat.DEFAULT_BIAS_FORMAT);

		layer5Weights = new WeightsMatrixImpl(
				matrixFactory.createMatrixFromRowsByRowsArray(10, 100,
						helper.deserialize(float[].class, "layer5Weights")),
				new WeightsFormatImpl(Arrays.asList(Dimension.INPUT_FEATURE), Arrays.asList(Dimension.OUTPUT_FEATURE), WeightsMatrixOrientation.ROWS_SPAN_OUTPUT_DIMENSIONS));

		layer5Biases = new BiasVectorImpl(matrixFactory.createMatrixFromRowsByRowsArray(10, 1,
				helper.deserialize(float[].class, "layer5Biases")), FeaturesVectorFormat.DEFAULT_BIAS_FORMAT);
	}

	public WeightsMatrix getLayer1Weights() {
		return layer1Weights;
	}

	public BiasVector getLayer1Biases() {
		return layer1Biases;
	}

	public WeightsMatrix getLayer3Weights() {
		return layer3Weights;
	}

	public BiasVector getLayer3Biases() {
		return layer3Biases;
	}

	public WeightsMatrix getLayer4Weights() {
		return layer4Weights;
	}

	public BiasVector getLayer4Biases() {
		return layer4Biases;
	}

	public WeightsMatrix getLayer5Weights() {
		return layer5Weights;
	}

	public BiasVector getLayer5Biases() {
		return layer5Biases;
	}

}
