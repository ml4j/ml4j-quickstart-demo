package org.ml4j.nn.quickstart.demos;


import static org.ml4j.nn.activationfunctions.ActivationFunctionBaseType.SIGMOID;
import static org.ml4j.nn.activationfunctions.ActivationFunctionBaseType.SOFTMAX;

import java.util.logging.LogManager;
import java.util.stream.IntStream;

import org.ml4j.jblas.JBlasRowMajorMatrixFactoryOptimised;
import org.ml4j.nd4j.Nd4jRowMajorMatrixFactory;
import org.ml4j.nn.LayeredFeedForwardNeuralNetworkContext;
import org.ml4j.nn.activationfunctions.ActivationFunctionBaseType;
import org.ml4j.nn.demo.util.MnistData;
import org.ml4j.nn.demo.util.PretrainedMnistWeights;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.Neurons3D;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.quickstart.sessions.factories.QuickstartSessionFactory;
import org.ml4j.nn.sessions.DefaultSession;
import org.ml4j.nn.supervised.LayeredSupervisedFeedForwardNeuralNetwork;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Demo of a simple pretrained 5-layer LayeredSupervisedFeedForwardNeuralNetwork including a Convolutional layer for Mnist Data.
 * 
 * @author Michael Lavelle
 */
public class PretrainedFiveLayerNetworkWithConvLayerMnistClassificationDemo {
	
	static {
		// Quieten Logging for JBlas
		org.jblas.util.Logger.getLogger().setLevel(org.jblas.util.Logger.ERROR);
		LogManager.getLogManager().reset();
	}

	private static final Logger LOGGER = LoggerFactory.getLogger(PretrainedFiveLayerNetworkWithConvLayerMnistClassificationDemo.class);
	
	public static void main(String[] args) {

		QuickstartSessionFactory sessionFactory = new QuickstartSessionFactory(new JBlasRowMajorMatrixFactoryOptimised(), false);
		runTrainNeuralNetworkDemo(sessionFactory);
	}

	private static void runTrainNeuralNetworkDemo(QuickstartSessionFactory sessionFactory) {
		
		// 1. CREATE NETWORK
		
		// Configure for training examples of length 784 ( grey-scale image vectors of length 28 * 28 * 1), with
		// label vectors of length 10 - indicating the assigned digit class.
				
		// Create session
		DefaultSession session = sessionFactory.createSession();
		
		// Obtain pretrained weights for this network architecture, from our Kaggle competion entry.
		PretrainedMnistWeights pretrainedWeights = new PretrainedMnistWeights(session.getMatrixFactory());
	
		// Create a pretrained five-layer neural network with a convolutional layer.
		LayeredSupervisedFeedForwardNeuralNetwork neuralNetwork = session
				.buildLayeredSupervised3DNeuralNetwork("pretrainedFiveLayerNetwork")
					.withConvolutionalLayer("firstLayer")
					.withInputNeurons(new Neurons3D(28, 28, 1, true)).withOutputNeurons(new Neurons3D(20, 20, 6, withBiasUnit(true)))
					.withConfig(config -> config.withFilterHeight(9).withFilterWidth(9).withFilterCount(6))
					.withWeightsMatrix(pretrainedWeights.getLayer1Weights())
					.withBiasVector(pretrainedWeights.getLayer1Biases())
					.withActivationFunction(ActivationFunctionBaseType.SIGMOID)
				.withMaxPoolingLayer("secondLayer")
					.withInputNeurons(new Neurons3D(20, 20, 6, false))
					.withScaleOutputs() // Not normally required, but this network was trained with this setting enabled
					.withConfig(config -> config.withStrideHeight(2).withStrideWidth(2)
					.withFilterHeight(2).withFilterWidth(2)
					.withOutputNeurons(new Neurons3D(10, 10, 6, false)))
				.withFullyConnectedLayer("thirdLayer")
					.withInputNeurons(new Neurons(600, withBiasUnit(true)))
					.withWeightsMatrix(pretrainedWeights.getLayer3Weights())
					.withBiasVector(pretrainedWeights.getLayer3Biases())
					.withOutputNeurons(new Neurons(400, withBiasUnit(false)))
					.withActivationFunction(SIGMOID)
				.withFullyConnectedLayer("fourthLayer")
					.withInputNeurons(new Neurons(400, withBiasUnit(true)))
					.withWeightsMatrix(pretrainedWeights.getLayer4Weights())
					.withBiasVector(pretrainedWeights.getLayer4Biases())
					.withOutputNeurons(new Neurons(100, withBiasUnit(false)))
					.withActivationFunction(SIGMOID)
				.withFullyConnectedLayer("fifthLayer")
					.withInputNeurons(new Neurons(100, withBiasUnit(true)))
					.withWeightsMatrix(pretrainedWeights.getLayer5Weights())
					.withBiasVector(pretrainedWeights.getLayer5Biases())
					.withOutputNeurons(new Neurons(10, withBiasUnit(false)))
				.withActivationFunction(SOFTMAX).build();

		// Create a context for the neural network ( non-training, for classification/showcase purposes)
		LayeredFeedForwardNeuralNetworkContext neuralNetworkContext = session.createLayeredSupervisedFeedForwardNeuralNetworkContext();

	
		// USE NETWORK

		NeuronsActivation testSetDataActivations = MnistData.loadTestSetData(session.getMatrixFactory());
		NeuronsActivation testSetLabelActivations = MnistData.loadTestSetLabels(session.getMatrixFactory());
		
		NeuronsActivation predictedLabelsActivation = neuralNetwork.forwardPropagate(testSetDataActivations, neuralNetworkContext).getOutput();
	
		int[] predictedDigits = predictedLabelsActivation.getActivations(session.getMatrixFactory()).columnArgmaxs();
		int[] actualDigits = testSetLabelActivations.getActivations(session.getMatrixFactory()).columnArgmaxs();
		
		LOGGER.info("Showcasing on previously unseen test set data...");

		// Output the first 100 predictions of the test set.
		IntStream.range(0, 100).forEach(exampleIndex -> {			
			LOGGER.info("Test-set example {} : Actual digit = {} , Predicted digit = {}", (exampleIndex + 1), actualDigits[exampleIndex], predictedDigits[exampleIndex]);
		});

		LOGGER.info("Calculating test set accuracy...");
	

		// Obtain the accuracy of the trained network on a test set of previously unseen data, using the classification context.
		float testSetAccuracy = neuralNetwork.getClassificationAccuracy(testSetDataActivations,
				testSetLabelActivations, neuralNetworkContext);
		
		LOGGER.info("Post-training test-set accuracy = {} %", testSetAccuracy);

	}

	/**
	 * Convenience method to improve readability of network creation.
	 * 
	 * @param withBiasUnit
	 * @return
	 */
	private static boolean withBiasUnit(boolean withBiasUnit) {
		return withBiasUnit;
	}
}
