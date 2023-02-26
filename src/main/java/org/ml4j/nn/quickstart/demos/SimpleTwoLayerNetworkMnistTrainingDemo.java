package org.ml4j.nn.quickstart.demos;

import static org.ml4j.nn.activationfunctions.ActivationFunctionBaseType.SIGMOID;
import static org.ml4j.nn.activationfunctions.ActivationFunctionBaseType.SOFTMAX;

import java.util.logging.LogManager;
import java.util.stream.IntStream;

import org.ml4j.jblas.JBlasRowMajorMatrixFactoryOptimised;
import org.ml4j.nd4j.Nd4jRowMajorMatrixFactory;
import org.ml4j.nn.LayeredFeedForwardNeuralNetworkContext;
import org.ml4j.nn.axons.AxonsContext;
import org.ml4j.nn.demo.util.MnistData;
import org.ml4j.nn.layers.FullyConnectedFeedForwardLayer;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.quickstart.sessions.factories.QuickstartSessionFactory;
import org.ml4j.nn.sessions.DefaultSession;
import org.ml4j.nn.supervised.LayeredSupervisedFeedForwardNeuralNetwork;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Demo of a simple 2 layer fully connected LayeredSupervisedFeedForwardNeuralNetwork for Mnist Data.
 * 
 * @author Michael Lavelle
 */
public class SimpleTwoLayerNetworkMnistTrainingDemo {
	
	static {
		// Quieten Logging for JBlas
		org.jblas.util.Logger.getLogger().setLevel(org.jblas.util.Logger.ERROR);
		LogManager.getLogManager().reset();
	}

	private static final Logger LOGGER = LoggerFactory.getLogger(SimpleTwoLayerNetworkMnistTrainingDemo.class);
	
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

		// Create a simple two layer fully-connected neural network.
		LayeredSupervisedFeedForwardNeuralNetwork neuralNetwork = session
				.buildLayeredSupervisedNeuralNetwork("simpleTwoLayerNetwork")
					.withFullyConnectedLayer("firstLayer")
						.withInputNeurons(new Neurons(784, withBiasUnit(true)))						
						.withAxonsContextConfigurer(context -> context.withLeftHandInputDropoutKeepProbability(1))
						.withOutputNeurons(new Neurons(400, withBiasUnit(false)))
						.withActivationFunction(SIGMOID)
					.withFullyConnectedLayer("secondLayer").withInputNeurons(new Neurons(400, withBiasUnit(true)))
						.withOutputNeurons(new Neurons(10, withBiasUnit(false)))
						.withActivationFunction(SOFTMAX)
						.build();


		// 2. CREATE TRAINNG DATA AND TRAINING PARAMETERS
		
		// Obtain the training data and labels - for this demo the data is small enough to be loaded in memory.
		// We wrap the numeric data inside NeuronsActivation instances.
		
		NeuronsActivation trainingDataActivations = MnistData.loadTrainingData(session.getMatrixFactory());
		NeuronsActivation trainingLabelActivations = MnistData.loadTrainingLabels(session.getMatrixFactory());
		
		// Create a context for the neural network ( non-training, for classification/showcase purposes)
		LayeredFeedForwardNeuralNetworkContext neuralNetworkContext = session.createLayeredSupervisedFeedForwardNeuralNetworkContext();


		// Obtain the accuracy of the initial (untrained ) network, using the classification context.
		float preTrainingAccuracy = neuralNetwork.getClassificationAccuracy(trainingDataActivations,
				trainingLabelActivations, neuralNetworkContext);
		
		LOGGER.info("Pre-training training-set accuracy = {} %", preTrainingAccuracy);
	 
		// 3. TRAIN NETWORK
		
		// Create a training context
		LayeredFeedForwardNeuralNetworkContext trainingNeuralNetworkContext = neuralNetworkContext.asTrainingContext();
		
		// Configure training properties
		trainingNeuralNetworkContext.setTrainingEpochs(400);
		trainingNeuralNetworkContext.setTrainingLearningRate(0.1f);
			
		// Optionally configure hyper-parameters, such as regularisation and dropout for a given layer.
		FullyConnectedFeedForwardLayer secondLayer = FullyConnectedFeedForwardLayer.class.cast(neuralNetwork.getLayer(1));
		AxonsContext secondLayerAxonsContext = secondLayer.getPrimaryAxonsContext(trainingNeuralNetworkContext.getDirectedComponentsContext());
		secondLayerAxonsContext.withRegularisationLambda(0).withLeftHandInputDropoutKeepProbability(0.8f);
				
		// Train the neural network, using the training context.
				
		neuralNetwork.train(trainingDataActivations, trainingLabelActivations, trainingNeuralNetworkContext);

		// Obtain the accuracy of the trained network on the training set, using the classification context.
		float trainingSetAccuracy = neuralNetwork.getClassificationAccuracy(trainingDataActivations,
				trainingLabelActivations, neuralNetworkContext);
		
		LOGGER.info("Post-training training-set accuracy = {} %", trainingSetAccuracy);
		
		// 4. USE NETWORK

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
