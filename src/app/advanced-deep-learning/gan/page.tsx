"use client";
import React, { useState, useCallback, useEffect } from 'react';
import Plot from 'react-plotly.js';

const GANDemo: React.FC = () => {
  const [epoch, setEpoch] = useState(0);
  const [training, setTraining] = useState(false);
  const [generatorWeights, setGeneratorWeights] = useState([0.1, 0.1]);
  const [discriminatorWeights, setDiscriminatorWeights] = useState([0.5, 0.5]);
  const [generatorBias, setGeneratorBias] = useState(0.0);
  const [discriminatorBias, setDiscriminatorBias] = useState(0.0);
  const [realData] = useState(() => {
    // Generate real data: mixture of two Gaussians
    const data = [];
    for (let i = 0; i < 100; i++) {
      const rand = Math.random();
      let mean, std;
      if (rand < 0.5) {
        mean = -2; std = 0.5;
      } else {
        mean = 2; std = 0.5;
      }
      const x = mean + std * (Math.random() + Math.random() + Math.random() + Math.random() + Math.random() + Math.random() - 3) / Math.sqrt(2);
      data.push(x);
    }
    return data;
  });
  const [fakeData, setFakeData] = useState<number[]>([]);
  const [genLossHistory, setGenLossHistory] = useState<number[]>([]);
  const [discLossHistory, setDiscLossHistory] = useState<number[]>([]);

  // Simple generator: linear transformation of noise
  const generateFakeData = useCallback((numSamples: number = 100) => {
    const fake = [];
    for (let i = 0; i < numSamples; i++) {
      const noise = (Math.random() - 0.5) * 4; // noise in [-2, 2]
      const generated = generatorWeights[0] * noise + generatorWeights[1] * noise * noise + generatorBias;
      fake.push(generated);
    }
    return fake;
  }, [generatorWeights, generatorBias]);

  // Simple discriminator: perceptron
  const discriminator = useCallback((x: number) => {
    const net = discriminatorWeights[0] * x + discriminatorWeights[1] * x * x + discriminatorBias;
    return 1 / (1 + Math.exp(-net)); // sigmoid
  }, [discriminatorWeights, discriminatorBias]);

  // Training step
  const trainStep = useCallback(() => {
    if (!training) return;

    // Generate fake data
    const currentFakeData = generateFakeData(50);
    setFakeData(currentFakeData);

    // Train discriminator
    let discLoss = 0;
    const discGradW = [0, 0];
    let discGradB = 0;

    // Real data: target 1
    realData.forEach(x => {
      const pred = discriminator(x);
      const error = 1 - pred;
      discLoss += error * error;
      const dPred = pred * (1 - pred);
      discGradW[0] += -2 * error * dPred * x;
      discGradW[1] += -2 * error * dPred * x * x;
      discGradB += -2 * error * dPred;
    });

    // Fake data: target 0
    currentFakeData.forEach(x => {
      const pred = discriminator(x);
      const error = 0 - pred;
      discLoss += error * error;
      const dPred = pred * (1 - pred);
      discGradW[0] += -2 * error * dPred * x;
      discGradW[1] += -2 * error * dPred * x * x;
      discGradB += -2 * error * dPred;
    });

    // Update discriminator
    const learningRate = 0.01;
    setDiscriminatorWeights(prev => [
      prev[0] - learningRate * discGradW[0] / (realData.length + currentFakeData.length),
      prev[1] - learningRate * discGradW[1] / (realData.length + currentFakeData.length)
    ]);
    setDiscriminatorBias(prev => prev - learningRate * discGradB / (realData.length + currentFakeData.length));

    // Train generator
    let genLoss = 0;
    const genGradW = [0, 0];
    let genGradB = 0;

    currentFakeData.forEach(x => {
      const pred = discriminator(x);
      const error = 1 - pred; // generator wants discriminator to think it's real
      genLoss += error * error;
      const dPred = pred * (1 - pred);
      const dGen = 2 * discriminatorWeights[0] * x + 2 * discriminatorWeights[1] * x * x; // derivative through discriminator
      genGradW[0] += -2 * error * dPred * dGen * x;
      genGradW[1] += -2 * error * dPred * dGen * x * x;
      genGradB += -2 * error * dPred * dGen;
    });

    // Update generator
    setGeneratorWeights(prev => [
      prev[0] - learningRate * genGradW[0] / currentFakeData.length,
      prev[1] - learningRate * genGradW[1] / currentFakeData.length
    ]);
    setGeneratorBias(prev => prev - learningRate * genGradB / currentFakeData.length);

    // Update history
    setDiscLossHistory(prev => [...prev, discLoss / (realData.length + currentFakeData.length)]);
    setGenLossHistory(prev => [...prev, genLoss / currentFakeData.length]);

    setEpoch(prev => prev + 1);
  }, [training, realData, generateFakeData, discriminator]);

  useEffect(() => {
    if (training) {
      const timer = setTimeout(trainStep, 200);
      return () => clearTimeout(timer);
    }
  }, [training, trainStep]);

  // Initial fake data
  useEffect(() => {
    setFakeData(generateFakeData());
  }, [generateFakeData]);

  // Plot data
  const histogramData = [
    {
      type: 'histogram' as const,
      x: realData,
      name: 'Real Data',
      opacity: 0.7,
      nbinsx: 20,
      marker: { color: 'blue' }
    },
    {
      type: 'histogram' as const,
      x: fakeData,
      name: 'Generated Data',
      opacity: 0.7,
      nbinsx: 20,
      marker: { color: 'red' }
    }
  ];

  const histogramLayout = {
    title: { text: 'Data Distributions' },
    xaxis: { title: { text: 'Value' } },
    yaxis: { title: { text: 'Frequency' } },
    barmode: 'overlay' as const,
    width: 600,
    height: 400
  };

  const lossData = [
    {
      type: 'scatter' as const,
      x: Array.from({length: genLossHistory.length}, (_, i) => i),
      y: genLossHistory,
      mode: 'lines' as const,
      name: 'Generator Loss',
      line: { color: 'red' }
    },
    {
      type: 'scatter' as const,
      x: Array.from({length: discLossHistory.length}, (_, i) => i),
      y: discLossHistory,
      mode: 'lines' as const,
      name: 'Discriminator Loss',
      line: { color: 'blue' }
    }
  ];

  const lossLayout = {
    title: { text: 'Training Losses' },
    xaxis: { title: { text: 'Epoch' } },
    yaxis: { title: { text: 'Loss' } },
    width: 600,
    height: 400
  };

  return (
    <div className="p-8 max-w-7xl mx-auto">
      <h1 className="text-4xl font-bold mb-8">Generative Adversarial Networks (GAN) Demo</h1>
      <p className="mb-8 text-lg">
        This demo shows a simple GAN training to generate data similar to a mixture of two Gaussians.
        The generator creates fake data from noise, while the discriminator tries to tell real from fake.
        They are trained adversarially.
      </p>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
        <div>
          <Plot data={histogramData} layout={histogramLayout} config={{displayModeBar: false}} />
        </div>
        <div>
          <Plot data={lossData} layout={lossLayout} config={{displayModeBar: false}} />
        </div>
      </div>

      <div className="flex gap-4 mb-8">
        <button
          onClick={() => setTraining(!training)}
          className={`px-6 py-3 text-white rounded-lg font-medium ${
            training ? 'bg-red-500 hover:bg-red-600' : 'bg-green-500 hover:bg-green-600'
          }`}
        >
          {training ? 'Stop Training' : 'Start Training'}
        </button>
        <button
          onClick={() => {
            setEpoch(0);
            setGeneratorWeights([0.1, 0.1]);
            setDiscriminatorWeights([0.5, 0.5]);
            setGeneratorBias(0.0);
            setDiscriminatorBias(0.0);
            setGenLossHistory([]);
            setDiscLossHistory([]);
            setFakeData(generateFakeData());
          }}
          className="px-6 py-3 bg-blue-500 hover:bg-blue-600 text-white rounded-lg font-medium"
        >
          Reset
        </button>
      </div>

      <div className="bg-gray-100 p-6 rounded-lg mb-8">
        <h3 className="text-xl font-semibold mb-4">Training Status</h3>
        <p>Epoch: {epoch}</p>
        <p>Generator Loss: {genLossHistory[genLossHistory.length - 1]?.toFixed(4) || 'N/A'}</p>
        <p>Discriminator Loss: {discLossHistory[discLossHistory.length - 1]?.toFixed(4) || 'N/A'}</p>
      </div>

      <div className="prose max-w-none">
        <h2>How it Works</h2>
        <p>
          The real data is a mixture of two Gaussian distributions centered at -2 and +2.
          The generator starts with random weights and tries to transform noise into similar data.
          The discriminator is a simple neural network that learns to classify real vs fake data.
        </p>
        <p>
          Training alternates between updating the discriminator (to better classify) and the generator (to fool the discriminator).
          Over time, the generated data distribution should match the real data distribution.
        </p>
      </div>
    </div>
  );
};

export default GANDemo;