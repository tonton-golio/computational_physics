"use client";
import React, { useState, useCallback, useEffect, useMemo } from 'react';
import Plot from 'react-plotly.js';

const AdvancedDeepLearningPage: React.FC = () => {
  // Perceptron demo
  const [perceptronParams, setPerceptronParams] = useState({
    w1: 1.0,
    w2: 1.0,
    bias: -1.5,
    gate: 'AND'
  });

  // Training demo
  const [weights, setWeights] = useState([0.1, 0.1]);
  const [biasTrain, setBiasTrain] = useState(0.0);
  const [epoch, setEpoch] = useState(0);
  const [lossHistory, setLossHistory] = useState<number[]>([]);
  const [training, setTraining] = useState(false);

  // Generate training data for classification
  const [trainingData, setTrainingData] = useState<Array<{x: number, y: number, label: number}>>([]);

  useEffect(() => {
    const data = [];
    for (let i = 0; i < 20; i++) {
      const x1 = Math.random() * 2 - 1;
      const x2 = Math.random() * 2 - 1;
      const label = (x1 + x2 > 0) ? 1 : 0; // simple linear separation
      data.push({x: x1, y: x2, label});
    }
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setTrainingData(data);
  }, []);

  // Perceptron decision function
  const perceptronOutput = (x1: number, x2: number) => {
    const { w1, w2, bias } = perceptronParams;
    const sum = w1 * x1 + w2 * x2 + bias;
    return sum > 0 ? 1 : 0;
  };

  // Get gate data
  const getGateData = () => {
    const points = [];
    for (let x1 = 0; x1 <= 1; x1++) {
      for (let x2 = 0; x2 <= 1; x2++) {
        let expected = 0;
        if (perceptronParams.gate === 'AND') expected = x1 && x2 ? 1 : 0;
        else if (perceptronParams.gate === 'OR') expected = x1 || x2 ? 1 : 0;
        else if (perceptronParams.gate === 'XOR') expected = (x1 !== x2) ? 1 : 0;
        points.push({x1, x2, expected, actual: perceptronOutput(x1, x2)});
      }
    }
    return points;
  };

  // Perceptron plot
  const gateData = getGateData();
  const perceptronPlotData = [
    {
      type: 'scatter' as const,
      x: gateData.map(p => p.x1),
      y: gateData.map(p => p.x2),
      mode: 'markers' as const,
      marker: {
        color: gateData.map(p => p.actual === p.expected ? 'green' : 'red'),
        size: 10,
        symbol: 'circle' as const
      },
      name: 'Inputs'
    }
  ];

  const perceptronLayout = {
    title: { text: `${perceptronParams.gate} Gate Classification` },
    xaxis: { title: { text: 'x1' }, range: [-0.1, 1.1] },
    yaxis: { title: { text: 'x2' }, range: [-0.1, 1.1] },
    width: 500,
    height: 400
  };

  // Activation functions
  const activationData = [
    {
      type: 'scatter' as const,
      x: Array.from({length: 100}, (_, i) => -5 + i * 0.1),
      y: Array.from({length: 100}, (_, i) => {
        const x = -5 + i * 0.1;
        return 1 / (1 + Math.exp(-x)); // sigmoid
      }),
      mode: 'lines' as const,
      name: 'Sigmoid'
    },
    {
      type: 'scatter' as const,
      x: Array.from({length: 100}, (_, i) => -5 + i * 0.1),
      y: Array.from({length: 100}, (_, i) => {
        const x = -5 + i * 0.1;
        return Math.tanh(x); // tanh
      }),
      mode: 'lines' as const,
      name: 'Tanh'
    },
    {
      type: 'scatter' as const,
      x: Array.from({length: 100}, (_, i) => -5 + i * 0.1),
      y: Array.from({length: 100}, (_, i) => {
        const x = -5 + i * 0.1;
        return Math.max(0, x); // ReLU
      }),
      mode: 'lines' as const,
      name: 'ReLU'
    }
  ];

  const activationLayout = {
    title: { text: 'Activation Functions' },
    xaxis: { title: { text: 'Input' } },
    yaxis: { title: { text: 'Output' } },
    width: 500,
    height: 400
  };

  // Training demo
  const trainStep = useCallback(() => {
    if (!training) return;
    const newWeights = [...weights];
    let newBias = biasTrain;
    let totalLoss = 0;
    for (const point of trainingData) {
      const prediction = weights[0] * point.x + weights[1] * point.y + biasTrain > 0 ? 1 : 0;
      const error = point.label - prediction;
      totalLoss += Math.abs(error);
      newWeights[0] += 0.01 * error * point.x;
      newWeights[1] += 0.01 * error * point.y;
      newBias += 0.01 * error;
    }
    setWeights(newWeights);
    setBiasTrain(newBias);
    setLossHistory(prev => [...prev, totalLoss / trainingData.length]);
    setEpoch(prev => prev + 1);
    if (epoch > 100) setTraining(false);
  }, [training, weights, biasTrain, trainingData, epoch]);

  useEffect(() => {
    if (training) {
      const timer = setTimeout(trainStep, 100);
      return () => clearTimeout(timer);
    }
  }, [training, trainStep]);

  const decisionBoundaryX = [-1, 1];
  const decisionBoundaryY = decisionBoundaryX.map(x => -(weights[0] * x + biasTrain) / weights[1]);

  const trainingPlotData = [
    {
      type: 'scatter' as const,
      x: trainingData.filter(p => p.label === 1).map(p => p.x),
      y: trainingData.filter(p => p.label === 1).map(p => p.y),
      mode: 'markers' as const,
      marker: { color: 'blue', size: 8 },
      name: 'Class 1'
    },
    {
      type: 'scatter' as const,
      x: trainingData.filter(p => p.label === 0).map(p => p.x),
      y: trainingData.filter(p => p.label === 0).map(p => p.y),
      mode: 'markers' as const,
      marker: { color: 'red', size: 8 },
      name: 'Class 0'
    },
    {
      type: 'scatter' as const,
      x: decisionBoundaryX,
      y: decisionBoundaryY,
      mode: 'lines' as const,
      line: { color: 'black', width: 2 },
      name: 'Decision Boundary'
    }
  ];

  const trainingLayout = {
    title: { text: 'Perceptron Training Demo' },
    xaxis: { title: { text: 'x1' }, range: [-1.1, 1.1] },
    yaxis: { title: { text: 'x2' }, range: [-1.1, 1.1] },
    width: 500,
    height: 400
  };

  const lossPlotData = [
    {
      type: 'scatter' as const,
      x: Array.from({length: lossHistory.length}, (_, i) => i),
      y: lossHistory,
      mode: 'lines' as const,
      name: 'Loss'
    }
  ];

  const lossLayout = {
    title: { text: 'Training Loss' },
    xaxis: { title: { text: 'Epoch' } },
    yaxis: { title: { text: 'Loss' } },
    width: 500,
    height: 300
  };

  const updatePerceptronParam = (key: string) => (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const value = key === 'gate' ? e.target.value : parseFloat(e.target.value);
    setPerceptronParams({ ...perceptronParams, [key]: value });
  };

  return (
    <div className="p-8 max-w-7xl mx-auto">
      <h1 className="text-4xl font-bold mb-8">Advanced Deep Learning</h1>
      <p className="mb-8 text-lg">
        Neural networks are powerful models inspired by the brain. They consist of layers of neurons connected by weights.
        Training involves adjusting these weights to minimize error through backpropagation.
      </p>

      <h2 className="text-3xl font-semibold mb-4">1. The Perceptron</h2>
      <p className="mb-4">
        A perceptron is a single neuron that computes a weighted sum of inputs plus bias, then applies a step function.
        Output = 1 if ∑(w_i * x_i) + b {'>'} 0, else 0.
      </p>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
        <div>
          <Plot data={perceptronPlotData} layout={perceptronLayout} config={{displayModeBar: false}} />
        </div>
        <div className="bg-gray-100 p-6 rounded-lg">
          <h3 className="text-xl font-semibold mb-4">Parameters</h3>
          <div className="grid grid-cols-1 gap-4">
            <div>
              <label className="block">Gate: </label>
              <select value={perceptronParams.gate} onChange={updatePerceptronParam('gate')} className="w-full p-2 border rounded">
                <option value="AND">AND</option>
                <option value="OR">OR</option>
                <option value="XOR">XOR</option>
              </select>
            </div>
            <div>
              <label>w1: {perceptronParams.w1.toFixed(2)}</label>
              <input type="range" min={-2} max={2} step={0.1} value={perceptronParams.w1} onChange={updatePerceptronParam('w1')} className="w-full" />
            </div>
            <div>
              <label>w2: {perceptronParams.w2.toFixed(2)}</label>
              <input type="range" min={-2} max={2} step={0.1} value={perceptronParams.w2} onChange={updatePerceptronParam('w2')} className="w-full" />
            </div>
            <div>
              <label>Bias: {perceptronParams.bias.toFixed(2)}</label>
              <input type="range" min={-3} max={3} step={0.1} value={perceptronParams.bias} onChange={updatePerceptronParam('bias')} className="w-full" />
            </div>
          </div>
        </div>
      </div>

      <h2 className="text-3xl font-semibold mb-4">2. Activation Functions</h2>
      <p className="mb-4">
        Activation functions introduce nonlinearity. Common ones: Sigmoid (0-1), Tanh (-1-1), ReLU (max(0,x)).
      </p>
      <div className="mb-8">
        <Plot data={activationData} layout={activationLayout} config={{displayModeBar: false}} />
      </div>

      <h2 className="text-3xl font-semibold mb-4">3. Backpropagation</h2>
      <p className="mb-4">
        Backpropagation computes gradients of the loss with respect to weights by applying the chain rule backwards.
        For a simple network: loss to output to hidden to input.
      </p>
      <p className="mb-8">
        Example: For MSE loss and sigmoid activation, dL/dw = (y_pred - y_true) * sigmoid_derivative(net) * x
      </p>

      <h2 className="text-3xl font-semibold mb-4">4. Simple Training Demo</h2>
      <p className="mb-4">
        Train a perceptron to classify points above/below the line x1 + x2 = 0.
      </p>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
        <div>
          <Plot data={trainingPlotData} layout={trainingLayout} config={{displayModeBar: false}} />
        </div>
        <div>
          <Plot data={lossPlotData} layout={lossLayout} config={{displayModeBar: false}} />
        </div>
      </div>
      <div className="flex gap-4 mb-8">
        <button onClick={() => setTraining(!training)} className={`px-4 py-2 ${training ? 'bg-red-500' : 'bg-green-500'} text-white rounded`}>
          {training ? 'Stop' : 'Start'} Training
        </button>
        <button onClick={() => { setWeights([0.1, 0.1]); setBiasTrain(0); setLossHistory([]); setEpoch(0); }} className="px-4 py-2 bg-blue-500 text-white rounded">
          Reset
        </button>
      </div>
      <p>Epoch: {epoch}, Loss: {lossHistory[lossHistory.length - 1]?.toFixed(3) || 0}</p>
    </div>
  );
};

export default AdvancedDeepLearningPage;