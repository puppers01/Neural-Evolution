class ActivationFunction {
  constructor(func, dfunc) {
    this.func = func;
    this.dfunc = dfunc;
  }
}

let sigmoid = new ActivationFunction(
  x => 1 / (1 + Math.exp(-x)),
  y => y * (1 - y)
);

let tanh = new ActivationFunction(
  x => Math.tanh(x),
  y => 1 - (y * y)
);

class NeuralNetwork {
  constructor(inputNodes, hiddenNodes, outputNodes) {
    if (inputNodes instanceof NeuralNetwork) {
      let a = inputNodes;
      this.inputNodes = a.inputNodes;
      this.hiddenNodes = a.hiddenNodes;
      this.outputNodes = a.outputNodes;

      this.weightsInHi = a.weightsInHi.copy();
      this.weightsHiOu = a.weightsHiOu.copy();
    } else {
      this.inputNodes = inputNodes;
      this.hiddenNodes = hiddenNodes;
      this.outputNodes = outputNodes;

      this.weightsInHi = new Matrix(this.hiddenNodes, this.inputNodes);
      this.weightsHiOu = new Matrix(this.outputNodes, this.hiddenNodes);
      this.weightsInHi.randomize();
      this.weightsHiOu.randomize();
    }

    this.setLearningRate();
    this.setActivationFunction();
  }

  predict(input_array) {
    let inputs = Matrix.fromArray(input_array);
    let hidden = Matrix.multiply(this.weightsInHi, inputs);
    hidden.map(this.activation_function.func);

    let output = Matrix.multiply(this.weightsHiOu, hidden);
    output.map(this.activation_function.func);

    return output.toArray();
  }

  setLearningRate(learning_rate = 0.1) {
    this.learning_rate = learning_rate;
  }

  setActivationFunction(func = sigmoid) {
    this.activation_function = func;
  }

  train(input_array, target_array) {
    let inputs = Matrix.fromArray(input_array);
    let hidden = Matrix.multiply(this.weightsInHi, inputs);
    hidden.map(this.activation_function.func);

    let outputs = Matrix.multiply(this.weightsHiOu, hidden);
    outputs.map(this.activation_function.func);

    let targets = Matrix.fromArray(target_array);

    let outputErrors = Matrix.subtract(targets, outputs);

    let gradients = Matrix.map(outputs, this.activation_function.dfunc);
    gradients.multiply(outputErrors);
    gradients.multiply(this.learning_rate);

    let hiddenT = Matrix.transpose(hidden);
    let weightsHiOuDeltas = Matrix.multiply(gradients, hiddenT);

    this.weightsHiOu.add(weightsHiOuDeltas);

    let whoT = Matrix.transpose(this.weightsHiOu);
    let hiddenErrors = Matrix.multiply(whoT, outputErrors);

    let hiddenGradient = Matrix.map(hidden, this.activation_function.dfunc);
    hiddenGradient.multiply(hiddenErrors);
    hiddenGradient.multiply(this.learning_rate);

    let inputsT = Matrix.transpose(inputs);
    let weightInHiDeltas = Matrix.multiply(hiddenGradient, inputsT);

    this.weightsInHi.add(weightInHiDeltas);
  }

  serialize() {
    return JSON.stringify(this);
  }

  static deserialize(data) {
    if (typeof data == 'string') {
      data = JSON.parse(data);
    }
    let nn = new NeuralNetwork(data.inputNodes, data.hiddenNodes, data.outputNodes);
    nn.weightsInHi = Matrix.deserialize(data.weightsInHi);
    nn.weightsHiOu = Matrix.deserialize(data.weightsHiOu);
    nn.learning_rate = data.learning_rate;
    return nn;
  }

  copy() {
    return new NeuralNetwork(this);
  }

  mutate(rate) {
    function mutate(val) {
      if (Math.random() < rate) {
        return val + randomGaussian(0, 0.1);
      } else {
        return val;
      }
    }
    this.weightsInHi.map(mutate);
    this.weightsHiOu.map(mutate);
  }
}