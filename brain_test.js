var brain = require('brain.js');
//var net = new brain.NeuralNetwork();

var net = new brain.NeuralNetwork({
  activation: 'sigmoid', // activation function
  hiddenLayers: [3],
  learningRate: 0.6 // global learning rate, useful when training using streams
});


var data=[
  {input: { r: 0.03, g: 0.7, b: 0.5 }, output: { black: 1 }},
  {input: { r: 0.16, g: 0.09, b: 0.2 }, output: { white: 1 }},
  {input: { r: 0.5, g: 0.5, b: 1.0 }, output: { white: 1 }}
];


net.train(data, {
  errorThresh: 0.005,  // error threshold to reach
  iterations: 20000,   // maximum training iterations
  log: true,           // console.log() progress periodically
  logPeriod: 100,       // number of iterations between logging
  learningRate: 0.3    // learning rate
});
var json = net.toJSON();

var output = net.run({ r: 1, g: 0.4, b: 0 });  // { white: 0.99, black: 0.002 }
console.log(output);

var output = net.run({ r: 0.5, g: 0.4, b: 0.2 });  // { white: 0.99, black: 0.002 }
console.log(output);

console.log(json);
