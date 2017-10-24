var synaptic = require('synaptic'); // this line is not needed in the browser

var Neuron = synaptic.Neuron,
	Layer = synaptic.Layer,
	Network = synaptic.Network,
	Trainer = synaptic.Trainer,
	Architect = synaptic.Architect;

    function Perceptron(input, hidden, output)
    {
        // create the layers
        var inputLayer = new Layer(input);
        var hiddenLayer = new Layer(hidden);
        var outputLayer = new Layer(output);
    
        // connect the layers
        inputLayer.project(hiddenLayer);
        hiddenLayer.project(outputLayer);
    
        // set the layers
        this.set({
            input: inputLayer,
            hidden: [hiddenLayer],
            output: outputLayer
        });
    }
    
    // extend the prototype chain
    Perceptron.prototype = new Network();
    Perceptron.prototype.constructor = Perceptron;

    var myPerceptron = new Perceptron(2,5,1);
    var myTrainer = new Trainer(myPerceptron);
    
    var trainingSet = [
        { input: [0,0], output: [0] },
        { input: [0,1], output: [1] },
        { input: [1,0], output: [1] },
        { input: [1,1], output: [0] }
      ]
      
      myTrainer.train(trainingSet);

      myTrainer.train(trainingSet,{
        rate: .3,
        iterations: 20000,
        error: .005,
        shuffle: true,
        log: 100,
        cost: Trainer.cost.CROSS_ENTROPY
    });

    //console.log(myTrainer);

    myPerceptron.activate([0,0]); // 0.0268581547421616
    myPerceptron.activate([1,0]); // 0.9829673642853368
    myPerceptron.activate([0,1]); // 0.9831714267395621
    myPerceptron.activate([1,1]); // 0.02128894618097928

    console.log(myPerceptron.activate([0,0]));
    console.log(myPerceptron.activate([1,0]));
    console.log(myPerceptron.activate([0,1]));
    console.log(myPerceptron.activate([1,1]));
