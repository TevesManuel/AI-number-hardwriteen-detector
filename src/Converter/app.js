const tf = require('@tensorflow/tfjs-node');
const tfconv = require('@tensorflow/tfjs-converter');
const path = require('path');

const modelPath = './ModelNumbers.h5';

fn = async () => {
    console.log("Loading...");
    const model = await tf.loadLayersModel(path.join(__dirname, modelPath));
    console.log("Loaded.");
    // const convertedModel = await tfconv.convert(model);
    // const jsonData = JSON.stringify(convertedModel);
    // fs.writeFileSync('./tfjs/model.json', jsonData);
}
fn()