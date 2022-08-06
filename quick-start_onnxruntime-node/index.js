// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

const ort = require('onnxruntime-node');

// use an async context to call onnxruntime functions.
async function main() {
    try {
        // create a new session and load the specific model.
        //
        // the model in this example contains a single MatMul node
        // it has 2 inputs: 'a'(float32, 3x4) and 'b'(float32, 4x3)
        // it has 1 output: 'c'(float32, 3x3)
        const session = await ort.InferenceSession.create('../model/tf_model.onnx');

        // prepare inputs. a tensor need its corresponding TypedArray as data
        const dataA = Float32Array.from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
        const dataB = Float32Array.from([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]);
        const tensorA = new ort.Tensor('float32', dataA, [3, 4]);
        const tensorB = new ort.Tensor('float32', dataB, [4, 3]);
        const dataTest = Float32Array.from([6.7, 3.1, 4.4, 1.4]);
        const tensorTest = new ort.Tensor('float32', dataTest, [1, 4]);

        /// prepare feeds. use model input names as keys.
        const feeds = {'tensor_input':tensorTest}

        // feed inputs and run
        const results = await session.run(feeds);

        // read from results
        const output = results.tensor_output.data;
        console.log(`data of result tensor output: ${output}`);

    } catch (e) {
        console.error(`failed to inference ONNX model: ${e}.`);
    }
}

main();
