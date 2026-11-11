# Third-party notices

## MNIST ONNX model (`models/mnist-12.onnx`)

- Source: ONNX Model Zoo — https://github.com/onnx/models
  - Path: `validated/vision/classification/mnist/model/mnist-12.onnx`
  - Model card: https://github.com/onnx/models/blob/main/validated/vision/classification/mnist/README.md
  - SHA-256: `5c688690f8bacf667d4c2074af5ad0646ca328d7ab03eccf944a65b320171bdd` (same as the Git LFS object registered in the Model Zoo)
- The model card states: "The model is trained in CNTK following the tutorial CNTK 103D: Convolutional Neural Network with MNIST" and is trained on the MNIST dataset.
- License: the model card is marked `SPDX-License-Identifier: MIT` and its License section says "MIT".
  The model card does not name a specific copyright holder; credit goes to the ONNX Model Zoo contributors.

MIT License text:

```
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Python packages

Installed from PyPI at run time (not redistributed in this repository) under their own licenses,
including Streamlit, streamlit-drawable-canvas, ONNX Runtime, NumPy, pandas, Pillow and Altair.
