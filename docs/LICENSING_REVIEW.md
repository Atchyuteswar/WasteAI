# Licensing review before publishing a project license

WasteAI currently has a public repository, but it does not declare a project-wide reuse license. Do not add an MIT (or other) license until the components and rights have been reviewed.

## Components to check

1. **Original application and notebook code**
   - Confirm that the code is original or that incorporated snippets are compatible with the license you intend to use.
   - If you own the original code and want permissive reuse, an MIT license may be an option for the original code only.

2. **Kaggle Garbage Classification dataset**
   - Review the current dataset page and any attached license or usage terms: https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification
   - A license for your code does not automatically relicense the dataset.
   - Do not redistribute the original images unless the dataset terms permit it.

3. **ImageNet-pretrained architectures and weights**
   - The Keras VGG16, VGG19 and MobileNetV2 application architectures use pretrained ImageNet weights.
   - Review the applicable model, pretrained-weight and dataset terms before redistributing trained `.keras` files that incorporate those weights.
   - Hosting files on Hugging Face does not itself establish redistribution permission.

4. **Dependencies**
   - TensorFlow, Keras, Streamlit, Pillow, NumPy, Pandas, Altair and other packages retain their own licenses. Listing dependencies does not transfer their licenses to this project.

## Suggested current position

Keep the repository public, but leave the project-wide `LICENSE` file absent until the review is complete. The README explains that public visibility is not the same as a reuse license.

If you later add an MIT license for your original source code, make its scope clear and do not imply that it covers third-party datasets or model assets.
