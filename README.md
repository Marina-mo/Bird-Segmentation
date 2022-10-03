## Birds Segmentation 

This app uses UNet to get segmenated image of a bird. Encoder was initialized with ResNet18 weight and decoder was trained on relevant dataset. Final verion achieved 0.9 IOU on test dataset and was saved in segmentation_model.pth file 

**User guide**: 

1) You can run model training and inference on test set with the following command:

    `Inference.py run_on_test -t True/False`

    -t flag indicates whether model should be trained or loaded from file

    You can find test images and segmentation masks in 00_test_val_input/test folder

2) You can get segmentation mask for your image with the following command:

    `Inference.py run_on_image -i path_to_image`

    You can find the segmentated image in the same folder with "_pred" postfix
