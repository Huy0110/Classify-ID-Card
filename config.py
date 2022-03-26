class CFG:

  epochs =5                             # No. of epochs for training the model
  lr = 0.001                              # Learning rate
  batch_size = 16                     # Batch Size for Dataset

  model_name = 'tf_efficientnet_b4_ns'    # Model name (we are going to import model from timm)
  img_size = 224                          # Resize all the images to be 224 by 224

  # going to be used for loading dataset
  train_path='train'
  validate_path='val'
  test_path='test'