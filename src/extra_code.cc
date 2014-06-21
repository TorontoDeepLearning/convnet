DataHandler* DataHandler::ChooseDataHandler(const string& config_file) {
  config::DatasetConfig config;
  ReadDataConfig(config_file, &config);

  DataHandler* dh;
  switch(config.dataset_type()) {
    case config::DatasetConfig::DUMMY:
            dh = new DummyDataHandler(config);
            break;
    case config::DatasetConfig::HDF5:
            dh = new SimpleHDF5DataHandler(config);
            break;
    case config::DatasetConfig::BIG_HDF5:
            dh = new BigSimpleHDF5DataHandler(config);
            break;
    case config::DatasetConfig::IMAGE_HDF5:
            dh = new HDF5DataHandler(config);
            break;
    case config::DatasetConfig::IMAGE_HDF5_MULT_POS:
            dh = new HDF5MultiplePositionDataHandler(config);
            break;
    case config::DatasetConfig::IMAGENET_CLS_HDF5:
            dh = new ImageNetCLSDataHandler(config);
            break;
    case config::DatasetConfig::IMAGENET_CLS_HDF5_MULT_POS:
            dh = new ImageNetCLSMultiplePosDataHandler(config);
            break;
    case config::DatasetConfig::RAW_IMAGE:
            dh = new RawImageDataHandler<unsigned char>(config);
            break;
    case config::DatasetConfig::RAW_IMAGE_SLIDING_WINDOW:
            dh = new SlidingWindowDataHandler(config);
            break;
    default:
            cerr << "Error: Unknown dataset type. " << endl;
            exit(1);
  }
  return dh;
}
